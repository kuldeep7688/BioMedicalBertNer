import os
import json
import logging
import fire
import sys

import torch 
import torch.nn as nn

from biomedical_bert_ner.utils.utilities import train_epoch, eval_epoch
from biomedical_bert_ner.utils.utilities import load_and_cache_examples
from biomedical_bert_ner.utils.utilities import get_labels, count_parameters

from biomedical_bert_ner.models.models import *

from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Being used as {} \n".format(DEVICE))



def train_ner_model(
    model_config_path, data_dir,
    logger_file_dir=None, labels_file=None
):
    # loading model config path
    if os.path.exists(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as reader:
            text = reader.read()
        model_config_dict = json.loads(text)
    else:
        print("model_config_path doesn't exist.")
        sys.exit()

    if os.path.exists(model_config_dict["final_model_saving_dir"]):
        output_model_file = model_config_dict["final_model_saving_dir"] + "pytorch_model.bin"
        output_config_file = model_config_dict["final_model_saving_dir"] + "bert_config.json"
        output_vocab_file = model_config_dict["final_model_saving_dir"] + "vocab.txt"
    else:
        print("model_saving_dir doesn't exist.")
        sys.exit()

    if os.path.exists(logger_file_dir):
        logging.basicConfig(
            filename=logger_file_dir + "logs.txt",
            filemode="w"
        )
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
    else:
        print("logger_file_path doesn't exist.")
        sys.exit()

    if os.path.exists(labels_file):
        print("Labels file exist")
    else:
        print("labels_file doesn't exist.")
        sys.exit()

    logger.info("Training configurations are given below ::")
    for key, val in model_config_dict.items():
        logger.info("{} == {}".format(key, val))

    logger.info("Started training model :::::::::::::::::::::")
    
    bert_config = BertConfig.from_json_file(model_config_dict["bert_config_path"])
    bert_tokenizer = BertTokenizer.from_pretrained(
        model_config_dict["bert_vocab_path"],
        config=bert_config,
        do_lower_case=model_config_dict["tokenizer_do_lower_case"]
    )
    # saving confgi and tokenizer 
    bert_tokenizer.save_vocabulary(output_vocab_file)
    bert_config.to_json_file(output_config_file)

    labels = get_labels(labels_file) + ["<PAD>"]
    logger.info("Labels for Ner are: {}".format(labels))

    label2idx = {l: i for i, l in enumerate(labels)}

    # preparing training data
    train_dataset = load_and_cache_examples(
        data_dir=data_dir, 
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["<PAD>"],
        mode="train", logger=logger
    )
    # preparing eval data
    eval_dataset = load_and_cache_examples(
        data_dir=data_dir, 
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["<PAD>"],
        mode="dev", logger=logger
    )
    logger.info("Training data and eval data loaded successfully.")

    if model_config_dict["model_type"] == "crf":
        model = BertCrfForNER.from_pretrained(
            model_config_dict["bert_model_path"],
            config=bert_config,
            pad_idx=bert_tokenizer.pad_token_id,
            num_labels=len(labels)
        )   
    elif model_config_dict["model_type"] == "token_classification":
        model = BertForTokenClassification.from_pretrained(
            model_config_dict["bert_model_path"],
            config=bert_config,
            num_labels=len(labels),
            classification_layer_sizes=model_config_dict["classification_layer_sizes"]
        )
    elif  model_config_dict["model_type"] == "lstm_crf":
        model = BertLstmCrf.from_pretrained(
            model_config_dict["bert_model_path"],
            config=bert_config,
            num_labels=len(labels),
            pad_idx=bert_tokenizer.pad_token_id,
            lstm_hidden_dim=model_config_dict["lstm_hidden_dim"],
            num_lstm_layers=model_config_dict["num_lstm_layers"],
            bidirectional=model_config_dict["bidirectional"]
        )

    logger.info("{} model loaded successfully.".format(model_config_dict["model_type"]))

    # checking whether to finetune or not
    if model_config_dict["finetune"] == True:
        logger.info("Finetuning bert.")
    else:
        for param in list(model.bert.parameters()):
            param.requires_grad = False
        logger.infd("Freezing Berts weights.")

    # preparing optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]
    # total optimizer steps
    t_total = int((len(train_dataset) / model_config_dict["train_batch_size"]) * model_config_dict["num_epochs"])
    logger.info("t_total : {}".format(t_total))

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=model_config_dict["learning_rate"],
        eps=model_config_dict["epsilon"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=model_config_dict["warmup_steps"],
        num_training_steps=t_total
    )
    logger.info("{}".format(count_parameters))

    model.to(DEVICE)

    best_eval_f1 = 0.0
    for epoch in range(model_config_dict["num_epochs"]):
        train_result = train_epoch(
            model=model, dataset=train_dataset,
            batch_size=model_config_dict["train_batch_size"],
            label_map=label2idx,
            max_grad_norm=model_config_dict["max_grad_norm"],
            optimizer=optimizer, scheduler=scheduler, device=DEVICE
        )
        eval_result = eval_epoch(
            model=model, dataset=eval_dataset,
            batch_size=model_config_dict["validation_batch_size"],
            label_map=label2idx, device=DEVICE,
            give_lists=False
        )
        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {train_result["loss"]: .4f}| Train F1: {train_result["f1"]: .4f}')
        print(f'Eval Loss: {eval_result["loss"]: .4f}| Eval F1: {eval_result["f1"]: .4f}')
        logger.info(f'Epoch: {epoch + 1}')
        logger.info(f'Train Loss: {train_result["loss"]: .4f}| Train F1: {train_result["f1"]: .4f}')
        logger.info(f'Eval Loss: {eval_result["loss"]: .4f}| Eval F1: {eval_result["f1"]: .4f}')

        if best_eval_f1 < eval_result["f1"]:
            best_eval_f1 = eval_result["f1"]
            # saving model to disk
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), output_model_file)
            print("Saved a better model.")
            logger.info("Saved a beter model")
            del model_to_save

    # loading the best model and test results
    model.load_state_dict(torch.load(output_model_file))
    logger.info("Loaded best model successfully.")

    test_dataset = load_and_cache_examples(
        data_dir=data_dir, 
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["<PAD>"],
        mode="test", logger=logger
    )
    logger.info("Test data loaded successfully.")

    test_result = eval_epoch(
        model=model, dataset=eval_dataset,
        batch_size=model_config_dict["validation_batch_size"],
        label_map=label2idx, device=DEVICE,
        give_lists=True
    )
    print("Test Results classification report...")
    print(classification_report(test_result["out_label_list"], test_result["preds_list"]))
    logger.info("Results on test data are: {}".format(
            {
                key: val
                for key, val in test_result.items()
                if key not in ["out_label_list", "preds_list"]
            }
        )
    )
    return


if __name__ == "__main__":
    fire.Fire(train_ner_model)