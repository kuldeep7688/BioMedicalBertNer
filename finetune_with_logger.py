import os
import json
import logging
import fire
import sys

import torch
from biomedical_bert_ner.utils.utilities import train_epoch, eval_epoch
from biomedical_bert_ner.utils.utilities import predictions_from_model
from biomedical_bert_ner.utils.utilities import align_predicted_labels_with_original_sentence_tokens
from biomedical_bert_ner.utils.utilities import load_and_cache_examples
from biomedical_bert_ner.utils.utilities import get_labels, count_parameters

from biomedical_bert_ner.models.models import *

from transformers import BertTokenizer, BertConfig
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Being used as {} \n".format(DEVICE))



rint("model_saving_dir doesn't exist.")
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

    labels = get_labels(labels_file)
    logger.info("Labels for Ner are: {}".format(labels))

    label2idx = {l: i for i, l in enumerate(labels)}

    # preparing training data
    train_dataset = load_and_cache_examples(
        data_dir=data_dir,
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["O"],
        mode="train", logger=logger
    )
    # preparing eval data
    eval_dataset = load_and_cache_examples(
        data_dir=data_dir,
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["O"],
        mode="dev", logger=logger
    )
    logger.info("Training data and eval data loaded successfully.")

    if model_config_dict["model_type"] == "crf":
        model = BertCrfForNER.from_pretrained(
            model_config_dict["bert_model_path"],
            config=bert_config,
            pad_idx=bert_tokenizer.pad_token_id,
            sep_idx=bert_tokenizer.sep_token_id,
            num_labels=len(labels)
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
            optimizer=optimizer, scheduler=scheduler, device=DEVICE,
            sep_token_id=bert_tokenizer.sep_token_id
        )
        eval_result = eval_epoch(
            model=model, dataset=eval_dataset,
            batch_size=model_config_dict["validation_batch_size"],
            label_map=label2idx, device=DEVICE, sep_token_id=bert_tokenizer.sep_token_id,
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

    test_dataset, test_examples, test_features = load_and_cache_examples(
        data_dir=data_dir,
        max_seq_length=model_config_dict["max_seq_length"],
        tokenizer=bert_tokenizer,
        label_map=label2idx,
        pad_token_label_id=label2idx["O"],
        mode="test", logger=logger,
        return_features_and_examples=True
    )
    logger.info("Test data loaded successfully.")

    test_label_predictions = predictions_from_model(
        model=model, tokenizer=bert_tokenizer,
        dataset=test_dataset,
        batch_size=model_config_dict["validation_batch_size"],
        label2idx=label2idx, device=DEVICE
    )
    # restructure test_label_predictions with real labels
    aligned_predicted_labels, true_labels = align_predicted_labels_with_original_sentence_tokens(
        test_label_predictions, test_examples, test_features, max_seq_length=model_config_dict["max_seq_length"],
        num_special_tokens=model_config_dict["num_special_tokens"]
    )
    print("Test Results classification report...")
    print(classification_report(true_labels, aligned_predicted_labels))
    return

if __name__ == "__main__":
    fire.Fire(train_ner_model)
