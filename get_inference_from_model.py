import os
import json
import logging
from pprint import pprint
import sys

import torch
from biomedical_bert_ner.utils.utilities import predictions_from_model
from biomedical_bert_ner.utils.utilities import align_predicted_labels_with_original_sentence_tokens
from biomedical_bert_ner.utils.utilities import load_and_cache_examples
from biomedical_bert_ner.utils.utilities import get_labels, convert_to_ents
from biomedical_bert_ner.models.models import *

from transformers import BertTokenizer, BertConfig


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device Being used as {} \n".format(DEVICE))


logging.basicConfig(
    filename="inference_logs.txt",
    filemode="w"
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

class NERTagger:
    def __init__(
        self, labels_file,
        model_config_path, device
    ):
        self.model_config_path = model_config_path
        self.labels_file = labels_file
        self.device = device
        if os.path.exists(self.model_config_path):
            with open(self.model_config_path, "r", encoding="utf-8") as reader:
                text = reader.read()
            self.model_config_dict = json.loads(text)
        else:
            print("model_config_path doesn't exist.")
            sys.exit()

        if os.path.exists(self.model_config_dict["final_model_saving_dir"]):
            self.model_file = self.model_config_dict["final_model_saving_dir"] + "pytorch_model.bin"
            self.config_file = self.model_config_dict["final_model_saving_dir"] + "bert_config.json"
            self.vocab_file = self.model_config_dict["final_model_saving_dir"] + "vocab.txt"
        else:
            print("model_saving_dir doesn't exist.")
            sys.exit()
        if os.path.exists(self.labels_file):
            print("Labels file exist")
        else:
            print("labels_file doesn't exist.")
            sys.exit()

        self.bert_config = BertConfig.from_json_file(self.config_file)
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.vocab_file,
            config=self.bert_config,
            do_lower_case=self.model_config_dict["tokenizer_do_lower_case"]
        )
        self.labels = get_labels(self.labels_file) + ["<PAD>"]
        self.label2idx = {l: i for i, l in enumerate(self.labels)}


        if self.model_config_dict["model_type"] == "crf":
            self.model = BertCrfForNER.from_pretrained(
                self.model_file,
                config=self.bert_config,
                pad_idx=self.bert_tokenizer.pad_token_id,
                num_labels=len(self.labels)
            )
        elif self.model_config_dict["model_type"] == "token_classification":
            self.model = BertForTokenClassification.from_pretrained(
                self.model_file,
                config=self.bert_config,
                num_labels=len(self.labels),
                classification_layer_sizes=self.model_config_dict["classification_layer_sizes"]
            )
        elif  self.model_config_dict["model_type"] == "lstm_crf":
            self.model = BertLstmCrf.from_pretrained(
                self.model_file,
                config=self.bert_config,
                num_labels=len(self.labels),
                pad_idx=self.bert_tokenizer.pad_token_id,
                lstm_hidden_dim=self.model_config_dict["lstm_hidden_dim"],
                num_lstm_layers=self.model_config_dict["num_lstm_layers"],
                bidirectional=self.model_config_dict["bidirectional"]
            )
        self.model.to(self.device)
        print("Model loaded successfully from the config provided.")

    def tag_sentences(self, sentence_list, logger, batch_size):
        dataset, examples, features = load_and_cache_examples(
            max_seq_length=self.model_config_dict["max_seq_length"],
            tokenizer=self.bert_tokenizer,
            label_map=self.label2idx,
            pad_token_label_id=self.label2idx["<PAD>"],
            mode="inference", data_dir=None,
            logger=logger, sentence_list=sentence_list,
            return_features_and_examples=True
        )

        label_predictions = predictions_from_model(
            model=self.model, tokenizer=self.bert_tokenizer,
            dataset=dataset, batch_size=batch_size,
            label2idx=self.label2idx, device=self.device
        )
        # restructure test_label_predictions with real labels
        aligned_predicted_labels, _ = align_predicted_labels_with_original_sentence_tokens(
            label_predictions, examples, features,
            max_seq_length=self.model_config_dict["max_seq_length"],
            num_special_tokens=self.model_config_dict["num_special_tokens"]
        )
        results = []
        for label_tags, example in zip(aligned_predicted_labels, examples):
            results.append(
                convert_to_ents(example.words, label_tags)
            )
        return results


if __name__ == "__main__":
    sentence_list = [
        "Number of glucocorticoid receptors in lymphocytes and their sensitivity to hormone action .",
        "The study demonstrated a decreased level of glucocorticoid receptors ( GR ) in peripheral blood lymphocytes from hypercholesterolemic subjects , and an elevated level in patients with acute myocardial infarction .",
        "In the lymphocytes with a high GR number , dexamethasone inhibited [ 3H ] -thymidine and [ 3H ] -acetate incorporation into DNA and cholesterol , respectively , in the same manner as in the control cells .",
        "On the other hand , a decreased GR number resulted in a less efficient dexamethasone inhibition of the incorporation of labeled compounds .",
        "hese data showed that the sensitivity of lymphocytes to glucocorticoids changed only with a decrease of GR level .",
        "Treatment with I-hydroxyvitamin D3 ( 1-1.5 mg daily , within 4 weeks ) led to normalization of total and ionized form of Ca2+ and of 25 ( OH ) D , but did not affect the PTH content in blood .",
        "The data obtained suggest that under conditions of glomerulonephritis only high content of receptors to 1.25 ( OH ) 2D3 in lymphocytes enabled to perform the cell response to the hormone effect .",
        "To investigate whether the tumor expression of beta-2-microglobulin ( beta 2-M ) could serve as a marker of tumor biologic behavior , the authors studied specimens of breast carcinomas from 60 consecutive female patients .",
        "Presence of beta 2-M was analyzed by immunohistochemistry .",
        "I love data science",
        "Humira showed better results than Cimzia for treating psoriasis ."
    ]

    tagger = NERTagger(
        labels_file="/media/rabbit/Work_data/all_nlp_datasets/bio_ner_datasets/jnlpba/labels_file.txt",
        model_config_path="configs/crf_ner_config.json",
        device=DEVICE
    )
    pprint(tagger.tag_sentences(sentence_list, logger=logger, batch_size=2))
