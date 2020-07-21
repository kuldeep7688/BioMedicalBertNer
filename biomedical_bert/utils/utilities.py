import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import multiprocessing
from functools import partial
# from seqeval.metrics import classification_report


class InputExample(object):
    """
    A single training/test example.
    """
    def __init__(self, guid, words, labels):
        """Contructs a InputExample object.
        
        Args:
            guid (TYPE): unique id for the example
            words (TYPE): the words of the sequence
            labels (TYPE): the labels for each work of the sentence
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(objec):
	"""
	A sigle set of input features for an example.
	"""
	def __init__(self, input_ids, input_mask, segment_ids, label_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
	file_path = os.path.join(data_dir, "{}.txt".format(mode))
	guid_index = 1
	examples = []
	with open(file_path, encoding="utf-8") as f:
		words = []
		labels = []
		for line in f:
			if line.startswith("-DOCSTART-") or line == "" or line == "\n":
				if words:
					examples.append(
						InputExample(
							guid="{}-{}".format(mode, guid_index),
							words=words,
							labels=labels
						)
					)
					guid_index += 1
					words = []
					labels = []
			else:
				splits = line.split("\t")
				words.append(split[0])
				if len(splits) > 1:
					labels.append(splits[-1].replace("\n", ""))
				else:
					# examples could have no label for model == test
					labels.append("O")
		if words:
			examples.append(
				InputExample(
					guid="{}-{}".format(mode, guid_index),
					words=words,
					labels=labels
				)
			)
	return examples


def get_i_label(beginning_label, label_map):
	"""To properly label segments of words broken by BertTokenizer=.
	"""
	if "B-" in beginning_label:
		i_label = "I-" + beginning_label.split("B-")[-1]
		return i_label
	elif "I-" in beginning_label:
		i_label = "I-" + beginning_label.split("I-")[-1]
		return i_label
	else:
		return "O"


def convert_examples_to_features(
	examples, label_map, max_seq_length,
	tokenizer, label_end_token="<EOS>",
	pad_token_label_id=-1, mask_padding_with_zero=True,
	logger=None, summary_writer=None, mode=None
):
	"""
	Prepare features to be given as input to Bert
	"""

	features = []
	for (ex_index, example) in enumerate(examples):
		tokens = []
		label_ids = []
		for word, label in zip(example.words, example.labels):
			word_tokens = tokenizer.tokenizer(word)
			if len(word_tokens) > 0:
				tokens.extend(word_tokens)
				# USe the real label id for the first token of the word, and
				# propagate I-tag for the splitted tokens
				label_ids.extend(
					[label_map[label]] + [label_map[get_i_label(label, label_map)]]
					* (len(word_tokens) - 1)
				)

		special_tokens_count = 2 # for bert cls sentence sep
		if len(tokens) > max_seq_length - special_tokens_count:
			tokens = tokens[:(max_seq_length - special_tokens_count)]
			label_ids = label_ids[:(max_seq_length - special_tokens_count)]


		tokens += [tokenizer.sep_token]
		label_ids += [pad_token_label_id]
		segment_ids = [0]*len(tokens)

		tokens = [tokenizer.cls_token] + tokens
		label_ids = [pad_token_label_id] + label_ids
		segment_ids = [0] = segment_ids

		input_ids = tokenizer.convert_tokens_to_ids(tokens)
		input_mask = [1]*len(input_ids)

		# Zero pad up to the sequence length
		padding_length = max_seq_length - len(input_ids)
		input_ids += ([tokenizer.pad_token] * len(padding_length))
		input_mask += ([0] * padding_length)
		segment_ids += ([0] * padding_length)
		label_ids += ([pad_token_label_id] * padding_length)

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length
		assert len(label_ids) == max_seq_length

		if ex_index < 5:
			if logger:
				logger.info("****** EXAMPLES ********")
				logger.info("guid: {}".format(example.guid))
				logger.info("tokens: {}".format(" ".join([str(x) for x in tokens])))
				logger.info("input ids : {}".format(" ".join([str(x) for x in input_ids])))
				logger.info("input_mask : {}".format(" ".join([str(x) for x in input_mask])))
				logger.info("segment_ids : {}".format(" ".join([str(x) for x in segment_ids])))
				logger.info("label_ids : {}".format(" ".join([str(x) for x in label_ids])))

			if summary_writer:
				summary_writer.add_text(mode, "guid: {}".format(example.guid), 0)
				summary_writer.add_text(mode, "tokens: {}".format(" ".join([str(x) for x in tokens])), 0)
				summary_writer.add_text(mode, "input ids : {}".format(" ".join([str(x) for x in input_ids])), 0)
				summary_writer.add_text(mode, "input_mask : {}".format(" ".join([str(x) for x in input_mask])), 0)
				summary_writer.add_text(mode, "segment_ids : {}".format(" ".join([str(x) for x in segment_ids])), 0)
				summary_writer.add_text(mode, "label_ids : {}".format(" ".join([str(x) for x in label_ids])), 0)

		features.append(
			InputFeatures(
				input_ids=input_ids,
				input_mask=input_mask,
				segment_ids=segment_ids,
				label_ids=label_ids
			)
		)

	return features

