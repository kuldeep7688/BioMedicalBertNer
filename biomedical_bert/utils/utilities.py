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


def get_labels(path):
	if path:
		with open(path, "r") as f:
			labels = f.read().splitlines()

		if "O" not in labels:
			labels = ["O"] + labels
		return labels
	else:
		return None


def load_and_cache_examples(
	max_seq_length, tokenizer, label_map, pad_token_label_id,
	mode, data_dir=None, logger=None,
	summary_writer=None, examples=None
):
	"Loads data features from cache or dataset file"

	if data_dir:
		print("Creating features from dataset file at {}".format(data_dir))
		examples = read_examples_from_file(data_dir, mode)


	features = convert_examples_to_features(
		examples=examples, label_map=label_map,
		max_seq_length=max_seq_length, mode=mode,
		tokenizer=tokenizer,
		pad_token_label_id=pad_token_label_id,
		logger=logger, summary_writer=summary_writer
	)
	# Convert into tensors and build dataset
	all_input_ids_list = []
	all_input_mask_list = []
	all_segment_ids_list = []
	all_label_ids_list = []

	for f in features:
		all_input_ids_list.append(f.input_ids)
		all_input_mask_list.append(f.input_mask)
		all_segment_ids_list.append(f.segment_ids)
		all_label_ids_list.append(f.label_ids)

	all_input_ids = torch.tensor(all_input_ids_list, dtype=torch.long)
	all_input_mask = torch.tensor(all_input_mask_list, dtype=torch.long)
	all_segment_ids = torch.tensor(all_segment_ids_list, dtype=torch.long)
	all_label_ids = torch.tensor(all_label_ids_list, dtype=torch.long)

	dataset = TensorDataset(
		all_input_ids, all_input_mask, all_segment_ids, all_label_ids
	)

	return dataset


def count_parameters(model):
	print(
		"Number of trainable parameters in the model are {}".format(
			sum(p.numel() for p in model.parameters() if p.requires_grad)
		)
	)


def get_result_matrix(
	loss, label_map, predictions_tensor,
	labels_tensor, give_lists=False
):
	"""
	Get the results given predictions and labels
	"""
	label_to_not_consider_in_results = [
		idx for label, idx in label_map.items()
		if label in ["<PAD>", "<EOS>"]
	]
	label2idx = {i: label for label, i in label_map.items()}

	out_label_list = [[] for _ in range(labels_tensor.shape[0])]
	preds_list = [[] for _ in range(predictions_tensor.shape[0])]

	for i in range(labels_tensor.shape[0]):
		for j in range(labels_tensor.shape[1]):
			if labels_tensor[i, j] not in label_to_not_consider_in_results:
				out_label_list[i].append(label2idx[labels_tensor[i][j]])
				preds_list[i].append(label2idx[predictions_tensor[i][j]])

	if give_lists:
		results = {
			"loss": loss,
			"precision": precision_score(out_label_list, preds_list),
			"recall": recall_score(out_label_list, preds_list),
			"f1": f1_score(out_label_list, preds_list),
			"out_label_list": out_label_list,
			"preds_list": preds_list
		}
	else:
		results = {
			"loss": loss,
			"precision": precision_score(out_label_list, preds_list),
			"recall": recall_score(out_label_list, preds_list),
			"f1": f1_score(out_label_list, preds_list),
			"out_label_list": out_label_list,
			"preds_list": preds_list
		}
	return results


def train_epoch(
	model, dataset, batch_size, label_map, max_grad_norm,
	optimizer, scheduler, device, summary_writer=None
):
	tr_loss = 0.0

	preds = None
	out_label_ids = None

	model.train()
	sampler = RandomSampler(dataset)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
	print_stats_at_step = int(len(dataloader) / 20.0)
	epoch_iterator = tqdm(dataloader)
	for step, batch in enumerate(epoch_iterator):
		model.zero_grad()
		batch = tuple(t.to(device) for t in batch)
		inputs = {
			"input_ids": batch[0],
			"attention_mask": batch[1],
			"token_type_ids": batch[2],
			"labels": batch[3]
		}
		# getting outputs
		logits, inputs["labels"], loss = model(**inputs)

		# propagating loss backwards and scheduler and opt. steps
		loss.backward()
		step_loss = loss.items()
		tr_loss += step_loss

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
		optimizer.step()
		scheduler.step()

		if summary_writer:
			summary_writer.add_scaler("Loss/train", step_loss)
			summary_writer.add_scaler("LR/train", scheduler.get_lr()[0])


		# appending predictions and labels to list
		# for calculation of result
		if preds is None:
			preds = logits.detach().cpu().numpy()
			out_label_ids = inputs["labels"].detach().cpu().numpy()
		else:
			preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

		if step % print_stats_at_step == 0:
			temp_results = get_result_matrix(
				tr_loss / (step + 1), label_map, preds, out_label_ids, give_lists=False
			)
			epoch_iterator.set_description(
				f'Tr Iter: {step+1}| step_loss: {step_loss}| avg_tr_f1: {temp_results["f1"]}'
			)

	epoch_loss = tr_loss / len(dataloader)
	results = get_result_matrix(epoch_loss, label_map, preds, out_label_ids)

	if summary_writer:
		summary_writer.add_scaler("F1_epoch/train", results["f1"])
		summary_writer.add_scaler("Precision_epoch/train", results["precision"])
		summary_writer.add_scaler("Recall_epoch/train", results["recall"])

	return epoch_loss, results


def eval_epoch(
	model, dataset, batch_size, label_map, device,
	summary_writer=None, give_lists=False
):
	eval_loss = 0.0

	preds = None
	out_label_ids = None

	model.eval()
	sampler = SequentialSampler(dataset)
	dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
	print_stats_at_step = int(len(dataloader) / 10.0)
	epoch_iterator = tqdm(dataloader)
	with torch.no_grad():
		for step, batch in enumerate(epoch_iterator):
			batch = tuple(t.to(device) for t in batch)
			inputs = {
				"input_ids": batch[0],
				"attention_mask": batch[1],
				"token_type_ids": batch[2],
				"labels": batch[3]
			}
			# getting outputs
			logits, inputs["labels"], loss = model(**inputs)

			# propagating loss backwards and scheduler and opt. steps
			step_loss = loss.items()
			eval_loss += step_loss

			# appending predictions and labels to list
			# for calculation of result
			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs["labels"].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

			if step % print_stats_at_step == 0:
				temp_results = get_result_matrix(
					tr_loss / (step + 1), label_map, preds, out_label_ids, give_lists=False
				)
				epoch_iterator.set_description(
					f'Tr Iter: {step+1}| step_loss: {step_loss}| avg_tr_f1: {temp_results["f1"]}'
				)

	epoch_loss = eval_loss / len(dataloader)
	results = get_result_matrix(epoch_loss, label_map, preds, out_label_ids, give_lists=give_lists)

	if summary_writer:
		summary_writer.add_scaler("Loss/eval", epoch_loss)
		summary_writer.add_scaler("F1_epoch/eval", results["f1"])
		summary_writer.add_scaler("Precision_epoch/eval", results["precision"])
		summary_writer.add_scaler("Recall_epoch/eval", results["recall"])

	return epoch_loss, results


