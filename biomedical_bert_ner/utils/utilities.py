import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class InputExample(object):
    """
    A single training/test example.
    """
    def __init__(self, guid, words=None, labels=None, sentence=None):
        """Contructs a InputExample object.
        Args:
            guid (TYPE): unique id for the example
            words (TYPE): the words of the sequence
            labels (TYPE): the labels for each work of the sentence
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.sentence = sentence

        if self.words is None and self.sentence:
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            # split sentence on whitepsace so that different tokens may be attributed to their original positions
            for c in self.sentence:
                if _is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            self.words = doc_tokens
            if self.labels is None:
                self.labels = ["O"]*len(self.words)


class InputFeatures(object):
    """
    A sigle set of input features for an example.
    """
    def __init__(self, input_ids, input_mask, segment_ids, label_ids=None, token_to_orig_index=None, orig_to_token_index=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.token_to_orig_index = token_to_orig_index
        self.orig_to_token_index = orig_to_token_index


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
                words.append(splits[0])
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
        token_to_orig_index = []
        orig_to_token_index = []
        for word_idx, (word, label) in enumerate(zip(example.words, example.labels)):
            orig_to_token_index.append(len(tokens))
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # USe the real label id for the first token of the word, and
                # propagate I-tag for the splitted tokens
                label_ids.extend(
                    [label_map[label]] + [label_map[get_i_label(label, label_map)]]
                    * (len(word_tokens) - 1)
                )
            for tok in word_tokens:
                token_to_orig_index.append(word_idx)

        special_tokens_count = 2 # for bert cls sentence sep
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]


        tokens += [tokenizer.sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [0]*len(tokens)

        tokens = [tokenizer.cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(input_ids)

        # Zero pad up to the sequence length
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

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
                label_ids=label_ids,
                token_to_orig_index=token_to_orig_index,
                orig_to_token_index=orig_to_token_index
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
    mode, data_dir=None, logger=None, summary_writer=None,
    sentence_list=None, return_features_and_examples=False
):
    "Loads data features from cache or dataset file"

    if sentence_list is None:
        if data_dir:
            print("Creating features from dataset file at {}".format(data_dir))
            examples = read_examples_from_file(data_dir, mode)
    else:
        # will mainly be used in
        examples = []
        for idx, sentence in enumerate(sentence_list):
            examples.append(
                InputExample(
                    guid=idx, words=None, labels=None, sentence=sentence
                )
            )



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
    if return_features_and_examples:
        return dataset, examples, features
    else:
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
        step_loss = loss.item()
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
                f'Tr Iter: {step+1}| step_loss: {step_loss: .3f}| avg_tr_f1: {temp_results["f1"]: .3f}'
            )

    epoch_loss = tr_loss / len(dataloader)
    results = get_result_matrix(epoch_loss, label_map, preds, out_label_ids)

    if summary_writer:
        summary_writer.add_scaler("F1_epoch/train", results["f1"])
        summary_writer.add_scaler("Precision_epoch/train", results["precision"])
        summary_writer.add_scaler("Recall_epoch/train", results["recall"])

    return results


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
            step_loss = loss.item()
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
                    eval_loss / (step + 1), label_map, preds, out_label_ids, give_lists=False
                )
                epoch_iterator.set_description(
                    f'Eval Iter: {step+1}| step_loss: {step_loss: .3f}| avg_ev_f1: {temp_results["f1"]: .3f}'
                )

    epoch_loss = eval_loss / len(dataloader)
    results = get_result_matrix(epoch_loss, label_map, preds, out_label_ids, give_lists=give_lists)

    if summary_writer:
        summary_writer.add_scaler("Loss/eval", epoch_loss)
        summary_writer.add_scaler("F1_epoch/eval", results["f1"])
        summary_writer.add_scaler("Precision_epoch/eval", results["precision"])
        summary_writer.add_scaler("Recall_epoch/eval", results["recall"])

    return results


# below functions are helpful in Inferencing
def predictions_from_model(model, tokenizer, dataset, batch_size, label2idx, device):
    pred_logits = None
    input_ids_list = None
    model.eval()
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    epoch_iterator = tqdm(dataloader, total=len(dataloader))
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": None
            }
            # getting outputs
            logits, _, _ = model(**inputs)

            # appending predictions and labels to list
            if pred_logits is None:
                pred_logits = logits.detach().cpu().numpy()
                input_ids_list = inputs["input_ids"][:, 1:].detach().cpu().numpy()
            else:
                pred_logits = np.append(pred_logits, logits.detach().cpu().numpy(), axis=0)
                input_ids_list = np.append(
                    input_ids_list,
                    inputs["input_ids"][:, 1:].detach().cpu().numpy(),
                    axis=0
                )

    idx2label = {i: label for label, i in label2idx.items()}
    prediction_labels  = []
    for sentence_label_logits, sentence_input_ids in zip(pred_logits, input_ids_list):
        temp = []
        for i, (p, w) in enumerate(zip(sentence_label_logits, sentence_input_ids)):
            if w == tokenizer.sep_token_id:
                break
            temp.append(idx2label[p])
        prediction_labels.append(temp)
    return prediction_labels


def align_predicted_labels_with_original_sentence_tokens(predicted_labels, examples, features, max_seq_length, num_special_tokens):
    """The label_predictions out of the model is according to the tokens (that we get after tokenizing every word using tokenizer).
    We need to align the predictions with the original words of the sentence.
    """
    aligned_predicted_labels = []
    for idx, (feature, p_l_s) in enumerate(zip(features, predicted_labels)):
#         print(idx)
        temp = []
        for i in range(len(feature.orig_to_token_index)):
            token_idx = feature.orig_to_token_index[i]
            if token_idx + 1 < (max_seq_length - num_special_tokens):
                temp.append(p_l_s[token_idx])
            else:
                temp.append("O")
        aligned_predicted_labels.append(temp)

    return aligned_predicted_labels, [ex.labels for ex in examples]


def convert_to_ents(tokens, tags):
    start_offset = None
    end_offset = None
    ent_type = None

    text = " ".join(tokens)
    entities = []
    start_char_offset = 0
    for offset, (token, tag) in enumerate(zip(tokens, tags)):
        token_tag = tag
        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                entity = {
                    "type": ent_type,
                    "entity": " ".join(tokens[start_offset: end_offset + 1]),
                    "start_offset": start_char_offset,
                    "end_offset": start_char_offset + len(" ".join(tokens[start_offset: end_offset + 1]))
                }
                entities.append(entity)
                start_char_offset += len(" ".join(tokens[start_offset: end_offset + 2])) + 1
                start_offset = None
                end_offset = None
                ent_type = None
            else:
                start_char_offset += len(tokens) + 1
        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset
        elif ent_type != token_tag[2:]:
            end_offset = offset - 1
            entity = {
                "type": ent_type,
                "entity": " ".join(tokens[start_offset: end_offset + 1]),
                "start_offset": start_char_offset,
                "end_offset": start_char_offset + len(" ".join(tokens[start_offset: end_offset + 1]))
            }
            entities.append(entity)
            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that foes up untill the last token
    if ent_type and start_offset is not None and end_offset is not None:
        entity = {
            "type": ent_type,
            "entity": " ".join(tokens[start_offset:]),
            "start_offset": start_char_offset,
            "end_offset": start_char_offset + len(" ".join(tokens[start_offset:]))
        }
        entities.append(entity)
    return text, entities
