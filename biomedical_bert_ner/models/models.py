from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from torchcrf import CRF


class BertCrfForNER(BertModel):
    """
    This class inherits functionality from huggingface BertModel.
    It applies a crf layer on the Bert outputs.
    """
    def __init__(self, config, pad_idx, num_labels):
        """Inititalization
        
        Args:
            config (TYPE): model config flie (similar to bert_config.json)
            num_labels : total number of layers using the bio format
            pad_idx (TYPE): pad_idx of the tokenizer
            device (TYPE): torch.device()
        """
        super(BertCrfForNER, self).__init__(config)
        self.num_labels = num_labels
        self.pad_idx = pad_idx

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf_layer = CRF(self.num_labels, batch_first=True)
        self.linear = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def create_mask_for_crf(self, inp):
        """Creates a mask for the feesing to crf layer.
        
        Args:
            inp (TYPE): input given to bert layer
        """
        mask = inp != self.pad_idx
        # mask = [seq_len, batch_size]

        return mask
    
    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, labels=None
    ):
        """Forwar propagate.
        
        Args:
            input_ids (TYPE): bert input ids
            attention_mask (None, optional): attention mask for bert
            token_type_ids (None, optional): token type ids for bert
            position_ids (None, optional): position ids for bert
            head_mask (None, optional): head mask for bert
            labels (None, optional): labels required while training crf
        """
        # getting outputs from Bert
        outputs = self.bert(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        # taking tokens embeddings from the output
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # sequence_ouputs = [batch_size, seq_len, hidden_size]

        logits = self.linear(sequence_output)
        # logits = [batch_size, seq_len, num_labels]

        # removing cls token
        logits = logits[:, 1:, :]
        if labels is not None:
            labels = labels[:, 1:] # check whether labels include the cls token too or not
        input_ids = input_ids[:, 1:]

        mask = self.create_mask_for_crf(input_ids)
        if labels is not None:
            loss = self.crf_layer(
                logits, labels, mask=mask
            ) * torch.tensor(-1, device=self.device)
        else:
            loss = None
        # this is the crf loss

        out = self.crf_layer.decode(logits)
        out = torch.tensor(out, dtype=torch.long, device=self.device)

        # out = [batch_size, seq_length]
        return out, labels, loss