from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # sequence_ouput = [batch_size, seq_len, hidden_size]

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


class BertForTokenClassification(BertModel):
    """
    Simply doing token classification over bert outputs.
    """
    def __init__(self, config, num_labels, classification_layer_sizes=[]):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.dropout_layer = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config)
        self.input_layer_sizes = [config.hidden_size] + classification_layer_sizes
        self.output_layer_size = classification_layer_sizes + [self.num_labels]
        self.classification_module = nn.ModuleList(
            nn.Linear(inp, out)
            for inp, out in zip(self.input_layer_sizes, self.output_layer_size)
        )
        self.num_linear_layer = len(classification_layer_sizes) + 1 
        self.init_weights()

    def forward(
        self, input_ids, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )

        logits = outputs[0]
        for layer_idx, layer in enumerate(self.classification_module):
            if layer_idx + 1 != self.num_linear_layer:
                logits = self.dropout_layer(F.relu(layer(logits)))
            else:
                logits = layer(logits)

        # escaping cls token
        logits = logits[:, 1:, :].contiguous()
        if labels is not None:
            labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        attention_mask = attention_mask[:, 1:].contiguous()

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss =loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        softs, out = torch.max(logits, axis=2)
        return out, labels, loss


class BertLstmCrf(BertModel):
    """On the outputs of Bert there is a LSTM layer.
    On top of the LSTM there is a  CRF layer.
    """
    def __init__(
        self, config, pad_idx, lstm_hidden_dim,
        num_lstm_layers, bidirectional, num_labels
    ):
        super(BertLstmCrf, self).__init__(config)
        self.dropout_prob = config.hidden_dropout_prob
        self.pad_idx = pad_idx
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.bidirectional = bidirectional
        self.num_labels = num_labels

        self.bert = BertModel(config)

        self.lstm = nn.LSTM(
            input_size=config.hidden_size, hidden_size=self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers, bidirectional=self.bidirectional,
            dropout=self.dropout_prob, batch_first=True
        )
        if self.bidirectional is True:
            self.linear = nn.Linear(self.lstm_hidden_dim*2, self.num_labels)
        else:
            self.linear = nn.Linear(self.lstm_hidden_dim, self.num_labels)

        self.crf_layer = CRF(self.num_labels, batch_first=True)
        self.dropout_layer = nn.Dropout(self.dropout_prob)

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
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        sequence_output = outputs[0]

        lstm_out, (hidden, cell) = self.lstm(sequence_output)
        logits = self.linear(self.dropout_layer(lstm_out))

        # removing cls token
        logits = logits[:, 1:, :]
        if labels is not None:
            labels = labels[:, 1:]
        input_ids = input_ids[:, 1:]

        # creating mask for crf
        mask = self.create_mask_for_crf(input_ids)

        # crf part 
        loss = self.crf_layer(logits, labels, mask=mask) * torch.tensor(-1, device=self.device)

        out = self.crf_layer.decode(logits)
        out = torch.tensor(out, dtype=torch.long, device=self.device)
        # out = [batch_Size, seq_len]
        return out, labels, loss