import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel, BertForSequenceClassification
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


class Bert_C(nn.Module):
    def __init__(self, bert_model_name, num_classes, *args, **kwargs):
        super(Bert_C, self).__init__()
        self.num_classes = num_classes
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes, *args, **kwargs)

    def forward(self, input_ids, labels=None):
        bert_out = self.bert_model(input_ids, labels=labels)
        return bert_out


class SelfDefinedBert_C(nn.Module):
    def __init__(self, bert_config_file, num_classes, *args, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.bert_config = BertConfig.from_json_file(bert_config_file)
        self.bert_model = BertModel(self.bert_config, *args, **kwargs)
        self.d_model = self.bert_config.hidden_size
        self.p_drop = self.bert_config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=self.p_drop)
        self.fc = nn.Linear(self.d_model, self.num_classes)

    def forward(self, input_ids, labels=None):
        bert_out = self.bert_model(input_ids)      #
        '''
        (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        '''
        logits = self.fc(self.dropout(bert_out.pooler_output))
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return StarRiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_out.hidden_states,
            attentions=bert_out.attentions
        )


class Lstm_C(nn.Module):
    def __init__(self, vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_classes):
        super().__init__()
        self.vocab_len = vocab_len
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=self.d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size,
                                           num_layers=self.num_layers, bias=True, batch_first=True,
                                           dropout=self.p_dropout, bidirectional=True)
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_classes, bias=True)
        for p in self.embedding.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, input_ids, labels=None):
        emb_input = self.embedding(input_ids)
        lstm_out, (h,c) = self.lstm(emb_input)
        logits = self.fc(self.dropout(lstm_out[:, -1, :]))   # 用隐层输出最后一位。
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return StarRiverClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=h,
            attentions=None
        )


@dataclass
class StarRiverClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None