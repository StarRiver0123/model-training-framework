import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel, BertConfig


class BertCRF(nn.Module):
    def __init__(self, bert_name, num_tags):
        super().__init__()
        self.bert_name = bert_name
        self.num_tags = num_tags
        self.bert_config = BertConfig.from_pretrained(self.bert_name)
        self.bert_model = BertModel.from_pretrained(self.bert_name)
        self.d_model = self.bert_config.hidden_size
        self.p_dropout = self.bert_config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.fc = nn.Linear(self.d_model, self.num_tags, bias=True)
        self.crf = CRF(self.num_tags, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.num_tags)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def bert_emit(self, seq_input):
        bert_out = self.bert_model(seq_input).last_hidden_state[:, 1:, :]
        # logit = self.fc(self.dropout(bert_out))
        logit = self.layer_norm(self.fc(self.dropout(bert_out)))
        return logit
