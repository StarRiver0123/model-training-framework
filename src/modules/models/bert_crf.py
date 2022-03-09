import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, BertModel, BertPreTrainedModel


class BertCRF(BertPreTrainedModel):
    def __init__(self, config, num_tags):
        super(BertCRF, self).__init__(config)
        self.num_tags = num_tags
        self.bert_config = config
        self.bert_model = BertModel(config)
        self.d_model = self.bert_config.hidden_size
        self.p_dropout = self.bert_config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.fc = nn.Linear(self.d_model, self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)
        # self.layer_norm = nn.LayerNorm(self.num_tags)
        self.init_weights()

    def emit(self, seq_input):
        bert_out = self.bert_model(seq_input).last_hidden_state       # last_hidden_state[:, 1:, :]
        # logit = self.fc(self.dropout(bert_out))
        logits = self.fc(self.dropout(bert_out))
        # logits = self.layer_norm(logits)
        return logits
