import torch
import torch.nn as nn
from torchcrf import CRF


class LstmCRF(nn.Module):
    def __init__(self, vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx, num_tags):
        super().__init__()
        self.vocab_len = vocab_len
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.num_tags = num_tags
        self.embedding = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=self.d_model, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size,
                                           num_layers=self.num_layers, bias=True, batch_first=True,
                                           dropout=self.p_dropout, bidirectional=True)
        self.dropout = nn.Dropout(p=self.p_dropout)
        self.fc = nn.Linear(self.hidden_size * 2, self.num_tags, bias=True)
        # self.layer_norm = nn.LayerNorm(self.num_tags)
        self.crf = CRF(self.num_tags, batch_first=True)
        for p in self.embedding.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.lstm.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def emit(self, seq_input):
        emb_input = self.embedding(seq_input)
        lstm_out, (h,c) = self.lstm(emb_input)
        logits = self.fc(self.dropout(lstm_out))
        # logits = self.layer_norm(logits)
        return logits
