import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class TwinTextRNN(nn.Module):
    def __init__(self, vocab_len, d_model, hidden_size, num_layers, p_dropout, pad_idx):
        super().__init__()
        self.vocab_len = vocab_len
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self.embedding_a = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=self.d_model, padding_idx=pad_idx)
        self.embedding_b = nn.Embedding(num_embeddings=self.vocab_len, embedding_dim=self.d_model, padding_idx=pad_idx)
        self.feature_extractor_a = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, batch_first=True, dropout=self.p_dropout, bidirectional=True)
        self.feature_extractor_b = nn.LSTM(input_size=self.d_model, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, bias=True, batch_first=True, dropout=self.p_dropout,
                                       bidirectional=True)


    def forward(self, t_a, t_b, t_c=None):
        emb_input_a = self.embedding_a(t_a)
        emb_input_b = self.embedding_b(t_b)
        feature_output_a, (h, c) = self.feature_extractor_a(emb_input_a)           # N,L,D
        feature_output_b, (h, c) = self.feature_extractor_b(emb_input_b)
        sentence_vector_a = torch.mean(feature_output_a, dim=1)                    # N,D
        sentence_vector_b = torch.mean(feature_output_b, dim=1)
        if t_c is not None:
            emb_input_c = self.embedding_b(t_c)
            feature_output_c, (h, c) = self.feature_extractor_b(emb_input_c)
            sentence_vector_c = torch.mean(feature_output_c, dim=1)
            return sentence_vector_a, sentence_vector_b, sentence_vector_c
        # similarity = F.cosine_similarity(sentence_vector_a, sentence_vector_b, dim=1)
        return sentence_vector_a, sentence_vector_b


class TwinBert(nn.Module):
    def __init__(self, bert_name):
        super().__init__()
        self.bert_name = bert_name
        self.bert_config = BertConfig.from_pretrained(self.bert_name)
        self.bert_model = BertModel.from_pretrained(self.bert_name)
        self.d_model = self.bert_config.hidden_size
        self.p_dropout = self.bert_config.hidden_dropout_prob


    def forward(self, t_a, t_b, t_c=None):
        # emb_input_a = self.embedding_a(t_a)
        # emb_input_b = self.embedding_b(t_b)
        # feature_output_a, (h, c) = self.feature_extractor_a(emb_input_a)           # N,L,D
        # feature_output_b, (h, c) = self.feature_extractor_b(emb_input_b)
        # sentence_vector_a = torch.mean(feature_output_a, dim=1)                    # N,D
        # sentence_vector_b = torch.mean(feature_output_b, dim=1)
        # if t_c is not None:
        #     emb_input_c = self.embedding_b(t_c)
        #     feature_output_c, (h, c) = self.feature_extractor_b(emb_input_c)
        #     sentence_vector_c = torch.mean(feature_output_c, dim=1)
        #     return sentence_vector_a, sentence_vector_b, sentence_vector_c
        # similarity = F.cosine_similarity(sentence_vector_a, sentence_vector_b, dim=1)
        # return similarity
        pass