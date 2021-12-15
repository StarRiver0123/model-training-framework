import math
import numpy as np
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.d_vocab = d_vocab
        self.d_model = d_model
        self.embedding = nn.Embedding(d_vocab, d_model)

    def forward(self, x):
        # input size:  *,N,L
        # output size: *,N,L,D_model
        y = self.embedding(x) * math.sqrt(self.d_model)
        return y


class PositionalEncoding(nn.Module):
    # __instance = None
    #  应用了单例模式
    def __init__(self, d_model, max_len, p_drop):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.p_drop = p_drop
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        dim_temp = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * dim_temp)
        pe[:, 1::2] = torch.cos(position * dim_temp)
        pe = pe.unsqueeze(0)    # size: 1,L,D
        self.register_buffer('pe', pe)    #反向传播不会更新weight
        self.dropout = nn.Dropout(p=self.p_drop)

    # def __new__(cls, *args, **kargs):
    #     if cls.__instance is None:
    #         cls.__instance = nn.Module.__new__(cls)
    #     return cls.__instance

    def forward(self, x):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        s_len = x.size(1)
        y = self.dropout(x + self.pe[:, :s_len])
        return y


class Predictor(nn.Module):
    def __init__(self, d_model, d_vocab):
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.linear = nn.Linear(self.d_model, self.d_vocab, bias=False)

    def forward(self, x):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_target
        return self.linear(x)


def gen_pad_only_mask(k_input, pad_token=None):
    # input size:  N,L
    # output size: N,L_k
    if pad_token is not None:
        mask = (k_input == pad_token)
    else:
        mask = None
    return mask


def gen_seq_only_mask(q_input, k_input):
    # input size:  N,L
    # output size: L_q,L_k
    device = q_input.device
    square = np.ones((q_input.size(-1), k_input.size(-1)))
    mask = torch.from_numpy(np.triu(square, k=1).astype(np.bool)).to(device)  # size: L_q,L_k
    return mask

def gen_full_false_mask(q_input, k_input):
    device = q_input.device
    return torch.zeros((q_input.size(-1), k_input.size(-1))).type(torch.bool).to(device)

def cat_seq_pad_mask(seq_mask, pad_mask):
    # input size:  seq_mask:L_q,L_k; pad_mask: N,L_k
    # output size: N,L_q,L_k
    if (pad_mask is not None) and (seq_mask is not None):
        return pad_mask.unsqueeze(-2) | (seq_mask.unsqueeze(0) == 1)
    elif (pad_mask is not None) and (seq_mask is None):
        return pad_mask.unsqueeze(-2)
    elif (pad_mask is None) and (seq_mask is not None):
        return seq_mask.unsqueeze(0)
    else:
        return None


def gen_pad_seq_mask(q_input, k_input, pad_token=None):
    # input size:  N,L
    # output size: N,L_q,L_k
    seq_mask = gen_seq_only_mask(q_input, k_input).unsqueeze(0)   #  L_q,L_k -> 1,L_q,L_k
    if pad_token is not None:
        pad_mask = gen_pad_only_mask(k_input, pad_token).unsqueeze(-2)   # N,L_k -> N,1,L_k
        mask = pad_mask | (seq_mask == 1)
    else:
        mask = seq_mask
    return mask


