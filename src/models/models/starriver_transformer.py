import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.models.base_component import PositionalEmbedding, Predictor, cat_seq_pad_mask


# 实现的是pre-LN结构
class StarRiverTransformer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.p_drop = p_drop
        self.src_vocab_len = src_vocab_len
        self.tgt_vocab_len = tgt_vocab_len
        self.src_bert_model = None
        self.tgt_bert_model = None
        self.encoder_embedding = PositionalEmbedding(self.src_vocab_len, self.d_model, self.max_len, self.p_drop)
        self.decoder_embedding = PositionalEmbedding(self.tgt_vocab_len, self.d_model, self.max_len, self.p_drop)
        self.encoder_layers = Encoder(self.d_model, self.nhead, self.d_ff, self.p_drop, self.num_encoder_layers)
        self.decoder_layers = Decoder(self.d_model, self.nhead, self.d_ff, self.p_drop, self.num_decoder_layers)
        self.predictor = Predictor(self.d_model, self.tgt_vocab_len)

    def forward(self, enc_input, dec_input, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # input size:  N,L
        # output size: N,L,D_target
        if self.src_bert_model is None:
            enc_emb = self.encoder_embedding(enc_input)
        else:
            with torch.no_grad():
                enc_emb = self.src_bert_model(enc_input).last_hidden_state
        if self.tgt_bert_model is None:
            dec_emb = self.decoder_embedding(dec_input)
        else:
            with torch.no_grad():
                dec_emb = self.tgt_bert_model(dec_input).last_hidden_state
        enc_out = self.encoder_layers(enc_emb, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decoder_layers(dec_emb, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.predictor(dec_out)

    def encoder(self, enc_input, src_key_padding_mask=None):
        # input size:  N,L
        # output size: N,L,D
        if self.src_bert_model is None:
            enc_emb = self.encoder_embedding(enc_input)  # enc_input: N,L,D
        else:
            with torch.no_grad():
                enc_emb = self.src_bert_model(enc_input).last_hidden_state
        return self.encoder_layers(enc_emb, src_key_padding_mask=src_key_padding_mask)

    def decoder(self, dec_input, enc_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # input size:  N,L
        # output size: N,L,D
        if self.tgt_bert_model is None:
            dec_emb = self.decoder_embedding(dec_input)  # dec_input: N,L,D
        else:
            with torch.no_grad():
                dec_emb = self.tgt_bert_model(dec_input).last_hidden_state
        return self.decoder_layers(dec_emb, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, p_drop, num_encoder_layers):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.d_model = d_model
        self.encoder_layer_stack = nn.ModuleList(EncoderLayer(d_model, nhead, d_ff, p_drop) for _ in range(self.num_encoder_layers))
        self.layer_norm = nn.LayerNorm(self.d_model)  # 这一层在原论文中并不存在，根据修改后的残差结构补上的，参考了The Annotated Transformer中的实现

    def forward(self, x, src_key_padding_mask=None):
        # input size:  *,N,L,D
        # output size: *,N,L,D_model
        layer_x = x
        for layer in self.encoder_layer_stack:
            layer_x = layer(layer_x, src_key_padding_mask)
        return self.layer_norm(layer_x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, p_drop):
        super().__init__()
        self.mh_attention = MultiHeadAttention(d_model, nhead, p_drop)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, p_drop)
        self.rln0 = Residual_LN(d_model, p_drop)
        self.rln1 = Residual_LN(d_model, p_drop)

    def forward(self, x, src_key_padding_mask=None):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.unsqueeze(-2)   # N,L_k -> N,1,L_k
        attention = self.rln0(x, lambda x: self.mh_attention(x, x, x, src_key_padding_mask))
        y = self.rln1(attention, lambda x: self.ffn(x))
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, p_drop, num_decoder_layers):
        super().__init__()
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.decoder_layer_stack = nn.ModuleList(DecoderLayer(d_model, nhead, d_ff, p_drop) for _ in range(self.num_decoder_layers))
        self.layer_norm = nn.LayerNorm(self.d_model)  # 这一层在原论文中并不存在，根据修改后的残差结构补上的，参考了The Annotated Transformer中的实现

    def forward(self, x, k_v, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # input size:  N,L,D_model
        # output size: N,L,D_model
        q = x
        for layer in self.decoder_layer_stack:
            q = layer(q, k_v, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.layer_norm(q)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, p_drop):
        super().__init__()
        self.masked_mh_attention = MultiHeadAttention(d_model, nhead, p_drop)
        self.mh_attention = MultiHeadAttention(d_model, nhead, p_drop)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, p_drop)
        self.rln0 = Residual_LN(d_model, p_drop)
        self.rln1 = Residual_LN(d_model, p_drop)
        self.rln2 = Residual_LN(d_model, p_drop)

    def forward(self, q, k_v, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        tgt_seq_pad_mask = cat_seq_pad_mask(tgt_mask, tgt_key_padding_mask)   #N,L_q,L_k
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(-2)    # N,L_k -> N,1,L_k
        masked_attention = self.rln0(q, lambda q: self.masked_mh_attention(q, q, q, tgt_seq_pad_mask))
        attention = self.rln1(masked_attention, lambda x: self.mh_attention(x, k_v, k_v, memory_key_padding_mask))
        y = self.rln2(attention, lambda x: self.ffn(x))
        return y


class Residual_LN(nn.Module):
    def __init__(self, d_model, p_drop):
        super().__init__()
        self.d_model = d_model
        self.p_drop = p_drop
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=self.p_drop)

    def forward(self, x, sub_func_layer):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        # return self.layer_norm(x + self.dropout(sub_func_layer(x)))   # 这是严格按照transformer论文中的说明实现的，似乎有问题。
        return x + self.dropout(sub_func_layer(self.layer_norm(x)))    #参考了The Annotated Transformer中的实现, 这是re-LN模式


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, p_drop):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(p=self.p_drop),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, x):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        y = self.fc(x)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, p_drop):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.p_drop = p_drop
        assert self.d_model % self.nhead == 0
        self.d_k = self.d_model // self.nhead
        self.d_v = self.d_model // self.nhead
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_out = nn.Linear(self.d_model, self.d_model, bias=False)
        # 需要参考The Annotated Transformer中的实现加上dropout吗？

    def forward(self, q_input, k_input, v_input, mask=None):
        # input size:  *,N,L,D_model
        # output size: *,N,L,D_model
        q_batch_size = q_input.size(0)
        k_batch_size = k_input.size(0)
        v_batch_size = v_input.size(0)
        q = self.W_Q(q_input).view(q_batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # permute(0,2,1,3)
        k = self.W_K(k_input).view(k_batch_size, -1, self.nhead, self.d_k).transpose(1, 2)  # permute(0,2,1,3)
        v = self.W_V(v_input).view(v_batch_size, -1, self.nhead, self.d_v).transpose(1, 2)  # permute(0,2,1,3)
        if mask is not None:
            mask = mask.unsqueeze(1)  # 扩充nhead维度
        sdp_attention = self._scaled_dot_product_attention(q, k, v, mask)
        mh_attention = self.W_out(sdp_attention.transpose(1, 2).contiguous().view(q_batch_size, -1, self.d_model))
        return mh_attention

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        # input shape: (*,L,D)
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # scores.masked_fill(mask, -1e9)
            scores = scores.masked_fill(mask, -1e9)
        return torch.matmul(F.softmax(scores, dim=-1), v)

