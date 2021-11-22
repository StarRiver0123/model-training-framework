import torch
import torch.nn as nn
from src.models.models.starriver_transformer import PositionalEmbedding, Predictor

class TorchTransformer(nn.Module):
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
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.d_ff, dropout=self.p_drop, batch_first=True, norm_first=True)
        self.predictor = Predictor(self.d_model, self.tgt_vocab_len)

    def forward(self, enc_input, dec_input, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
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
        out = self.transformer(src=enc_emb, tgt=dec_emb, src_mask=src_mask, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.predictor(out)

    def encoder(self, enc_input, src_key_padding_mask=None):
        if self.src_bert_model is None:
            enc_emb = self.encoder_embedding(enc_input)  # enc_input: N,L,D
        else:
            with torch.no_grad():
                enc_emb = self.src_bert_model(enc_input).last_hidden_state
        return self.transformer.encoder(src=enc_emb, src_key_padding_mask=src_key_padding_mask)

    def decoder(self, dec_input, enc_out, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.tgt_bert_model is None:
            dec_emb = self.decoder_embedding(dec_input)  # dec_input: N,L,D
        else:
            with torch.no_grad():
                dec_emb = self.tgt_bert_model(dec_input).last_hidden_state
        return self.transformer.decoder(tgt=dec_emb, memory=enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)

# 另一种分段拼接方式：
# class TorchTransformer(nn.Module):
#     def __init__(self, d_model, nhead, d_ff, max_len, num_encoder_layers, num_decoder_layers, p_drop, src_vocab_len, tgt_vocab_len):
#         super().__init__()
#         self.d_model = d_model
#         self.nhead = nhead
#         self.d_ff = d_ff
#         self.max_len = max_len
#         self.num_encoder_layers = num_encoder_layers
#         self.num_decoder_layers = num_decoder_layers
#         self.p_drop = p_drop
#         self.src_vocab_len = src_vocab_len
#         self.tgt_vocab_len = tgt_vocab_len
#         self.encoder_embedding = PositionalEmbedding(self.src_vocab_len, self.d_model, self.max_len, self.p_drop)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_ff, dropout=self.p_drop, batch_first=True, norm_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers, norm=nn.LayerNorm(self.d_model))
#         self.decoder_embedding = PositionalEmbedding(self.tgt_vocab_len, self.d_model, self.max_len, self.p_drop)
#         decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.d_ff,
#                                                    dropout=self.p_drop, batch_first=True, norm_first=True)
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers, norm=nn.LayerNorm(self.d_model))
#         self.predictor = Predictor(self.d_model, self.tgt_vocab_len)
#
#     def forward(self, enc_input, dec_input, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         # input size:  N,L
#         # output size: N,L,D_target
#         enc_input = self.encoder_embedding(enc_input)     # enc_input: N,L,D
#         dec_input = self.decoder_embedding(dec_input)     # dec_input: N,L,D
#         enc_out = self.encoder(src=enc_input, src_key_padding_mask=src_key_padding_mask)
#         dec_out = self.decoder(tgt=dec_input, memory=enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
#         return self.predictor(dec_out)