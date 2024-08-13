import torch.nn as nn

from .transformer_subLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class EncoderLayer_cross(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_cross, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            dec_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer_style(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_style, self).__init__()
        self.enc_attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    # style decoder no need mask
    def forward(self, dec_input, enc_input):
        dec_output, dec_slf_attn = self.enc_attn1(
            dec_input, enc_input, enc_input, mask=None
        )
        dec_output, dec_enc_attn = self.enc_attn2(
            dec_output, enc_input, enc_input, mask=None
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output
