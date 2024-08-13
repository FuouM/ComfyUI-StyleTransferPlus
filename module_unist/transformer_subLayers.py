import math

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .transformer_modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, granular=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.query_embedding = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        
        self.granular = granular

        # option b
        self.w_qs_h = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks_h = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs_h = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc_h = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention_h = ScaledDotProductAttention(temperature=d_k**0.5)

        self.w_qs_w = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks_w = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs_w = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc_w = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention_w = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, q, k, v, mask=None):
        residual = q
        bt, l, _ = q.size()  # noqa: E741

        q = rearrange(q, "b (m1 m2) c -> b c m1 m2", m1=int(math.sqrt(l)))
        k = rearrange(k, "b (m1 m2) c -> b c m1 m2", m1=int(math.sqrt(l)))
        v = rearrange(v, "b (m1 m2) c -> b c m1 m2", m1=int(math.sqrt(l)))

        q = self.query_embedding(q)
        k_prev = self.key_embedding(k)
        v = self.value_embedding(v)

        q = rearrange(q, "b c m1 m2 -> (b m1) m2 c")
        k = rearrange(k_prev, "b c m1 m2 -> (b m1) m2 c")
        v = rearrange(v, "b c m1 m2 -> (b m1) m2 c")

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.w_qs_h(q)
        k = self.w_ks_h(k)
        v = self.w_vs_h(v)

        q = rearrange(q, "b len (head d_k) -> b len head d_k", d_k=d_k, head=n_head)
        k = rearrange(k, "b len (head d_k) -> b len head d_k", d_k=d_k, head=n_head)
        v = rearrange(v, "b len (head d_v) -> b len head d_v", d_v=d_v, head=n_head)

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention_h(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout1(self.fc_h(q))
        q = rearrange(q, "(bt h) w c ->bt (h w) c", h=int(math.sqrt(l)))
        q_mid = self.layer_norm1(q)

        q = rearrange(q_mid, "bt (h w) c ->(bt w) h c", h=int(math.sqrt(l)))
        if self.granular:
            k = rearrange(q_mid,'bt (h w) c ->(bt w) h c',h =int(math.sqrt(l)))
        else:
            k = rearrange(k_prev, "b c m1 m2 -> (b m2) m1 c")
        v = rearrange(q_mid, "bt (h w) c ->(bt w) h c", h=int(math.sqrt(l)))

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.w_qs_w(q)
        k = self.w_ks_w(k)
        v = self.w_vs_w(v)

        q = rearrange(q, "b len (head d_k) -> b len head d_k", d_k=d_k, head=n_head)
        k = rearrange(k, "b len (head d_k) -> b len head d_k", d_k=d_k, head=n_head)
        v = rearrange(v, "b len (head d_v) -> b len head d_v", d_v=d_v, head=n_head)

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # noqa: F841

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention_w(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout2(self.fc_w(q))
        q = rearrange(q, "(bt w) h c -> bt (h w) c", w=int(math.sqrt(l)))
        q += residual
        q = self.layer_norm2(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
