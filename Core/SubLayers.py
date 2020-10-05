
import torch
import torch.nn as nn
import torch.nn.functional as F
from Core.Modules import ScaledDotProductAttention
from typing import Optional


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,
                 n_head: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 dropout: Optional[float] = 0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_normal_(self.w_qs.weight)
        nn.init.xavier_normal_(self.w_ks.weight)
        nn.init.xavier_normal_(self.w_vs.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.layer_norm(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        q, attn = self.attention(q, k, v, mask=mask)

        q = q.view(n_head, sz_b, len_q, d_v)
        q = q.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        q = self.dropout(self.fc(q))
        # q = self.layer_norm(q + residual)
        q += residual

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        # x = self.layer_norm(x)
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        # x += residual
        return x


class MultiEncAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self,
                 n_txt_attn_head: int,
                 n_syn_attn_head: int,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 dropout: Optional[float] = 0.1):
        super().__init__()
        n_head = n_txt_attn_head + n_syn_attn_head

        self.n_txt_attn_head = n_txt_attn_head
        self.n_syn_attn_head = n_syn_attn_head
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs1 = nn.Linear(d_model, n_txt_attn_head * d_k, bias=False)
        self.w_ks1 = nn.Linear(d_model, n_txt_attn_head * d_k, bias=False)
        self.w_vs1 = nn.Linear(d_model, n_txt_attn_head * d_v, bias=False)
        self.w_qs2 = nn.Linear(d_model, n_syn_attn_head * d_k, bias=False)
        self.w_ks2 = nn.Linear(d_model, n_syn_attn_head * d_k, bias=False)
        self.w_vs2 = nn.Linear(d_model, n_syn_attn_head * d_v, bias=False)
        nn.init.xavier_normal_(self.w_qs1.weight)
        nn.init.xavier_normal_(self.w_ks1.weight)
        nn.init.xavier_normal_(self.w_vs1.weight)
        nn.init.xavier_normal_(self.w_qs2.weight)
        nn.init.xavier_normal_(self.w_ks2.weight)
        nn.init.xavier_normal_(self.w_vs2.weight)

        self.attention1 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention2 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k1, v1, k2, v2, txt_mask=None, syn_mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        n_txt_attn_head = self.n_txt_attn_head
        n_syn_attn_head = self.n_syn_attn_head

        sz_b, len_q, _ = q.size()
        len_k1 = k1.size(1)
        len_v1 = v1.size(1)
        len_k2 = k2.size(1)
        len_v2 = v2.size(1)

        residual = q
        q = self.layer_norm(q)

        q1 = self.w_qs1(q).view(sz_b, len_q, n_txt_attn_head, d_k)
        q2 = self.w_qs2(q).view(sz_b, len_q, n_syn_attn_head, d_k)
        k1 = self.w_ks1(k1).view(sz_b, len_k1, n_txt_attn_head, d_k)
        v1 = self.w_vs1(v1).view(sz_b, len_v1, n_txt_attn_head, d_v)
        k2 = self.w_ks2(k2).view(sz_b, len_k2, n_syn_attn_head, d_k)
        v2 = self.w_vs2(v2).view(sz_b, len_v2, n_syn_attn_head, d_v)

        q1 = q1.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q2 = q2.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k1 = k1.permute(2, 0, 1, 3).contiguous().view(-1, len_k1, d_k)  # (n*b) x lk x dk
        v1 = v1.permute(2, 0, 1, 3).contiguous().view(-1, len_v1, d_v)  # (n*b) x lv x dv
        k2 = k2.permute(2, 0, 1, 3).contiguous().view(-1, len_k2, d_k)  # (n*b) x lk x dk
        v2 = v2.permute(2, 0, 1, 3).contiguous().view(-1, len_v2, d_v)  # (n*b) x lv x dv

        txt_mask = txt_mask.repeat(n_txt_attn_head, 1, 1)  # (n*b) x .. x ..
        syn_mask = syn_mask.repeat(n_syn_attn_head, 1, 1)  # (n*b) x .. x ..
        q1, txt_attn = self.attention1(q1, k1, v1, mask=txt_mask)
        q2, syn_attn = self.attention2(q2, k2, v2, mask=syn_mask)

        q1 = q1.view(n_txt_attn_head, sz_b, len_q, d_v)
        q2 = q2.view(n_syn_attn_head, sz_b, len_q, d_v)
        q = torch.cat((q1, q2), dim=0)
        q = q.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        q = self.dropout(self.fc(q))
        # q = self.layer_norm(q + residual)
        q += residual

        return q, (txt_attn, syn_attn)
