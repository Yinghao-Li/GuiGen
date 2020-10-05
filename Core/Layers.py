""" Define the Layers """
import torch.nn as nn
import torch.nn.functional as F
from Core.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiEncAttention
from Core.Utils import get_attn_key_pad_mask


class TrfEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TrfEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class SynTrfEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SynTrfEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, path_mask=None):
        n_path = path_mask.size(1)
        s_batch, l_sent, d_model = enc_input.size()

        x = (path_mask.sum(dim=-1) != 0).sum(dim=-1).view(-1, 1, 1)

        # Mask calculation
        slf_attn_mask = slf_attn_mask.unsqueeze(1).repeat(1, n_path, 1, 1).view(s_batch*n_path, l_sent, l_sent)
        path_attn_mask = get_attn_key_pad_mask(path_mask.view(-1, l_sent), path_mask.view(-1, l_sent))
        pos = path_attn_mask.sum(dim=2)[:, 0] == l_sent
        path_attn_mask[pos, :, :] = False
        slf_attn_mask = slf_attn_mask | path_attn_mask

        expanded_input = enc_input.unsqueeze(1).repeat(1, n_path, 1, 1) * path_mask.unsqueeze(-1)
        expanded_input = expanded_input.view(s_batch*n_path, l_sent, d_model)
        enc_output, enc_slf_attn = self.slf_attn(
            expanded_input, expanded_input, expanded_input, mask=slf_attn_mask
        )

        enc_output = enc_output.view(s_batch, n_path, l_sent, d_model) * path_mask.unsqueeze(-1)
        enc_output = enc_output.sum(dim=1) / x
        enc_output *= non_pad_mask

        enc_output = enc_output.unsqueeze(1).repeat(1, n_path, 1, 1) * path_mask.unsqueeze(-1)
        enc_output = enc_output.view(s_batch*n_path, l_sent, d_model)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask
        )

        enc_output = enc_output.view(s_batch, n_path, l_sent, d_model) * path_mask.unsqueeze(-1)
        enc_output = enc_output.sum(dim=1) / x
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class SimplifiedSynTrfEncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SimplifiedSynTrfEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, path_mask=None):
        n_path = path_mask.size(1)
        x = (path_mask.sum(dim=-1) != 0).sum(dim=-1).view(-1, 1, 1)

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output *= non_pad_mask
        enc_output = enc_output.unsqueeze(1).repeat(1, n_path, 1, 1) * path_mask.unsqueeze(-1)
        enc_output = enc_output.sum(dim=1) / x

        enc_output, enc_slf_attn = self.slf_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask
        )
        enc_output *= non_pad_mask
        enc_output = enc_output.unsqueeze(1).repeat(1, n_path, 1, 1) * path_mask.unsqueeze(-1)
        enc_output = enc_output.sum(dim=1) / x

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class MultiEncDecoderLayer(nn.Module):
    """ Multi-encoder Transformer decoder layer """

    def __init__(self, d_model, d_inner, n_txt_attn_head, n_syn_attn_head, d_k, d_v,
                 dropout=0.1):
        super(MultiEncDecoderLayer, self).__init__()

        n_head = n_txt_attn_head + n_syn_attn_head
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiEncAttention(
            n_txt_attn_head, n_syn_attn_head, d_model, d_k, d_v, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self,
                dec_input,
                txt_enc_output,
                syn_enc_output,
                non_pad_mask=None,
                slf_attn_mask=None,
                dec_txt_attn_mask=None,
                dec_syn_attn_mask=None
                ):

        dec_output, slf_attn = self.slf_attn(
            q=dec_input,
            k=dec_input,
            v=dec_input,
            mask=slf_attn_mask
        )
        dec_output *= non_pad_mask

        dec_output, (txt_attn, syn_attn) = self.enc_attn(
            q=dec_output,
            k1=txt_enc_output,
            v1=txt_enc_output,
            k2=syn_enc_output,
            v2=syn_enc_output,
            txt_mask=dec_txt_attn_mask,
            syn_mask=dec_syn_attn_mask
        )
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, (slf_attn, txt_attn, syn_attn)


class TrfDecoderLayer(nn.Module):
    """ classic Transformer Decoder layer (with some new features) """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v,
                 dropout=0.1, use_dec_enc_attn=True, use_highway=False):
        super(TrfDecoderLayer, self).__init__()

        self._use_dec_enc_attn = use_dec_enc_attn
        self._use_highway = use_highway
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        if use_dec_enc_attn:
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        if use_highway:
            self.gate = nn.Linear(d_model, d_model)

    def forward(self, dec_input, enc_output, non_pad_mask=None,
                slf_attn_mask=None, dec_enc_attn_mask=None,
                highway_activ_f=F.sigmoid):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        if self._use_dec_enc_attn:
            dec_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
            dec_output *= non_pad_mask
        else:
            dec_enc_attn = None

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        if self._use_highway:
            gate_output = highway_activ_f(self.gate(dec_input))
            dec_output = gate_output * dec_output + (1-gate_output) * dec_input
            dec_output *= non_pad_mask

        return dec_output, (dec_slf_attn, dec_enc_attn)
