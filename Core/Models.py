""" Define the Transformer model """

import torch
import torch.nn as nn
from typing import Optional
from Core.Layers import TrfEncoderLayer, TrfDecoderLayer, MultiEncDecoderLayer, \
    SynTrfEncoderLayer, SimplifiedSynTrfEncoderLayer
from Core.Utils import get_sinusoid_encoding_table, get_attn_key_pad_mask, \
    get_non_pad_mask, get_subsequent_mask


class TrfEncoder(nn.Module):

    def __init__(self,
                 token_emb: nn.Embedding,
                 max_seq_len: int,
                 n_layer: int,
                 n_head: int,
                 d_k: int,
                 d_v: int,
                 d_model: int,
                 d_inner: int,
                 dropout: Optional[float] = 0.1,
                 lvl_emb: Optional[nn.Embedding] = None
                 ):

        super().__init__()
        n_position = max_seq_len + 1

        self.token_emb = token_emb
        self.lvl_emb = lvl_emb

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            TrfEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)])

    def forward(self, token_seq, pos_seq, lvl_seq=None):
        enc_attn_list = list()

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=token_seq, seq_q=token_seq)
        non_pad_mask = get_non_pad_mask(token_seq)

        # -- Forward
        if self.lvl_emb is not None:
            assert lvl_seq is not None
            enc_output = self.token_emb(token_seq) + self.lvl_emb(lvl_seq)
        else:
            enc_output = self.token_emb(token_seq)
        enc_output += self.position_enc(pos_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask
            )
            enc_attn_list += [enc_slf_attn]

        return enc_output, enc_attn_list


class SynTrfEncoder(nn.Module):

    def __init__(self,
                 token_emb: nn.Embedding,
                 lvl_emb: nn.Embedding,
                 max_seq_len: int,
                 n_layer: int,
                 n_head: int,
                 d_k: int,
                 d_v: int,
                 d_model: int,
                 d_inner: int,
                 dropout: Optional[float] = 0.1,
                 simplify: Optional[bool] = False,
                 ):

        super().__init__()
        n_position = max_seq_len + 1

        self.token_emb = token_emb
        self.lvl_emb = lvl_emb

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        if simplify:
            self.layer_stack = nn.ModuleList([
                SimplifiedSynTrfEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layer)])
        else:
            self.layer_stack = nn.ModuleList([
                SynTrfEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layer)])

    def forward(self, syn_seq, lvl_seq, pos_seq, path_mask):
        enc_attn_list = list()

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=syn_seq, seq_q=syn_seq)
        non_pad_mask = get_non_pad_mask(syn_seq)

        # -- Forward
        enc_output = self.token_emb(syn_seq) + self.lvl_emb(lvl_seq) + self.position_enc(pos_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_input=enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                path_mask=path_mask
            )
            enc_attn_list += [enc_slf_attn]

        return enc_output, enc_attn_list


class MultiEncDecoder(nn.Module):
    """ Multi-Encoder Transformer Decoder """

    def __init__(self,
                 max_seq_len: int,
                 token_emb: nn.Embedding,
                 n_layer: int,
                 n_txt_attn_head: int,
                 n_syn_attn_head: int,
                 d_k: int,
                 d_v: int,
                 d_model: int,
                 d_inner: int,
                 dropout: Optional[float] = 0.1,
                 ):

        super().__init__()

        n_position = max_seq_len + 1

        self.token_emb = token_emb

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.trf_dec_layers = nn.ModuleList([
            MultiEncDecoderLayer(
                d_model, d_inner, n_txt_attn_head, n_syn_attn_head, d_k, d_v, dropout=dropout
            ) for _ in range(n_layer)
        ])

    def forward(self,
                tgt_seq,
                tgt_pos_seq,
                src_txt_seq,
                src_syn_seq,
                txt_enc_output,
                syn_enc_output,
                ):
        slf_attn_list = list()
        txt_attn_list = list()
        syn_attn_list = list()

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        # .gt(0): greater than 0
        slf_attn_mask = (slf_attn_mask_keypad.byte() + slf_attn_mask_subseq.byte()).gt(0)

        dec_text_attn_mask = get_attn_key_pad_mask(seq_k=src_txt_seq, seq_q=tgt_seq)
        dec_syn_attn_mask = get_attn_key_pad_mask(seq_k=src_syn_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.token_emb(tgt_seq) + self.position_enc(tgt_pos_seq)

        for dec_layer in self.trf_dec_layers:
            dec_output, (slf_attn, txt_attn, syn_attn) = dec_layer(
                dec_input=dec_output,
                txt_enc_output=txt_enc_output,
                syn_enc_output=syn_enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_txt_attn_mask=dec_text_attn_mask,
                dec_syn_attn_mask=dec_syn_attn_mask
            )
            slf_attn_list.append(slf_attn)
            txt_attn_list.append(txt_attn)
            syn_attn_list.append(syn_attn)

        return dec_output, (slf_attn_list, txt_attn_list, syn_attn_list)


class MultiEncSynDecoder(nn.Module):
    """ Multi-Encoder Transformer Decoder """

    def __init__(self,
                 max_seq_len: int,
                 syn_emb: nn.Embedding,
                 lvl_emb: nn.Embedding,
                 n_layer: int,
                 n_src_attn_head: int,
                 n_tmpl_attn_head: int,
                 d_k: int,
                 d_v: int,
                 d_model: int,
                 d_inner: int,
                 dropout: Optional[float] = 0.1,
                 ):

        super().__init__()

        n_position = max_seq_len + 1

        self.syn_emb = syn_emb
        self.lvl_emb = lvl_emb

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.trf_dec_layers = nn.ModuleList([
            MultiEncDecoderLayer(
                d_model, d_inner, n_src_attn_head, n_tmpl_attn_head, d_k, d_v, dropout=dropout
            ) for _ in range(n_layer)
        ])

    def forward(self,
                tgt_syn_seq,
                tgt_lvl_seq,
                tgt_pos_seq,
                src_seq,
                tmpl_seq,
                src_enc_output,
                tmpl_enc_output,
                ):
        slf_attn_list = list()
        src_attn_list = list()
        tmpl_attn_list = list()

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_syn_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_syn_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_syn_seq, seq_q=tgt_syn_seq)
        # .gt(0): greater than 0
        slf_attn_mask = (slf_attn_mask_keypad.byte() + slf_attn_mask_subseq.byte()).gt(0)

        dec_src_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_syn_seq)
        dec_tmpl_attn_mask = get_attn_key_pad_mask(seq_k=tmpl_seq, seq_q=tgt_syn_seq)

        # -- Forward
        dec_output = self.syn_emb(tgt_syn_seq) + self.lvl_emb(tgt_lvl_seq) + self.position_enc(tgt_pos_seq)

        for dec_layer in self.trf_dec_layers:
            dec_output, (slf_attn, src_attn, tmpl_attn) = dec_layer(
                dec_input=dec_output,
                txt_enc_output=src_enc_output,
                syn_enc_output=tmpl_enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_txt_attn_mask=dec_src_attn_mask,
                dec_syn_attn_mask=dec_tmpl_attn_mask
            )
            slf_attn_list.append(slf_attn)
            src_attn_list.append(src_attn)
            tmpl_attn_list.append(tmpl_attn)

        return dec_output, (slf_attn_list, src_attn_list, tmpl_attn_list)


class TransformerDecoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self,
                 n_layers: int,
                 n_head: int,
                 d_k: int,
                 d_v: int,
                 d_model: int,
                 d_inner: int,
                 dropout: Optional[float] = 0.1,
                 ):
        """
        :param n_layers: number of layers
        :param n_head: number of heads
        :param d_k: dimension of queries and keys
        :param d_v: dimension of values
        :param d_model: the output of all sub-layers, including embedding layer
        :param d_inner: inner dimension of position-wise feed-forward network (input and
            output are both d_model)
        :param dropout: drop out ratio
        """

        super().__init__()

        self.joint_attn_stack = nn.ModuleList([
            TrfDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self,
                tgt_seq,
                src_seq,
                enc_output,
                rnn_dec_output):
        """
        :param tgt_seq: target sequence (indices)
        :param src_seq: source sequence (used for constructing masks)
        :param enc_output: encoder output
        :param rnn_dec_output: rnn decoder output
        :return: (encoder output, attention matrices/None)
        """

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        # .gt(0): greater than 0
        slf_attn_mask = (slf_attn_mask_keypad.byte() + slf_attn_mask_subseq.byte()).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = rnn_dec_output

        for dec_layer in self.joint_attn_stack:
            dec_output, (dec_slf_attn, dec_enc_attn) = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )

            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        return dec_output, (dec_slf_attn_list, dec_enc_attn_list)


class RNNEncoder(nn.Module):
    def __init__(self,
                 syn_emb: nn.Embedding,
                 lvl_emb: nn.Embedding,
                 d_model: int,
                 n_layers: Optional[int] = 1,
                 dropout: Optional[float] = 0,
                 ):
        """
        RNN Encoder
        :param d_model: RNN hidden size (equals to d_model)
        :param n_layers: RNN layers
        :param dropout: RNN dropout ratio
        :param syn_emb: (pre-set) word embedding
        :param lvl_emb: (pre-set) level embedding
        """
        super(RNNEncoder, self).__init__()

        # construct word embedding and level embedding
        self.syn_emb = syn_emb
        self.lvl_emb = lvl_emb
        self.hidden_size = d_model

        # Initialize GRU
        self.rnn = nn.GRU(
            d_model, d_model, n_layers,
            batch_first=True,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True
        )

    def forward(self,
                syn_seq: torch.Tensor,
                lvl_seq: torch.Tensor,
                seq_lengths: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ):
        """
        :param syn_seq: input word sequence
        :param lvl_seq: input level sequence
        :param seq_lengths: lengths of sequences in a batch
            (both word and level, without padding)
        :param hidden: Initial hidden state
        :return: (every hidden state in the last layer,
            the last hidden state for every layer)
        """
        # Convert indexes to embeddings
        syn_embedding = self.syn_emb(syn_seq)
        lvl_embedding = self.lvl_emb(lvl_seq)
        embedding = syn_embedding + lvl_embedding
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            embedding, seq_lengths, batch_first=True, enforce_sorted=False
        )
        # Forward pass through GRU
        outputs, hidden = self.rnn(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


class RNNDecoder(nn.Module):
    def __init__(self,
                 syn_emb: nn.Embedding,
                 lvl_emb: nn.Embedding,
                 d_model: int,
                 n_layer: Optional[int] = 1,
                 dropout: Optional[float] = 0,
                 ):
        """
        RNN Decoder
        :param d_model: RNN hidden size (equals to d_model)
        :param n_layer: RNN layers
        :param dropout: RNN dropout ratio
        :param syn_emb: (pre-set) word embedding
        :param lvl_emb: (pre-set) level embedding
        """
        super(RNNDecoder, self).__init__()

        # construct word embedding and level embedding
        self.word_emb = syn_emb
        self.level_emb = lvl_emb
        self.hidden_size = d_model

        # Initialize GRU
        self.rnn = nn.GRU(
            d_model, d_model, n_layer,
            batch_first=True,
            dropout=(0 if n_layer == 1 else dropout),
            bidirectional=False
        )

    def forward(self,
                syn_seq: torch.Tensor,
                lvl_seq: torch.Tensor,
                seq_lengths: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ):
        """
        :param syn_seq: input word sequence
        :param lvl_seq: input level sequence
        :param seq_lengths: lengths of sequences in a batch
            (both word and level, without padding)
        :param hidden: Initial hidden state
        :return: (every hidden state in the last layer,
            the last hidden state for every layer)
        """
        # Convert indexes to embeddings
        word_embedding = self.word_emb(syn_seq)
        level_embedding = self.level_emb(lvl_seq)
        embedding = word_embedding + level_embedding
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(
            embedding, seq_lengths, batch_first=True, enforce_sorted=False
        )
        # Forward pass through GRU
        outputs, hidden = self.rnn(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Return output and final hidden state
        return outputs, hidden
