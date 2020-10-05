import torch
import torch.nn as nn
import Core.Constants as Constants
from typing import Optional
from Core.Models import MultiEncSynDecoder, TrfEncoder


class MultiEncVAETransformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 n_syn_token: int,
                 n_lvl_token: int,
                 max_tmpl_len: int,
                 max_src_len: int,
                 max_tgt_len: int,
                 d_model: Optional[int] = 64,
                 d_inner: Optional[int] = 512,
                 n_trf_enc_layer: Optional[int] = 4,
                 n_trf_dec_layer: Optional[int] = 6,
                 n_tmpl_attn_head: Optional[int] = 6,
                 n_src_attn_head: Optional[int] = 2,
                 d_k: Optional[int] = 64,
                 d_v: Optional[int] = 64,
                 dropout: Optional[int] = 0.1,
                 tgt_emb_prj_weight_sharing: Optional[bool] = True,
                 syn_emb: Optional[torch.Tensor] = None,
                 lvl_emb: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = torch.device('cpu')
                 ):
        """
        :param n_syn_token: source vocabulary size
        :param max_tmpl_len: maximum length of encoder sentence
        :param d_model: the output of all sub-layers, including embedding layer
        :param d_inner: the inner layer dimension of position-wise feed-forward network
        :param n_trf_dec_layer: number of layers (encoding & decoding)
        :param d_k: dimension of queries and keys
        :param d_v: dimension of values
        :param dropout: dropout ratio
        :param tgt_emb_prj_weight_sharing:
        :param syn_emb: pre-trained word embeddings
        """

        super().__init__()
        self.device = device
        # build embeddings
        if syn_emb is None:
            self.syn_emb = nn.Embedding(
                n_syn_token, d_model, padding_idx=Constants.PAD
            )
        else:
            self.syn_emb = syn_emb
        if lvl_emb is None:
            self.lvl_emb = nn.Embedding(
                n_lvl_token, d_model, padding_idx=Constants.PAD
            )
        else:
            self.lvl_emb = lvl_emb

        self.tmpl_trf_encoder = TrfEncoder(
            token_emb=self.syn_emb,
            lvl_emb=self.lvl_emb,
            max_seq_len=max_tmpl_len,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_enc_layer,
            n_head=n_tmpl_attn_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.src_trf_encoder = TrfEncoder(
            token_emb=self.syn_emb,
            lvl_emb=self.lvl_emb,
            max_seq_len=max_src_len,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_enc_layer,
            n_head=n_src_attn_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.multi_enc_decoder = MultiEncSynDecoder(
            max_seq_len=max_tgt_len,
            syn_emb=self.syn_emb,
            lvl_emb=self.lvl_emb,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_dec_layer,
            n_tmpl_attn_head=n_tmpl_attn_head,
            n_src_attn_head=n_src_attn_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.tgt_syn_prj = nn.Linear(d_model, n_syn_token, bias=False)
        nn.init.xavier_normal_(self.tgt_syn_prj.weight)

        self.tgt_lvl_prj = nn.Linear(d_model, n_lvl_token, bias=False)
        nn.init.xavier_normal_(self.tgt_lvl_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_syn_prj.weight = self.syn_emb.weight
            self.tgt_lvl_prj.weight = self.lvl_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self,
                src_syn_seqs,
                src_lvl_seqs,
                src_pos_seqs,
                src_path_masks,
                tmpl_syn_seqs,
                tmpl_lvl_seqs,
                tmpl_pos_seqs,
                tmpl_path_masks,
                tgt_syn_seqs,
                tgt_lvl_seqs,
                tgt_pos_seqs
                ):

        batch_size = src_lvl_seqs.size(0)

        tgt_syn_seqs = tgt_syn_seqs[:, :-1]
        tgt_lvl_seqs = tgt_lvl_seqs[:, :-1]
        tgt_pos_seqs = tgt_pos_seqs[:, :-1]

        tmpl_enc_output, tmpl_enc_attn = self.tmpl_trf_encoder(
            token_seq=tmpl_syn_seqs,
            lvl_seq=tmpl_lvl_seqs,
            pos_seq=tmpl_pos_seqs
        )

        src_enc_output, src_enc_attn = self.src_trf_encoder(
            token_seq=src_syn_seqs,
            lvl_seq=src_lvl_seqs,
            pos_seq=src_pos_seqs
        )

        transformer_dec_output, (slf_attn_list, tmpl_attn_list, src_attn_list) = self.multi_enc_decoder(
            tgt_syn_seq=tgt_syn_seqs,
            tgt_lvl_seq=tgt_lvl_seqs,
            tgt_pos_seq=tgt_pos_seqs,
            tmpl_seq=tmpl_syn_seqs,
            src_seq=src_syn_seqs,
            tmpl_enc_output=tmpl_enc_output,
            src_enc_output=src_enc_output
        )

        syn_inst_logit = self.tgt_syn_prj(transformer_dec_output) * self.x_logit_scale
        lvl_inst_logit = self.tgt_lvl_prj(transformer_dec_output) * self.x_logit_scale

        return syn_inst_logit.view(-1, syn_inst_logit.size(2)), lvl_inst_logit.view(-1, lvl_inst_logit.size(2)), \
               (tmpl_enc_attn, src_enc_attn, slf_attn_list, tmpl_attn_list, src_attn_list)
