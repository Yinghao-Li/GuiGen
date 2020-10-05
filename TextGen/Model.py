import torch
import torch.nn as nn
import Core.Constants as Constants
from typing import Optional
from Core.Models import TrfEncoder, SynTrfEncoder, MultiEncDecoder


class MultiEncTransformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 n_txt_token: int,
                 n_syn_token: int,
                 n_lvl_token: int,
                 max_syn_len: int,
                 max_txt_len: int,
                 d_model: Optional[int] = 256,
                 d_inner: Optional[int] = 1024,
                 n_trf_txt_enc_layer: Optional[int] = 6,
                 n_trf_syn_enc_layer: Optional[int] = 6,
                 n_trf_dec_layer: Optional[int] = 6,
                 n_txt_attn_head: Optional[int] = 5,
                 n_syn_attn_head: Optional[int] = 3,
                 d_k: Optional[int] = 64,
                 d_v: Optional[int] = 64,
                 dropout: Optional[int] = 0.1,
                 tgt_emb_prj_weight_sharing: Optional[bool] = True,
                 txt_token_emb: Optional[torch.Tensor] = None,
                 syn_token_emb: Optional[torch.Tensor] = None,
                 lvl_token_emb: Optional[torch.Tensor] = None
                 ):

        n_head = n_txt_attn_head + n_syn_attn_head

        super().__init__()

        # build embeddings
        if txt_token_emb is None:
            self.txt_token_emb = nn.Embedding(
                n_txt_token, d_model, padding_idx=Constants.PAD
            )
        else:
            self.txt_token_emb = txt_token_emb

        if syn_token_emb is None:
            self.syn_token_emb = nn.Embedding(
                n_syn_token, d_model, padding_idx=Constants.PAD
            )
        else:
            self.syn_token_emb = syn_token_emb

        if lvl_token_emb is None:
            self.lvl_token_emb = nn.Embedding(
                n_lvl_token, d_model, padding_idx=Constants.PAD
            )
        else:
            self.lvl_token_emb = lvl_token_emb

        self.syn_encoder = SynTrfEncoder(
            token_emb=self.syn_token_emb,
            lvl_emb=self.lvl_token_emb,
            max_seq_len=max_syn_len,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_syn_enc_layer,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.txt_encoder = TrfEncoder(
            token_emb=self.txt_token_emb,
            max_seq_len=max_txt_len,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_txt_enc_layer,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.multi_enc_decoder = MultiEncDecoder(
            max_seq_len=max_txt_len,
            token_emb=self.txt_token_emb,
            d_model=d_model,
            d_inner=d_inner,
            n_layer=n_trf_dec_layer,
            n_txt_attn_head=n_txt_attn_head,
            n_syn_attn_head=n_syn_attn_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

        self.tgt_word_prj = nn.Linear(d_model, n_txt_token, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.txt_token_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self,
                src_txt_seqs,
                syn_seqs,
                lvl_seqs,
                tgt_txt_seqs,
                src_pos_seqs,
                tgt_pos_seqs,
                syn_pos_seqs,
                path_mask
                ):

        tgt_txt_seqs = tgt_txt_seqs[:, :-1]
        tgt_pos_seqs = tgt_pos_seqs[:, :-1]

        syn_enc_output, syn_enc_attn = self.syn_encoder(
            syn_seq=syn_seqs,
            lvl_seq=lvl_seqs,
            pos_seq=syn_pos_seqs,
            path_mask=path_mask
        )

        txt_enc_output, txt_enc_attn = self.txt_encoder(
            token_seq=src_txt_seqs,
            pos_seq=src_pos_seqs
        )

        multi_enc_decoder_output, (dec_slf_attn, dec_txt_attn, dec_syn_attn) = self.multi_enc_decoder(
            tgt_seq=tgt_txt_seqs,
            tgt_pos_seq=tgt_pos_seqs,
            src_txt_seq=src_txt_seqs,
            src_syn_seq=syn_seqs,
            txt_enc_output=txt_enc_output,
            syn_enc_output=syn_enc_output,
        )

        txt_logit = self.tgt_word_prj(multi_enc_decoder_output) * self.x_logit_scale

        return txt_logit.view(-1, txt_logit.size(2)), \
            (syn_enc_attn, txt_enc_attn, dec_slf_attn, dec_txt_attn, dec_syn_attn)
