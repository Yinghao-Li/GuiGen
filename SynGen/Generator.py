import torch
import torch.nn.functional as F
import Core.Constants as Constants

from SynGen.Model import MultiEncVAETransformer
from SynGen.Beam import Beam


class Generator(object):
    """ Load trained model(s) and handle the beam search """

    def __init__(self, args):
        """
        Initialization

        :param args: pre-set arguments
        """

        # load args
        self._args = args
        self._device = torch.device('cuda' if args.cuda else 'cpu')

        ckp = torch.load(args.syn_model_path)

        # network hyper-parameters
        params = ckp['settings']

        self._params = params

        # load models
        model = MultiEncVAETransformer(
            n_syn_token=params.n_syn_token,
            n_lvl_token=params.n_lvl_token,
            max_tmpl_len=params.max_token_tmpl_len,
            max_src_len=params.max_token_src_len,
            max_tgt_len=params.max_token_tgt_len,
            d_model=params.d_model,
            d_inner=params.d_inner,
            n_trf_enc_layer=params.n_trf_enc_layer,
            n_trf_dec_layer=params.n_trf_dec_layer,
            n_src_attn_head=params.n_src_attn_head,
            n_tmpl_attn_head=params.n_tmpl_attn_head,
            d_k=params.d_k,
            d_v=params.d_v,
            dropout=params.dropout,
            tgt_emb_prj_weight_sharing=params.tgt_emb_prj_weight_sharing,
            device=self._device
        )

        model.load_state_dict(ckp['model'])
        model = model.to(self._device)
        print('[INFO] Trained model states loaded.')

        self._model = model
        self._model.eval()

    def inference(self, tmpl_syn_seq, tmpl_lvl_seq, tmpl_pos_seq,
                  src_syn_seq, src_lvl_seq, src_pos_seq):

        with torch.no_grad():

            tmpl_syn_seq = tmpl_syn_seq.to(self._device)
            tmpl_lvl_seq = tmpl_lvl_seq.to(self._device)
            tmpl_pos_seq = tmpl_pos_seq.to(self._device)
            src_syn_seq = src_syn_seq.to(self._device)
            src_lvl_seq = src_lvl_seq.to(self._device)
            src_pos_seq = src_pos_seq.to(self._device)

            # ------ word  and level inference ------

            # transformer encoding of syntax
            tmpl_enc_output, _ = self._model.tmpl_trf_encoder(
                token_seq=tmpl_syn_seq,
                lvl_seq=tmpl_lvl_seq,
                pos_seq=tmpl_pos_seq
            )
            src_enc_output, _ = self._model.src_trf_encoder(
                token_seq=src_syn_seq,
                lvl_seq=src_lvl_seq,
                pos_seq=src_pos_seq
            )
            
            # batch size, source length and model dimension
            sz_b, len_src, d_model = src_enc_output.size()
            len_tmpl = tmpl_enc_output.size(1)

            # -- the transformer embeddings and vae outputs are settled --

            beam_size = self._args.beam_size
            # -- expand sequences to facilitate beam search --
            # these sequence will only be used for calculating masks
            tmpl_syn_seq = tmpl_syn_seq.repeat(1, beam_size).view(sz_b * beam_size, len_tmpl)
            src_syn_seq = src_syn_seq.repeat(1, beam_size).view(sz_b * beam_size, len_src)

            src_enc_output = src_enc_output.repeat(1, beam_size, 1) \
                .view(sz_b * beam_size, len_src, d_model)
            tmpl_enc_output = tmpl_enc_output.repeat(1, beam_size, 1) \
                .view(sz_b * beam_size, len_tmpl, d_model)

            # initialize decoder beam
            dec_beam = Beam(beam_size=beam_size, device=self._device)

            for len_dec_seq in range(1, self._params.max_token_tgt_len + 1):

                # print('operating sequence length: {}'.format(len_dec_seq))

                # -- get the decoded word and level sequences --
                prev_syn_dec_seq = dec_beam.syn_seq
                prev_lvl_dec_seq = dec_beam.lvl_seq

                tgt_pos_seq = torch.tensor(
                    [[pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                     for inst in prev_syn_dec_seq], dtype=torch.long
                ).to(device=self._device)

                # -- calculate the probabilities for the next word --
                trf_dec_output, _ = self._model.multi_enc_decoder(
                    tgt_syn_seq=prev_syn_dec_seq,
                    tgt_lvl_seq=prev_lvl_dec_seq,
                    tgt_pos_seq=tgt_pos_seq,
                    tmpl_seq=tmpl_syn_seq,
                    src_seq=src_syn_seq,
                    tmpl_enc_output=tmpl_enc_output,
                    src_enc_output=src_enc_output
                )
                # Pick the last step: batch size * beam size \times n_vocab
                dec_output = trf_dec_output[:, -1, :]
                # get scores
                syn_prob = F.log_softmax(self._model.tgt_syn_prj(dec_output), dim=1).view(beam_size, -1)

                # -- update beams with syntax probabilities --
                dec_beam.syn_forward(syn_prob=syn_prob)

                if dec_beam.done:
                    break

                # update word and level sequences
                prev_syn_dec_seq = dec_beam.syn_seq[:, :-1]
                prev_lvl_dec_seq = dec_beam.lvl_seq

                # -- calculate the probabilities for the next level --
                trf_dec_output, _ = self._model.multi_enc_decoder(
                    tgt_syn_seq=prev_syn_dec_seq,
                    tgt_lvl_seq=prev_lvl_dec_seq,
                    tgt_pos_seq=tgt_pos_seq,
                    tmpl_seq=tmpl_syn_seq,
                    src_seq=src_syn_seq,
                    tmpl_enc_output=tmpl_enc_output,
                    src_enc_output=src_enc_output
                )
                # Pick the last step: batch size * beam size \times n_level
                dec_output = trf_dec_output[:, -1, :]
                # get scores
                lvl_prob = F.log_softmax(self._model.tgt_lvl_prj(dec_output), dim=1).view(beam_size, -1)

                # -- update beams with level probabilities --
                dec_beam.lvl_forward(lvl_prob, self._args.lvl_scale)

            syn_seq = dec_beam.syn_seq
            lvl_seq = dec_beam.lvl_seq

        return syn_seq, lvl_seq
