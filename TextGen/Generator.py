import torch
import torch.nn as nn
import torch.nn.functional as F

from TextGen.Model import MultiEncTransformer
from TextGen.Beam import Beam
from Core import Constants


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

        ckp = torch.load(args.txt_model_path)

        # network hyper-parameters
        params = ckp['settings']
        self._params = params

        # load models
        model = MultiEncTransformer(
            n_txt_token=params.n_txt_token,
            n_syn_token=params.n_syn_token,
            n_lvl_token=params.n_lvl_token,
            max_syn_len=params.max_syn_token_len,
            max_txt_len=params.max_txt_token_len,
            d_model=params.d_model,
            d_inner=params.d_inner,
            n_trf_txt_enc_layer=params.n_trf_txt_enc_layer,
            n_trf_syn_enc_layer=params.n_trf_syn_enc_layer,
            n_trf_dec_layer=params.n_trf_dec_layer,
            n_txt_attn_head=params.n_txt_attn_head,
            n_syn_attn_head=params.n_syn_attn_head,
            d_k=params.d_k,
            d_v=params.d_v,
            dropout=params.dropout,
            tgt_emb_prj_weight_sharing=params.tgt_emb_prj_weight_sharing
        )
        model = nn.DataParallel(model)
        model.load_state_dict(ckp['model'])
        model = model.module
        print('[INFO] Trained model states loaded.')

        model = model.to(self._device)
        self._model = model
        self._model.eval()

    def inference(self, txt_seq, syn_seq, lvl_seq, txt_pos_seq, syn_pos_seq, path_mask):
        """
        Predict the next word and level using the source sequences
        :param txt_seq: input text sequence as semantic guide
        :param syn_seq: syntax sequence, batch size = 1
        :param lvl_seq: level sequence, batch size = 1
        :param txt_pos_seq: text token positions, batch size = 1
        :param syn_pos_seq: syntax token positions, batch size = 1
        :param path_mask:
        :return:
        """

        assert syn_seq.size(0) == 1
        assert lvl_seq.size(0) == 1
        assert syn_pos_seq.size(0) == 1
        assert txt_seq.size(0) == 1
        assert txt_pos_seq.size(0) == 1

        with torch.no_grad():

            syn_seq = syn_seq.to(self._device)
            lvl_seq = lvl_seq.to(self._device)
            syn_pos_seq = syn_pos_seq.to(self._device)
            txt_seq = txt_seq.to(self._device)
            txt_pos_seq = txt_pos_seq.to(self._device)
            path_mask = path_mask.to(self._device)

            # ------ text inference ------

            syn_enc_output, _ = self._model.syn_encoder(
                syn_seq=syn_seq,
                lvl_seq=lvl_seq,
                pos_seq=syn_pos_seq,
                path_mask=path_mask
            )

            sz_b, len_syn, d_model = syn_enc_output.size()

            txt_enc_output, _ = self._model.txt_encoder(
                token_seq=txt_seq,
                pos_seq=txt_pos_seq
            )

            _, len_txt, _ = txt_enc_output.size()

            beam_size = self._args.beam_size
            # -- expand sequences to facilitate beam search --
            # these sequences will only be used for calculating masks
            syn_seq = syn_seq.repeat(1, beam_size).view(sz_b * beam_size, len_syn)
            txt_seq = txt_seq.repeat(1, beam_size).view(sz_b * beam_size, len_txt)

            syn_enc_output = syn_enc_output.repeat(1, beam_size, 1) \
                .view(sz_b * beam_size, len_syn, d_model)
            txt_enc_output = txt_enc_output.repeat(1, beam_size, 1) \
                .view(sz_b * beam_size, len_txt, d_model)

            # initialize decoder beam
            dec_beam = Beam(beam_size=beam_size, device=self._device)

            for len_dec_seq in range(1, self._params.max_txt_token_len + 1):

                # -- get the decoded sequences --
                tgt_txt_seq = dec_beam.txt_seq

                tgt_pos_seq = torch.tensor(
                    [[pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                     for inst in tgt_txt_seq], dtype=torch.long
                ).to(device=self._device)

                # -- calculate the probabilities for the next word --
                transformer_dec_output, _ = self._model.multi_enc_decoder(
                    tgt_seq=tgt_txt_seq,
                    tgt_pos_seq=tgt_pos_seq,
                    src_txt_seq=txt_seq,
                    src_syn_seq=syn_seq,
                    txt_enc_output=txt_enc_output,
                    syn_enc_output=syn_enc_output,
                )
                # Pick the last step: batch size * beam size \times n_vocab
                dec_output = transformer_dec_output[:, -1, :]
                # get scores
                txt_prob = F.log_softmax(self._model.tgt_word_prj(dec_output), dim=1)\
                    .view(beam_size, -1)

                # -- update beams with word probabilities --
                dec_beam.forward(txt_prob=txt_prob)

                if dec_beam.done:
                    break

            tgt_txt_seq = dec_beam.txt_seq

        return tgt_txt_seq
