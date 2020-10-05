import torch
import numpy as np
import Core.Constants as Constants
from typing import Optional


class Beam:
    """ Beam Search """

    def __init__(self, beam_size, device):
        self._beam_size = beam_size
        self._device = device
        self._done = False

        self._scores = torch.zeros((beam_size,), dtype=torch.float, device=device)

        self._syn_seq = torch.full(
            (self._beam_size, 1), Constants.PAD,
            dtype=torch.long, device=device
        )
        self._syn_seq[0] = Constants.BOS

        self._lvl_seq = torch.full(
            (self._beam_size, 1), Constants.PAD,
            dtype=torch.long, device=device
        )
        self._lvl_seq[0] = Constants.BOS

    @property
    def done(self):
        return self._done

    @property
    def syn_seq(self):
        return self._syn_seq

    @property
    def lvl_seq(self):
        return self._lvl_seq

    def syn_forward(self, syn_prob: torch.Tensor):

        n_vocab = syn_prob.size(1)

        # repeat the present scores for `n_vocab` times
        expanded_scores = self._scores.unsqueeze(1).expand_as(syn_prob)

        syn_prob[torch.isnan(syn_prob)] = -np.inf  # eliminate the bug caused by 'nan'
        new_all_scores = syn_prob + expanded_scores

        # looking for the largest scores
        flat_beam_lk = new_all_scores.view(-1)
        # best_scores: top to down, sorted
        best_scores, best_scores_id = flat_beam_lk.topk(
            k=self._beam_size, dim=0, largest=True, sorted=True)

        self._scores = best_scores

        # get the upper beam of the scores
        prev_beam_indices = best_scores_id / n_vocab
        new_syns = best_scores_id - prev_beam_indices * n_vocab

        # update word sequence and sort level sequence accordingly
        sorted_prev_word_seq = self._syn_seq[prev_beam_indices]
        self._syn_seq = torch.cat((sorted_prev_word_seq, new_syns.unsqueeze(1)), dim=1)
        self._lvl_seq = self._lvl_seq[prev_beam_indices]

        if self._syn_seq[0, -1].item() == Constants.EOS:
            self._done = True

        return self._done

    def lvl_forward(self,
                    lvl_prob: torch.Tensor,
                    lvl_scale: Optional[float] = 0.0
                    ):
        # best_scores: top to down, sorted
        best_scores, best_scores_id = lvl_prob.topk(1, dim=1, largest=True, sorted=True)

        # reorder the beams according to word prediction scores (done in word_forward())
        # sorted_prev_level_seq = self._level_seq[prev_beam_indices]

        # update 
        self._lvl_seq = torch.cat((self._lvl_seq, best_scores_id), dim=1)

        # update cumulative scores
        self._scores += lvl_scale * best_scores.squeeze()
