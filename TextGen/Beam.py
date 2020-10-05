import torch
import numpy as np
import Core.Constants as Constants


class Beam:
    """ Beam Search """

    def __init__(self, beam_size, device):
        """
        Initialization

        :param beam_size: beam size
        :param device: device
        """

        self._beam_size = beam_size
        self._device = device
        self._done = False

        self._scores = torch.zeros((beam_size,), dtype=torch.float, device=device)

        self._txt_seq = torch.full(
            (self._beam_size, 1), Constants.PAD,
            dtype=torch.long, device=device
        )
        self._txt_seq[0] = Constants.BOS

    @property
    def done(self):
        return self._done

    @property
    def txt_seq(self):
        return self._txt_seq

    @property
    def scores(self):
        return self._scores

    def forward(self, txt_prob: torch.Tensor):
        """
        Update beam status and check whether it is finished.
        :param txt_prob: beam_size \times n_vocab
        :return: whether the beam hit EOS
        """

        n_vocab = txt_prob.size(1)

        # repeat the present scores for `n_vocab` times
        expanded_scores = self._scores.unsqueeze(1).expand_as(txt_prob)

        txt_prob[torch.isnan(txt_prob)] = -np.inf  # fix abnormal output caused by 'nan'
        new_all_scores = txt_prob + expanded_scores

        # looking for the largest scores
        flat_beam_lk = new_all_scores.view(-1)
        # best_scores: top to down, sorted
        best_scores, best_scores_id = flat_beam_lk.topk(
            k=self._beam_size, dim=0, largest=True, sorted=True
        )

        self._scores = best_scores

        # get the upper beam of the scores
        prev_beam_indices = best_scores_id / n_vocab
        new_words = best_scores_id - prev_beam_indices * n_vocab

        # update word sequence and sort level sequence according to
        sorted_prev_word_seq = self._txt_seq[prev_beam_indices]
        self._txt_seq = torch.cat((sorted_prev_word_seq, new_words.unsqueeze(1)), dim=1)

        # TODO: 现在这里是得分最高的 beam 结束之后就退出全部 search
        # TODO: 后续可以改为所有 beam 全部结束才退出
        if self._txt_seq[0, -1].item() == Constants.EOS:
            self._done = True

        return self._done
