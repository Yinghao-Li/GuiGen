import numpy as np
import torch
import torch.utils.data
from typing import List, Optional
from Core import Constants
from Core.Utils import get_tree_path_mask


def collate_fn(insts):
    all_insts = list(zip(*insts))
    if len(all_insts) == 8:
        src_syn, src_lvl, src_path, tmpl_syn, tmpl_lvl, tmpl_path, tgt_syn, tgt_lvl = all_insts
        src_batch = batch_prep(src_syn, src_lvl, src_path)
        tmpl_batch = batch_prep(tmpl_syn, tmpl_lvl, tmpl_path)
        tgt_batch = batch_prep(tgt_syn, tgt_lvl)
        return (*src_batch, *tmpl_batch, *tgt_batch)
    elif len(all_insts) == 6:
        src_syn, src_lvl, src_path, tmpl_syn, tmpl_lvl, tmpl_path = all_insts
        src_batch = batch_prep(src_syn, src_lvl, src_path)
        tmpl_batch = batch_prep(tmpl_syn, tmpl_lvl, tmpl_path)
        return (*src_batch, *tmpl_batch)
    else:
        raise ValueError("The number of instances does not match.")


def batch_prep(syn_insts: List[List[int]],
               lvl_insts: List[List[int]],
               path_insts: Optional[List[List[int]]] = None
               ):

    len_batch = np.array([len(inst) for inst in syn_insts])
    max_len = np.max(len_batch)

    syn_batch = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in syn_insts]
    )

    lvl_batch = np.array([
        level + [Constants.PAD] * (max_len - len(level))
        for level in lvl_insts
    ])

    pos_batch = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in syn_batch]
    )

    syn_batch = torch.tensor(syn_batch, dtype=torch.long)
    lvl_batch = torch.tensor(lvl_batch, dtype=torch.long)
    pos_batch = torch.tensor(pos_batch, dtype=torch.long)
    len_batch = torch.tensor(len_batch, dtype=torch.long)
    path_mask_batch = torch.empty(1)

    if path_insts is not None:
        max_path_num = np.max([len(path) for path in path_insts])
        path_mask_batch = np.array(
            [get_tree_path_mask(path, max_path_num, max_len) for path in path_insts]
        )
        path_mask_batch = np.c_[np.zeros(path_mask_batch.shape[:2] + (1,)), path_mask_batch[:, :, :-1]]
        path_mask_batch = torch.tensor(path_mask_batch, dtype=torch.bool)

    return syn_batch, lvl_batch, pos_batch, len_batch, path_mask_batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 syn_token2idx: dict,
                 lvl_token2idx: dict,
                 src_syn_insts: List[List[int]],
                 src_lvl_insts: List[List[int]],
                 src_path_insts: List[List[int]],
                 tmpl_syn_insts: List[List[int]],
                 tmpl_lvl_insts: List[List[int]],
                 tmpl_path_insts: List[List[int]],
                 tgt_syn_insts: Optional[List[List[int]]] = None,
                 tgt_lvl_insts: Optional[List[List[int]]] = None
                 ):
        super().__init__()

        # training period
        assert src_syn_insts
        assert src_lvl_insts
        assert len(src_syn_insts) == len(src_lvl_insts)
        # testing / application period
        assert not tgt_syn_insts or (len(src_syn_insts) == len(tgt_syn_insts))
        assert not tgt_syn_insts or (len(tgt_syn_insts) == len(tgt_lvl_insts))

        syn_idx2token = {idx: token for token, idx in syn_token2idx.items()}
        lvl_idx2token = {idx: token for token, idx in lvl_token2idx.items()}
        self._syn_token2idx = syn_token2idx
        self._syn_idx2token = syn_idx2token
        self._lvl_token2idx = lvl_token2idx
        self._lvl_idx2token = lvl_idx2token

        self._src_syn_insts = src_syn_insts
        self._tmpl_syn_insts = tmpl_syn_insts
        self._tgt_syn_insts = tgt_syn_insts
        self._src_level_insts = src_lvl_insts
        self._tmpl_level_insts = tmpl_lvl_insts
        self._tgt_level_insts = tgt_lvl_insts
        self._src_path_insts = src_path_insts
        self._tmpl_path_insts = tmpl_path_insts

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._src_syn_insts)

    @property
    def n_syn_token(self):
        """ Property for the number of tags """
        return len(self._syn_token2idx)

    @property
    def n_lvl_token(self):
        """ Property for the number of levels """
        return len(self._lvl_token2idx)

    @property
    def syn_token2idx(self):
        """ Property for tag to indices conversion """
        return self._syn_token2idx

    @property
    def syn_idx2token(self):
        """ Property for indices to tag conversion """
        return self._syn_idx2token

    @property
    def lvl_token2idx(self):
        """ Property for level to indices conversion """
        return self._lvl_token2idx

    @property
    def lvl_idx2token(self):
        """ Property for indices to level conversion """
        return self._lvl_idx2token

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_syn_insts:
            return self._src_syn_insts[idx], self._src_level_insts[idx], self._src_path_insts[idx], \
                   self._tmpl_syn_insts[idx], self._tmpl_level_insts[idx], self._tmpl_path_insts[idx], \
                   self._tgt_syn_insts[idx], self._tgt_level_insts[idx]
        return self._src_syn_insts[idx], self._src_level_insts[idx], self._src_path_insts[idx], \
               self._tmpl_syn_insts[idx], self._tmpl_level_insts[idx], self._tmpl_path_insts[idx]
