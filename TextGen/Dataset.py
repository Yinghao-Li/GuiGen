""" Dataset Class """

import numpy as np
import torch
import torch.utils.data
from typing import List, Optional
from Core import Constants
from Core.Utils import get_tree_path_mask


def collate_fn(insts):
    """
    Principle used to construct dataloader

    :param insts: original instances
    :return: padded instances
    """
    all_insts = list(zip(*insts))
    if len(all_insts) == 5:
        src_text_insts, syn_insts, level_insts, path_insts, tgt_text_insts = all_insts
        batch = batch_prep(src_text_insts, syn_insts, level_insts, path_insts, tgt_text_insts)
    elif len(all_insts) == 4:
        src_text_insts, syn_insts, level_insts, path_insts = all_insts
        batch = batch_prep(src_text_insts, syn_insts, level_insts, path_insts)
    else:
        raise ValueError("The number of instances does not match.")
    return batch


def batch_prep(src_txt_insts: List[List[int]],
               syn_insts: List[List[int]],
               lvl_insts: List[List[int]],
               path_insts: List[List[int]],
               tgt_txt_insts: Optional[List[List[int]]] = None
               ):
    """
    Pad the instance to the max seq length in batch
    :param src_txt_insts
    :param tgt_txt_insts: the text instances
    :param syn_insts: the syntactic instances used in a batch
    :param lvl_insts: the level instances of syntactic tokens
    :param path_insts: paths
    :return: padded position indices (0 if padded);
    """

    src_max_txt_len = np.max([len(inst) for inst in src_txt_insts])
    max_syn_len = np.max(np.array([len(inst) for inst in syn_insts]))
    max_path_num = np.max([len(path) for path in path_insts])

    src_txt_batch = np.array([
        inst + [Constants.PAD] * (src_max_txt_len - len(inst))
        for inst in src_txt_insts
    ])

    syn_batch = np.array([
        inst + [Constants.PAD] * (max_syn_len - len(inst))
        for inst in syn_insts
    ])

    lvl_batch = np.array([
        level + [Constants.PAD] * (max_syn_len - len(level))
        for level in lvl_insts
    ])

    src_txt_pos_batch = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in src_txt_batch
    ])

    syn_pos_batch = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in syn_batch
    ])

    path_mask_batch = np.array(
        [get_tree_path_mask(path, max_path_num, max_syn_len) for path in path_insts]
    )
    path_mask_batch = np.c_[np.zeros(path_mask_batch.shape[:2] + (1,)), path_mask_batch[:, :, :-1]]

    if tgt_txt_insts is not None:

        tgt_max_txt_len = np.max([len(inst) for inst in tgt_txt_insts])
        tgt_txt_batch = np.array([
            inst + [Constants.PAD] * (tgt_max_txt_len - len(inst))
            for inst in tgt_txt_insts
        ])

        tgt_txt_pos_batch = np.array([
            [pos_i + 1 if w_i != Constants.PAD else 0
             for pos_i, w_i in enumerate(inst)] for inst in tgt_txt_batch
        ])

        tgt_txt_pos_batch = torch.tensor(tgt_txt_pos_batch, dtype=torch.long)
        tgt_txt_batch = torch.tensor(tgt_txt_batch, dtype=torch.long)
    else:
        tgt_txt_batch = None
        tgt_txt_pos_batch = None

    src_txt_batch = torch.tensor(src_txt_batch, dtype=torch.long)
    syn_batch = torch.tensor(syn_batch, dtype=torch.long)
    lvl_batch = torch.tensor(lvl_batch, dtype=torch.long)
    syn_pos_batch = torch.tensor(syn_pos_batch, dtype=torch.long)
    src_txt_pos_batch = torch.tensor(src_txt_pos_batch, dtype=torch.long)
    path_mask_batch = torch.tensor(path_mask_batch, dtype=torch.bool)

    return src_txt_batch, tgt_txt_batch, syn_batch, lvl_batch, \
           src_txt_pos_batch, tgt_txt_pos_batch, syn_pos_batch, path_mask_batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 txt_token2idx: dict,
                 syn_token2idx: dict,
                 lvl_token2idx: dict,
                 src_txt_insts: List[List[int]],
                 syn_insts: List[List[int]],
                 lvl_insts: List[List[int]],
                 path_insts: List[List[int]],
                 tgt_txt_insts: Optional[List[List[int]]] = None
                 ):
        """
        A wrapper class to create syntax dataset for syntax expansion training.
        """
        super().__init__()

        # training period
        assert syn_insts
        assert lvl_insts
        assert len(syn_insts) == len(lvl_insts)
        # testing / application period
        assert not tgt_txt_insts or (len(syn_insts) == len(tgt_txt_insts))

        txt_idx2token = {idx: token for token, idx in txt_token2idx.items()}
        syn_idx2token = {idx: token for token, idx in syn_token2idx.items()}
        lvl_idx2token = {idx: token for token, idx in lvl_token2idx.items()}
        self._txt_token2idx = txt_token2idx
        self._txt_idx2token = txt_idx2token
        self._syn_token2idx = syn_token2idx
        self._syn_idx2token = syn_idx2token
        self._lvl_token2idx = lvl_token2idx
        self._lvl_idx2token = lvl_idx2token

        self._src_text_insts = src_txt_insts
        self._syn_insts = syn_insts
        self._lvl_insts = lvl_insts
        self._path_insts = path_insts
        self._tgt_text_insts = tgt_txt_insts

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._syn_insts)

    @property
    def n_syn_token(self):
        """ Property for the number of syntax tags """
        return len(self._syn_token2idx)

    @property
    def n_lvl_token(self):
        """ Property for the number of levels """
        return len(self._lvl_token2idx)

    @property
    def n_txt_token(self):
        """ Property for the number of sentence tags """
        return len(self._txt_token2idx)

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

    @property
    def txt_token2idx(self):
        return self._txt_token2idx

    @property
    def txt_idx2token(self):
        return self._txt_idx2token

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_text_insts:
            return self._src_text_insts[idx], self._syn_insts[idx], \
                   self._lvl_insts[idx], self._path_insts[idx], self._tgt_text_insts[idx]
        return self._src_text_insts[idx], self._syn_insts[idx], self._lvl_insts[idx], self._path_insts[idx]
