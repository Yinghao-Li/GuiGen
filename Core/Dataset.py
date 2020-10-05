import numpy as np
import torch
import torch.utils.data
import sentencepiece as spm
from typing import List, Optional
from Core import Constants
from tqdm.auto import tqdm
from Core.Utils import text_regularize, get_tree_path, get_tree_path_mask


def collate_fn(insts):
    all_insts = list(zip(*insts))
    if len(all_insts) == 10:
        src_txt, src_syn, src_lvl, src_path, tmpl_syn, tmpl_lvl, tmpl_path, tgt_txt, tgt_syn, tgt_lvl = all_insts
        txt_src_batch = txt_batch_prep(src_txt)
        txt_tgt_batch = txt_batch_prep(tgt_txt)
        syn_src_batch = syn_batch_prep(src_syn, src_lvl, src_path)
        syn_tmpl_batch = syn_batch_prep(tmpl_syn, tmpl_lvl, tmpl_path)
        syn_tgt_batch = syn_batch_prep(tgt_syn, tgt_lvl)
        return (*txt_src_batch, *txt_tgt_batch, *syn_src_batch,
                *syn_tmpl_batch, *syn_tgt_batch)
    elif len(all_insts) == 7:
        src_txt, src_syn, src_lvl, src_path, tmpl_syn, tmpl_lvl, tmpl_path = all_insts
        txt_src_batch = txt_batch_prep(src_txt)
        syn_src_batch = syn_batch_prep(src_syn, src_lvl, src_path)
        syn_tmpl_batch = syn_batch_prep(tmpl_syn, tmpl_lvl, tmpl_path)
        return (*txt_src_batch, *syn_src_batch, *syn_tmpl_batch)
    else:
        raise ValueError('The number of instances does not match.')


def syn_batch_prep(syn_insts: List[List[int]],
                   lvl_insts: List[List[int]],
                   path_insts: Optional[List[List[int]]] = None
                   ):

    batch_lengths = np.array([len(inst) for inst in syn_insts])
    max_len = np.max(batch_lengths)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in syn_insts]
    )

    batch_level = np.array([
        level + [Constants.PAD] * (max_len - len(level))
        for level in lvl_insts
    ])

    batch_pos = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq]
    )

    batch_seq = torch.tensor(batch_seq, dtype=torch.long)
    batch_level = torch.tensor(batch_level, dtype=torch.long)
    batch_pos = torch.tensor(batch_pos, dtype=torch.long)
    batch_lengths = torch.tensor(batch_lengths, dtype=torch.long)

    path_mask_batch = torch.empty(1)
    if path_insts is not None:
        max_path_num = np.max([len(path) for path in path_insts])
        path_mask_batch = np.array(
            [get_tree_path_mask(path, max_path_num, max_len) for path in path_insts]
        )
        path_mask_batch = np.c_[np.zeros(path_mask_batch.shape[:2] + (1,)), path_mask_batch[:, :, :-1]]
        path_mask_batch = torch.tensor(path_mask_batch, dtype=torch.bool)

    return batch_seq, batch_level, batch_pos, batch_lengths, path_mask_batch


def txt_batch_prep(txt_insts: List[List[int]]):
    batch_lengths = np.array([len(inst) for inst in txt_insts])
    max_len = np.max(batch_lengths)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in txt_insts]
    )

    batch_pos = np.array([
        [pos_i + 1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq]
    )

    batch_seq = torch.tensor(batch_seq, dtype=torch.long)
    batch_pos = torch.tensor(batch_pos, dtype=torch.long)

    return batch_seq, batch_pos


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 txt_token2idx: dict,
                 syn_token2idx: dict,
                 lvl_token2idx: dict,
                 src_txt_insts: List[List[int]],
                 src_syn_insts: List[List[int]],
                 src_lvl_insts: List[List[int]],
                 src_path_insts: List[List[int]],
                 tmpl_syn_insts: List[List[int]],
                 tmpl_lvl_insts: List[List[int]],
                 tmpl_path_insts: List[List[int]],
                 tgt_txt_insts: Optional[List[List[int]]] = None,
                 tgt_syn_insts: Optional[List[List[int]]] = None,
                 tgt_lvl_insts: Optional[List[List[int]]] = None
                 ):
        super().__init__()

        # training period
        assert src_syn_insts
        assert src_lvl_insts
        assert src_txt_insts
        assert len(src_syn_insts) == len(src_lvl_insts)
        assert len(src_txt_insts) == len(src_syn_insts)
        # testing / application period
        assert not tgt_syn_insts or (len(src_syn_insts) == len(tgt_syn_insts))
        assert not tgt_syn_insts or (len(tgt_syn_insts) == len(tgt_lvl_insts))

        txt_idx2token = {idx: token for token, idx in txt_token2idx.items()}
        syn_idx2token = {idx: token for token, idx in syn_token2idx.items()}
        lvl_idx2token = {idx: token for token, idx in lvl_token2idx.items()}
        self._txt_token2idx = txt_token2idx
        self._txt_idx2token = txt_idx2token
        self._syn_token2idx = syn_token2idx
        self._syn_idx2token = syn_idx2token
        self._lvl_token2idx = lvl_token2idx
        self._lvl_idx2token = lvl_idx2token

        self._src_txt_insts = src_txt_insts
        self._tgt_txt_insts = tgt_txt_insts
        self._src_syn_insts = src_syn_insts
        self._tmpl_syn_insts = tmpl_syn_insts
        self._tgt_syn_insts = tgt_syn_insts
        self._src_lvl_insts = src_lvl_insts
        self._tmpl_lvl_insts = tmpl_lvl_insts
        self._tgt_lvl_insts = tgt_lvl_insts

        self._src_path_insts = src_path_insts
        self._tmpl_path_insts = tmpl_path_insts

    @property
    def n_insts(self):
        """ Property for dataset size """
        return len(self._src_syn_insts)

    @property
    def n_txt_token(self):
        """ Returns the number of text tokens """
        return len(self._txt_token2idx)

    @property
    def n_syn_token(self):
        """ Property for the number of tags """
        return len(self._syn_token2idx)

    @property
    def n_lvl_token(self):
        """ Property for the number of levels """
        return len(self._lvl_token2idx)

    @property
    def txt_token2idx(self):
        return self._txt_token2idx

    @property
    def txt_idx2token(self):
        return self._txt_idx2token

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
            return self._src_txt_insts[idx],\
                   self._src_syn_insts[idx], self._src_lvl_insts[idx], self._src_path_insts[idx],\
                   self._tmpl_syn_insts[idx], self._tmpl_lvl_insts[idx], self._tmpl_path_insts[idx], \
                   self._tgt_txt_insts[idx], self._tgt_syn_insts[idx], self._tgt_lvl_insts[idx]
        return self._src_txt_insts[idx],\
               self._src_syn_insts[idx], self._src_lvl_insts[idx], self._src_path_insts[idx], \
               self._tmpl_syn_insts[idx], self._tmpl_lvl_insts[idx], self._tmpl_path_insts[idx]


def read_instances_from_file(data_file: str,
                             model_file: str,
                             max_txt_len: Optional[int] = np.inf,
                             src_syn_depth: Optional[int] = 8,
                             tmpl_syn_depth: Optional[int] = 3,
                             max_syn_len: Optional[int] = np.inf,
                             min_txt_len: Optional[int] = 8,
                             min_syn_len: Optional[int] = 5,
                             n_lines: Optional[int] = np.inf,
                             use_fixed_level: Optional[bool] = True
                             ):

    txt_ori = []
    src_syn_ori = []
    src_lvl_ori = []
    tmpl_syn_ori = []
    tmpl_lvl_ori = []
    txt_ref = []
    src_syn_ref = []
    src_lvl_ref = []
    tmpl_syn_ref = []
    tmpl_lvl_ref = []
    src_tree_path_ori = []
    src_tree_path_ref = []
    tmpl_tree_path_ori = []
    tmpl_tree_path_ref = []
    discard_count = 0

    spp = spm.SentencePieceProcessor()
    assert spp.load(model_file)

    print("[INFO] Reading file.")
    with open(data_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
        if type(n_lines) == int:
            lines = lines[:n_lines]

    for line in tqdm(lines):
        sp = line.strip().split('\t\t')
        if len(sp) == 6:
            t1, s1, l1, t2, s2, l2 = sp
        else:
            continue

        # tokenize syntax and level
        s1 = np.array(s1.split())
        l1 = np.array(list(map(int, l1.split())))
        assert len(s1) == len(l1)

        s2 = np.array(s2.split())
        l2 = np.array(list(map(int, l2.split())))
        assert len(s2) == len(l2)

        if (0 in l1) or (0 in l2):
            continue

        # tokenize sentence
        t1 = text_regularize(t1)
        t1 = spp.encode_as_pieces(t1)
        t2 = text_regularize(t2)
        t2 = spp.encode_as_pieces(t2)

        if use_fixed_level:
            src_syn_depth = src_syn_depth
        else:
            src_syn_depth = np.random.randint(src_syn_depth - 2, src_syn_depth + 3)

        s1_src = s1[l1 <= src_syn_depth]
        l1_src = list(map(str, l1[l1 <= src_syn_depth]))
        s2_src = s2[l2 <= src_syn_depth]
        l2_src = list(map(str, l2[l2 <= src_syn_depth]))

        s1_tmpl = s1[l1 <= tmpl_syn_depth]
        l1_tmpl = list(map(str, l1[l1 <= tmpl_syn_depth]))
        s2_tmpl = s2[l2 <= tmpl_syn_depth]
        l2_tmpl = list(map(str, l2[l2 <= tmpl_syn_depth]))

        if len(s1_src) > max_syn_len or len(s1_src) < min_syn_len \
                or len(t1) > max_txt_len or len(t1) < min_txt_len:
            discard_count += 1
            continue
        if len(s2_src) > max_syn_len or len(s2_src) < min_syn_len \
                or len(t2) > max_txt_len or len(t2) < min_txt_len:
            discard_count += 1
            continue

        tp1 = get_tree_path(l1[l1 <= src_syn_depth])
        tp2 = get_tree_path(l2[l2 <= src_syn_depth])

        src_tree_path_ori.append(tp1)
        src_tree_path_ref.append(tp2)
        tmpl_tree_path_ori.append(get_tree_path(l1[l1 <= tmpl_syn_depth]))
        tmpl_tree_path_ref.append(get_tree_path(l2[l2 <= tmpl_syn_depth]))

        txt_ori += [[Constants.BOS_TOKEN] + t1 + [Constants.EOS_TOKEN]]
        src_syn_ori += [[Constants.BOS_TOKEN] + s1_src.tolist() + [Constants.EOS_TOKEN]]
        src_lvl_ori += [[Constants.BOS_TOKEN] + l1_src + [Constants.EOS_TOKEN]]
        tmpl_syn_ori += [[Constants.BOS_TOKEN] + s1_tmpl.tolist() + [Constants.EOS_TOKEN]]
        tmpl_lvl_ori += [[Constants.BOS_TOKEN] + l1_tmpl + [Constants.EOS_TOKEN]]

        txt_ref += [[Constants.BOS_TOKEN] + t2 + [Constants.EOS_TOKEN]]
        src_syn_ref += [[Constants.BOS_TOKEN] + s2_src.tolist() + [Constants.EOS_TOKEN]]
        src_lvl_ref += [[Constants.BOS_TOKEN] + l2_src + [Constants.EOS_TOKEN]]
        tmpl_syn_ref += [[Constants.BOS_TOKEN] + s2_tmpl.tolist() + [Constants.EOS_TOKEN]]
        tmpl_lvl_ref += [[Constants.BOS_TOKEN] + l2_tmpl + [Constants.EOS_TOKEN]]

    print(f'[Info] Got {len(src_syn_ori)} instances from {data_file}')

    if discard_count > 0:
        print(f'[Warning] {discard_count} instances were discarded due to exceeding length limit.')

    return txt_ori, src_syn_ori, src_lvl_ori, tmpl_syn_ori, tmpl_lvl_ori, \
           txt_ref, src_syn_ref, src_lvl_ref, tmpl_syn_ref, tmpl_lvl_ref, \
           src_tree_path_ori, src_tree_path_ref, tmpl_tree_path_ori, tmpl_tree_path_ref
