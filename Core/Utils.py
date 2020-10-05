
import random
import re
import numpy as np
import torch
import copy
import torch.nn.functional as F
from Core import Constants
from typing import Optional, List
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm


def get_non_pad_mask(seq: torch.Tensor):
    """
    Get the mask of non-padding elements (pad -> 0)

    :param seq: input sequence
    :return: the mask
    """
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(
        n_position: int, d_hid: int,
        padding_idx: Optional[int] = None) -> torch.Tensor:
    """
    Sinusoid position encoding table

    :param n_position: sequence maximum length
    :param d_hid: hidden dimension
    :param padding_idx: the index of 'padding' label

    :return position embedding
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_position_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.tensor(sinusoid_table, dtype=torch.float)


def get_attn_key_pad_mask(seq_k, seq_q):
    """
    For masking out the padding part of key sequence.
    This is part of the mask that mentioned in Figure 2 to prevent attention
    to PAD
    """

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """
    For masking out the subsequent info.
    This is part of the mask that mentioned in Figure 2 to prevent attention
    to subsequent tokens.
    """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def set_seed_everywhere(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    return None


def text_regularize(line):
    line = re.sub(r"(\S)\\", r"\1 \\", line)
    line = re.sub(r"\\(\S)", r"\\ \1", line)
    line = re.sub(r"(\S)/", r"\1 /", line)
    line = re.sub(r"/(\S)", r"/ \1", line)
    return ' '.join(word_tokenize(line.lower()))


def cal_accuracy(pred, gold):
    """ Apply label smoothing if needed """

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


# noinspection PyTypeChecker
def cal_nll_loss(pred, gold, smoothing=False):
    """ Calculate cross entropy loss, apply label smoothing if needed. """

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def cal_kl_weight(step: int,
                  k: Optional[float] = 0.0025,
                  x0: Optional[int] = 2500,
                  anneal_function: Optional = 'logistic'
                  ):
    """ Calculate the weight of KL Divergence for the final loss """
    if anneal_function == 'logistic':
        return float(1 / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1.0, step / x0)


def cal_kld(mean, logv):
    """ Calculate KL Divergence"""
    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

    return kl_loss


def build_vocab_idx(insts: List[List[int]],
                    min_token_count: Optional[int] = 0) -> dict:
    """
    Map words (tags) to its indices and trim vocab by number of occurrence

    :param insts: instances
    :param min_token_count: the minimal appearance number of a token

    :return token2idx: a dictionary convert tokens to corresponding indices
    """

    full_vocab = set(w for sent in insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    token2idx = {
        Constants.BOS_TOKEN: Constants.BOS,
        Constants.EOS_TOKEN: Constants.EOS,
        Constants.PAD_TOKEN: Constants.PAD,
        Constants.UNK_TOKEN: Constants.UNK
    }

    token_count = {w: 0 for w in full_vocab}

    for sent in insts:
        for token in sent:
            token_count[token] += 1

    ignored_token_count = 0
    for token, count in token_count.items():
        if token not in token2idx:
            if count > min_token_count:
                token2idx[token] = len(token2idx)
            else:
                ignored_token_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(token2idx)),
          'each with minimum occurrence = {}'.format(min_token_count))
    print("[Warning] Ignored token count = {}".format(ignored_token_count))
    return token2idx


def convert_instance_to_idx_seq(insts: List[List[str]],
                                token2idx: dict) -> List[List[int]]:
    """ Mapping words to idx sequence. """
    return [[token2idx.get(w, Constants.UNK) for w in s] for s in insts]


def part_shuffle(inst: List[int], shuffle_rate: Optional[float] = 0.5):
    index = np.arange(len(inst))
    inst = np.asarray(inst)
    slc = np.random.choice(index, int(round(len(inst) * shuffle_rate)))
    shf = np.random.permutation(slc)
    inst[slc] = inst[shf]
    return inst.tolist()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass


def prune_tree_by_depth(syn_insts: List[List[int]],
                        lvl_insts: List[List[int]],
                        lvl_token2idx: dict,
                        depth: int):
    print('[Info] Pruning syntax tree...')
    level_idx2token = {v: k for k, v in lvl_token2idx.items()}
    level_tokens = [[level_idx2token[idx] for idx in level_inst] for level_inst in lvl_insts]
    syn_list = []
    level_list = []
    for lev, syn in tqdm(zip(level_tokens, syn_insts), total=len(level_tokens)):
        l = np.array(lev)[1:-1].astype(np.int)
        s = np.array(syn)[1:-1]
        s = s[l <= depth]
        l = l[l <= depth]
        s = np.r_[Constants.BOS, s, Constants.EOS]
        syn_list.append(s.tolist())
        level_list.append([Constants.BOS] + [lvl_token2idx[k] for k in l.astype(str)]
                          + [Constants.EOS])

    return syn_list, level_list


def get_tree_path(lvl):
    path_idx = []
    l_stack = []
    i_stack = []
    for i, l in enumerate(lvl):
        if i == 0:
            l_stack.append(l)
            i_stack.append(i)
        else:
            if l - l_stack[-1] != 1:
                path_idx.append(copy.deepcopy(i_stack))
                l_stack = l_stack[:l - 1]
                i_stack = i_stack[:l - 1]
            i_stack.append(i)
            l_stack.append(l)

    try:
        if path_idx[-1][-1] != len(lvl) - 1:
            path_idx.append(i_stack)
    except IndexError:
        path_idx.append(i_stack)
    return path_idx


def get_tree_path_mask(path_idx, max_path_num, max_syn_len):

    path_mask = np.zeros([max_path_num, max_syn_len])
    for i, m in enumerate(path_idx):
        path_mask[i, m] = 1
    return path_mask
