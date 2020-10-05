import os
import random
import argparse
import torch
import numpy as np
import Core.Constants as Constants
from Core.Utils import build_vocab_idx, convert_instance_to_idx_seq,\
    set_seed_everywhere
from Core.Dataset import read_instances_from_file


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', type=str, default=os.path.join(Constants.DATA_PATH, 'paraphrase/integrated.txt'),
        help='dataset location.'
    )
    parser.add_argument(
        '--model_dir', type=str, default=os.path.join(Constants.MODEL_PATH, 'm.para.16000.model'),
        help='bpe model location.'
    )
    parser.add_argument(
        '--ori_save_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_ori.pt'),
        help='source training data save location.'
    )
    parser.add_argument(
        '--ref_save_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_ref.pt'),
        help='reference training data save location.'
    )
    parser.add_argument(
        '--dict_save_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_dict.pt'),
        help='token to index dictionary save location.'
    )
    parser.add_argument(
        '--max_txt_len', type=int, default=50, help='maximum sentence length'
    )
    parser.add_argument(
        '--max_syn_len', type=int, default=50, help='maximum constituency parse length'
    )
    parser.add_argument(
        '--max_src_depth', type=int, default=8, help='maximum constituency parse tree depth'
    )
    parser.add_argument(
        '--max_tmpl_depth', type=int, default=3, help='maximum template parse tree depth'
    )
    parser.add_argument(
        '--min_txt_len', type=int, default=3, help='minimum sentence length'
    )
    parser.add_argument(
        '--min_syn_len', type=int, default=3, help='minimum constituency parse length'
    )
    parser.add_argument(
        '--min_token_count', type=int, default=0, help='minimum appearance time of a token to be registered'
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.9, help='ratio of instances used for training'
    )
    parser.add_argument(
        '--keep_case', type=bool, default=False, help='whether keep the original case of a word'
    )
    parser.add_argument(
        '--n_lines', type=int, default=np.inf, help='how many lines are going to be used for training and validation'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    """ Main function """

    args = parse_args()
    set_seed_everywhere(args.random_seed, False)

    # Training set
    txt_ori, src_syn_ori, src_lvl_ori, tmpl_syn_ori, tmpl_lvl_ori, \
        txt_ref, src_syn_ref, src_lvl_ref, tmpl_syn_ref, tmpl_lvl_ref, \
        src_tree_path_ori, src_tree_path_ref, tmpl_tree_path_ori, tmpl_tree_path_ref = \
        read_instances_from_file(
            data_file=args.data_dir,
            model_file=args.model_dir,
            max_txt_len=args.max_txt_len,
            src_syn_depth=args.max_src_depth,
            tmpl_syn_depth=args.max_tmpl_depth,
            max_syn_len=args.max_syn_len,
            min_txt_len=args.min_txt_len,
            min_syn_len=args.min_syn_len,
            n_lines=args.n_lines,
            use_fixed_level=True
        )

    assert len(txt_ori) == len(src_syn_ori) == len(src_lvl_ori) == \
           len(txt_ref) == len(src_syn_ref) == len(src_lvl_ref) == \
           len(tmpl_syn_ori) == len(tmpl_lvl_ori) == len(tmpl_syn_ref) == len(tmpl_lvl_ref) == \
           len(src_tree_path_ori) == len(src_tree_path_ref) == len(tmpl_tree_path_ori) == len(tmpl_tree_path_ref)

    args.max_txt_token_len = max([len(s) for s in txt_ori + txt_ref])
    args.max_syn_token_len = max([len(s) for s in src_syn_ori + src_syn_ref])

    # shuffle data
    data_bag = list(zip(
        txt_ori, src_syn_ori, src_lvl_ori, tmpl_syn_ori, tmpl_lvl_ori,
        txt_ref, src_syn_ref, src_lvl_ref, tmpl_syn_ref, tmpl_lvl_ref,
        src_tree_path_ori, src_tree_path_ref, tmpl_tree_path_ori, tmpl_tree_path_ref
    ))
    random.shuffle(data_bag)
    txt_ori, src_syn_ori, src_lvl_ori, tmpl_syn_ori, tmpl_lvl_ori, \
        txt_ref, src_syn_ref, src_lvl_ref, tmpl_syn_ref, tmpl_lvl_ref, \
        src_tree_path_ori, src_tree_path_ref, tmpl_tree_path_ori, tmpl_tree_path_ref = zip(*data_bag)

    if os.path.exists(args.dict_save_dir):
        print('[Info] Loading word indices')
        w2i = torch.load(args.dict_save_dir)
        txt_word2idx = w2i['text']
        syn_word2idx = w2i['syntax']
        lvl_word2idx = w2i['level']
    else:
        print('[Info] Indexing words.')
        txt_word2idx = build_vocab_idx(txt_ori + txt_ref)
        syn_word2idx = build_vocab_idx(src_syn_ori + src_syn_ref)
        lvl_word2idx = build_vocab_idx(src_lvl_ori + src_lvl_ref)

    n_train_inst = int(round(args.train_ratio * len(txt_ori)))
    train_txt_ori = txt_ori[:n_train_inst]
    valid_txt_ori = txt_ori[n_train_inst:]
    train_src_syn_ori = src_syn_ori[:n_train_inst]
    valid_src_syn_ori = src_syn_ori[n_train_inst:]
    train_src_lvl_ori = src_lvl_ori[:n_train_inst]
    valid_src_lvl_ori = src_lvl_ori[n_train_inst:]
    train_tmpl_syn_ori = tmpl_syn_ori[:n_train_inst]
    valid_tmpl_syn_ori = tmpl_syn_ori[n_train_inst:]
    train_tmpl_lvl_ori = tmpl_lvl_ori[:n_train_inst]
    valid_tmpl_lvl_ori = tmpl_lvl_ori[n_train_inst:]
    train_src_path_ori = src_tree_path_ori[:n_train_inst]
    valid_src_path_ori = src_tree_path_ori[n_train_inst:]
    train_tmpl_path_ori = tmpl_tree_path_ori[:n_train_inst]
    valid_tmpl_path_ori = tmpl_tree_path_ori[n_train_inst:]

    train_txt_ref = txt_ref[:n_train_inst]
    valid_txt_ref = txt_ref[n_train_inst:]
    train_src_syn_ref = src_syn_ref[:n_train_inst]
    valid_src_syn_ref = src_syn_ref[n_train_inst:]
    train_src_lvl_ref = src_lvl_ref[:n_train_inst]
    valid_src_lvl_ref = src_lvl_ref[n_train_inst:]
    train_tmpl_syn_ref = tmpl_syn_ref[:n_train_inst]
    valid_tmpl_syn_ref = tmpl_syn_ref[n_train_inst:]
    train_tmpl_lvl_ref = tmpl_lvl_ref[:n_train_inst]
    valid_tmpl_lvl_ref = tmpl_lvl_ref[n_train_inst:]
    train_src_path_ref = src_tree_path_ref[:n_train_inst]
    valid_src_path_ref = src_tree_path_ref[n_train_inst:]
    train_tmpl_path_ref = tmpl_tree_path_ref[:n_train_inst]
    valid_tmpl_path_ref = tmpl_tree_path_ref[n_train_inst:]

    # word to index
    print('[Info] Converting instances into sequences of word index.')
    train_txt_ori = convert_instance_to_idx_seq(train_txt_ori, txt_word2idx)
    valid_txt_ori = convert_instance_to_idx_seq(valid_txt_ori, txt_word2idx)
    train_src_syn_ori = convert_instance_to_idx_seq(train_src_syn_ori, syn_word2idx)
    valid_src_syn_ori = convert_instance_to_idx_seq(valid_src_syn_ori, syn_word2idx)
    train_src_lvl_ori = convert_instance_to_idx_seq(train_src_lvl_ori, lvl_word2idx)
    valid_src_lvl_ori = convert_instance_to_idx_seq(valid_src_lvl_ori, lvl_word2idx)
    train_tmpl_syn_ori = convert_instance_to_idx_seq(train_tmpl_syn_ori, syn_word2idx)
    valid_tmpl_syn_ori = convert_instance_to_idx_seq(valid_tmpl_syn_ori, syn_word2idx)
    train_tmpl_lvl_ori = convert_instance_to_idx_seq(train_tmpl_lvl_ori, lvl_word2idx)
    valid_tmpl_lvl_ori = convert_instance_to_idx_seq(valid_tmpl_lvl_ori, lvl_word2idx)

    train_txt_ref = convert_instance_to_idx_seq(train_txt_ref, txt_word2idx)
    valid_txt_ref = convert_instance_to_idx_seq(valid_txt_ref, txt_word2idx)
    train_src_syn_ref = convert_instance_to_idx_seq(train_src_syn_ref, syn_word2idx)
    valid_src_syn_ref = convert_instance_to_idx_seq(valid_src_syn_ref, syn_word2idx)
    train_src_lvl_ref = convert_instance_to_idx_seq(train_src_lvl_ref, lvl_word2idx)
    valid_src_lvl_ref = convert_instance_to_idx_seq(valid_src_lvl_ref, lvl_word2idx)
    train_tmpl_syn_ref = convert_instance_to_idx_seq(train_tmpl_syn_ref, syn_word2idx)
    valid_tmpl_syn_ref = convert_instance_to_idx_seq(valid_tmpl_syn_ref, syn_word2idx)
    train_tmpl_lvl_ref = convert_instance_to_idx_seq(train_tmpl_lvl_ref, lvl_word2idx)
    valid_tmpl_lvl_ref = convert_instance_to_idx_seq(valid_tmpl_lvl_ref, lvl_word2idx)

    data_ori = {
        'settings': args,
        'train': {
            'text': train_txt_ori,
            'src_syntax': train_src_syn_ori,
            'src_level': train_src_lvl_ori,
            'src_path': train_src_path_ori,
            'tmpl_syntax': train_tmpl_syn_ori,
            'tmpl_level': train_tmpl_lvl_ori,
            'tmpl_path': train_tmpl_path_ori,
        },
        'valid': {
            'text': valid_txt_ori,
            'src_syntax': valid_src_syn_ori,
            'src_level': valid_src_lvl_ori,
            'src_path': valid_src_path_ori,
            'tmpl_syntax': valid_tmpl_syn_ori,
            'tmpl_level': valid_tmpl_lvl_ori,
            'tmpl_path': valid_tmpl_path_ori,
        }
    }

    data_ref = {
        'settings': args,
        'train': {
            'text': train_txt_ref,
            'src_syntax': train_src_syn_ref,
            'src_level': train_src_lvl_ref,
            'src_path': train_src_path_ref,
            'tmpl_syntax': train_tmpl_syn_ref,
            'tmpl_level': train_tmpl_lvl_ref,
            'tmpl_path': train_tmpl_path_ref,
        },
        'valid': {
            'text': valid_txt_ref,
            'src_syntax': valid_src_syn_ref,
            'src_level': valid_src_lvl_ref,
            'src_path': valid_src_path_ref,
            'tmpl_syntax': valid_tmpl_syn_ref,
            'tmpl_level': valid_tmpl_lvl_ref,
            'tmpl_path': valid_tmpl_path_ref,
        }
    }

    w2i_dict = {
        'settings': args,
        'text': txt_word2idx,
        'syntax': syn_word2idx,
        'level': lvl_word2idx
    }

    print('[Info] Dumping the processed data to pickle file')
    torch.save(data_ori, args.ori_save_dir)
    torch.save(data_ref, args.ref_save_dir)
    torch.save(w2i_dict, args.dict_save_dir)
    print('[Info] Finished.')


if __name__ == '__main__':
    main()
