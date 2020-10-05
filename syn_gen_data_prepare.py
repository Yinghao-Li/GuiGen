import os
import torch
import argparse
import torch.utils.data
import Core.Constants as Constants
from Core.Utils import prune_tree_by_depth


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ori_load_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_bpe_train_ori.pt'),
        help='source training data direction.'
    )
    parser.add_argument(
        '--ref_load_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_bpe_train_ref.pt'),
        help='reference training data direction.'
    )
    parser.add_argument(
        '--dict_load_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_bpe_train_dict.pt'),
        help='token to index dictionary direction.'
    )
    parser.add_argument(
        '--data_save_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_syn_train.pt'),
        help='training and validation data save direction.'
    )
    parser.add_argument(
        '--tmpl_depth', type=int, default=3, help='the depth of template parse tree'
    )

    args = parser.parse_args()
    print(args)
    return args


def main():
    """ Main Function"""

    args = parse_args()

    print('[Info] Reading files...')
    data_ori = torch.load(args.ori_load_dir)
    data_ref = torch.load(args.ref_load_dir)
    w2i_dict = torch.load(args.dict_load_dir)
    
    train_syn_src_insts = data_ori['train']['syntax_insts'] + data_ref['train']['syntax_insts']
    train_lvl_src_insts = data_ori['train']['level_insts'] + data_ref['train']['level_insts']
    valid_syn_src_insts = data_ori['valid']['syntax_insts'] + data_ref['valid']['syntax_insts']
    valid_lvl_src_insts = data_ori['valid']['level_insts'] + data_ref['valid']['level_insts']
    
    train_syn_tgt_insts = data_ref['train']['syntax_insts'] + data_ori['train']['syntax_insts']
    train_lvl_tgt_insts = data_ref['train']['level_insts'] + data_ori['train']['level_insts']
    valid_syn_tgt_insts = data_ref['valid']['syntax_insts'] + data_ori['valid']['syntax_insts']
    valid_lvl_tgt_insts = data_ref['valid']['level_insts'] + data_ori['valid']['level_insts']

    train_syn_tmpl_insts, train_lvl_tmpl_insts = prune_tree_by_depth(
        train_syn_tgt_insts, train_lvl_tgt_insts, w2i_dict['level'], args.tmpl_depth
    )
    valid_syn_tmpl_insts, valid_lvl_tmpl_insts = prune_tree_by_depth(
        valid_syn_tgt_insts, valid_lvl_tgt_insts, w2i_dict['level'], args.tmpl_depth
    )
    
    data_out = {
        'settings': args,
        'train': {
            'src_syntax': train_syn_src_insts,
            'src_level': train_lvl_src_insts,
            'tmpl_syntax': train_syn_tmpl_insts,
            'tmpl_level': train_lvl_tmpl_insts,
            'tgt_syntax': train_syn_tgt_insts,
            'tgt_level': train_lvl_tgt_insts,
        },
        'valid': {
            'src_syntax': valid_syn_src_insts,
            'src_level': valid_lvl_src_insts,
            'tmpl_syntax': valid_syn_tmpl_insts,
            'tmpl_level': valid_lvl_tmpl_insts,
            'tgt_syntax': valid_syn_tgt_insts,
            'tgt_level': valid_lvl_tgt_insts,
        }
    }

    print('[Info] Dumping the processed data to pickle file...')
    torch.save(data_out, args.data_save_dir)
    print('[Info] Finished.')


if __name__ == '__main__':
    main()
