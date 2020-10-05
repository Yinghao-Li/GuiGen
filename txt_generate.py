# -*- coding: UTF-8 -*-

import os
import numpy as np
import datetime
import argparse
import torch
import torch.utils.data
import Core.Constants as Constants
from tqdm.auto import tqdm
from Core.Utils import set_seed_everywhere, convert_instance_to_idx_seq
from Core.Dataset import read_instances_from_file
from TextGen.Dataset import Dataset, collate_fn
from TextGen.Generator import Generator


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--txt_model_path', type=str, required=True,
        help='Trained text generation model.'
    )
    parser.add_argument(
        '--bpe_model_path', type=str, default=os.path.join(Constants.MODEL_PATH, 'm.para.16000.model'),
        help='BPE encoding model location.'
    )
    parser.add_argument(
        '--test_data_path', type=str, default=os.path.join(Constants.TEST_PATH, 'test_syn_src.txt'),
        help='Path to test data file.'
    )
    parser.add_argument(
        '--dict_path', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_dict.pt'),
        help='token to index dictionary save location.'
    )
    parser.add_argument(
        '--beam_size', type=int, default=8, help='number of beams for beam search'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size'
    )
    parser.add_argument(
        '--n_best', type=int, default=1, help='output top-n generation'
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='disable cuda'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42
    )
    parser.add_argument(
        '--n_lines', type=int, default=np.inf,
        help='how many lines in the test file are used for testing. default: infinite'
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():

    args = parse_args()

    date = datetime.date.today()
    date = date.strftime('%m.%d')
    args.output = f'text.full.v-submit.{date}.txt'
    print('Arguments: ', args)
    set_seed_everywhere(args.random_seed, args.cuda)

    w2i_dict = torch.load(args.dict_path)
    dataset_args = w2i_dict['settings']
    print('Dateset arguments: ', dataset_args)
    src_txt, _, _, _, _, _, tgt_syn, tgt_lvl, _, _, _, tgt_path, _, _ = read_instances_from_file(
        data_file=args.test_data_path,
        model_file=args.bpe_model_path,
        max_txt_len=dataset_args.max_txt_len,
        src_syn_depth=dataset_args.max_src_depth,
        tmpl_syn_depth=dataset_args.max_tmpl_depth,
        max_syn_len=dataset_args.max_syn_len,
        min_txt_len=dataset_args.min_txt_len,
        min_syn_len=dataset_args.min_syn_len,
        n_lines=args.n_lines,
        use_fixed_level=True
    )

    src_txt = convert_instance_to_idx_seq(src_txt, w2i_dict['text'])
    tgt_syn = convert_instance_to_idx_seq(tgt_syn, w2i_dict['syntax'])
    tgt_lvl = convert_instance_to_idx_seq(tgt_lvl, w2i_dict['level'])

    test_loader = torch.utils.data.DataLoader(
        Dataset(
            txt_token2idx=w2i_dict['text'],
            syn_token2idx=w2i_dict['syntax'],
            lvl_token2idx=w2i_dict['level'],
            src_txt_insts=src_txt,
            syn_insts=tgt_syn,
            lvl_insts=tgt_lvl,
            path_insts=tgt_path
        ),
        num_workers=6,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False)

    generator = Generator(args)

    if not os.path.isdir('generations'):
        os.mkdir('generations')
    with open(os.path.join('generations', args.output), 'w', encoding="utf-8") as f:
        for batch in tqdm(test_loader, desc='[Generation]'):

            txt_batch, _, syn_batch, lvl_batch, txt_pos_batch, _, syn_pos_batch, path_mask_batch = batch

            text_seqs = generator.inference(
                txt_seq=txt_batch,
                syn_seq=syn_batch,
                lvl_seq=lvl_batch,
                txt_pos_seq=txt_pos_batch,
                syn_pos_seq=syn_pos_batch,
                path_mask=path_mask_batch,
            )

            for i in range(args.n_best):
                text_seq = text_seqs[i]

                text_line = ''.join([test_loader.dataset.txt_idx2token[word.item()]
                                     for word in text_seq[1:-1]]).replace('▁', ' ').strip()

                f.write(text_line + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
