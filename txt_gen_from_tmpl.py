import os
import torch
import torch.utils.data
import numpy as np
import argparse
import datetime
from tqdm.auto import tqdm
import Core.Constants as Constants
from Core.Utils import set_seed_everywhere, convert_instance_to_idx_seq, get_tree_path, get_tree_path_mask
from Core.Dataset import Dataset, collate_fn, read_instances_from_file
from SynGen.Generator import Generator as SynGenerator
from TextGen.Generator import Generator as TxtGenerator


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--txt_model_path', type=str, required=True,
        help='Trained text generation model.'
    )
    parser.add_argument(
        '--syn_model_path', type=str, required=True,
        help='Trained constituency parse token and level expansion model.'
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
        '--tmpl_depth', type=int, default=3, help='the depth of template parse tree'
    )
    parser.add_argument(
        '--n_best', type=int, default=1, help='output top-n generation'
    )
    parser.add_argument(
        '--lvl_scale', type=float, default=0.0, help='unknown'
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
    args.output = f'text.tmpl.v3.0.{date}.txt'
    print('Arguments: ', args)

    set_seed_everywhere(args.random_seed, args.cuda)
    device = torch.device('cuda' if args.cuda else 'cpu')

    w2i_dict = torch.load(args.dict_path)
    dataset_args = w2i_dict['settings']
    print('Dateset arguments: ', dataset_args)
    txt_seq, syn_seq, lvl_seq, _, _, _, _, _, tmpl_syn_seq, tmpl_lvl_seq, src_path, _, _, tmpl_path = \
        read_instances_from_file(
            data_file=args.test_data_path,
            model_file=args.bpe_model_path,
            max_txt_len=dataset_args.max_txt_len,
            src_syn_depth=dataset_args.max_src_depth,
            tmpl_syn_depth=args.tmpl_depth,
            max_syn_len=dataset_args.max_syn_len,
            min_txt_len=dataset_args.min_txt_len,
            min_syn_len=dataset_args.min_syn_len,
            n_lines=args.n_lines,
            use_fixed_level=True
        )

    txt_seq = convert_instance_to_idx_seq(txt_seq, w2i_dict['text'])
    syn_seq = convert_instance_to_idx_seq(syn_seq, w2i_dict['syntax'])
    lvl_seq = convert_instance_to_idx_seq(lvl_seq, w2i_dict['level'])
    tmpl_syn_seq = convert_instance_to_idx_seq(tmpl_syn_seq, w2i_dict['syntax'])
    tmpl_lvl_seq = convert_instance_to_idx_seq(tmpl_lvl_seq, w2i_dict['level'])

    test_loader = torch.utils.data.DataLoader(
        Dataset(
            txt_token2idx=w2i_dict['text'],
            syn_token2idx=w2i_dict['syntax'],
            lvl_token2idx=w2i_dict['level'],
            src_txt_insts=txt_seq,
            src_syn_insts=syn_seq,
            src_lvl_insts=lvl_seq,
            src_path_insts=src_path,
            tmpl_syn_insts=tmpl_syn_seq,
            tmpl_lvl_insts=tmpl_lvl_seq,
            tmpl_path_insts=tmpl_path
        ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False)

    syn_generator = SynGenerator(args)
    txt_generator = TxtGenerator(args)

    if not os.path.isdir('generations'):
        os.mkdir('generations')
    with open(os.path.join('generations', args.output), 'w', encoding="utf-8") as f:
        for batch in tqdm(test_loader, desc='[Generation]'):

            valid_flag = False

            src_txt, src_txt_pos, src_syn, src_level, src_pos, src_len, src_path_mask, \
                temp_syn, temp_level, temp_pos, temp_len, tmpl_path_mask = batch

            while not valid_flag:
                syn_seqs, level_seqs = syn_generator.inference(
                    temp_syn, temp_level, temp_pos, src_syn, src_level, src_pos
                )

                for syn_seq, lvl_seq in zip(syn_seqs[:2], level_seqs[:2]):
                    syn = syn_seq.clone()
                    lvl = lvl_seq.clone()
                    if Constants.EOS not in syn:
                        continue
                    syn_eos_idx = torch.where(syn == Constants.EOS)[0][0]
                    syn = syn[:syn_eos_idx]
                    lvl = lvl[:syn_eos_idx]
                    if Constants.EOS in lvl:
                        continue
                    if len(syn) == len(lvl):
                        valid_flag = True
                        break
            syn = torch.cat((syn, torch.tensor([Constants.EOS], device=device)))
            lvl = torch.cat((lvl, torch.tensor([Constants.EOS], device=device)))

            syn_seq = syn
            lvl_seq = lvl

            lvl_tokens = [int(test_loader.dataset.lvl_idx2token[li.item()]) for li in lvl[1:-1]]
            pred_path = get_tree_path(lvl_tokens)
            pred_path_mask = get_tree_path_mask(pred_path, len(pred_path), len(lvl))
            pred_path_mask = torch.as_tensor(
                np.c_[np.zeros([pred_path_mask.shape[0], 1]), pred_path_mask[:, :-1]],
                dtype=torch.bool, device=device
            ).unsqueeze(0)

            syn_pos = torch.tensor(
                [pos_i + 1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(syn_seq)], device=device
            )
            syn_seq = syn_seq.unsqueeze(0)
            lvl_seq = lvl_seq.unsqueeze(0)
            syn_pos = syn_pos.unsqueeze(0)

            text_seqs = txt_generator.inference(
                txt_seq=src_txt,
                syn_seq=syn_seq,
                lvl_seq=lvl_seq,
                txt_pos_seq=src_txt_pos,
                syn_pos_seq=syn_pos,
                path_mask=pred_path_mask
            )

            for i in range(args.n_best):
                text_seq = text_seqs[i]

                text_line = ''.join([test_loader.dataset.txt_idx2token[word.item()]
                                     for word in text_seq[1:-1]]).replace('▁', ' ').strip()

                f.write(text_line + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    main()
