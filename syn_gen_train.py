import os
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import Core.Constants as Constants
import argparse
from Core.Utils import set_seed_everywhere
from Core.Optim import ScheduledOptim
from SynGen.Dataset import Dataset, collate_fn
from SynGen.Model import MultiEncVAETransformer
from SynGen.TrainFunc import train


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ori_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_ori.pt'),
        help='source training data location.'
    )
    parser.add_argument(
        '--ref_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_ref.pt'),
        help='reference training data location.'
    )
    parser.add_argument(
        '--dict_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_dict.pt'),
        help='token to index dictionary save location.'
    )
    parser.add_argument(
        '--epoch', type=int, default=20, help='number of training epochs'
    )
    parser.add_argument(
        '--lr', type=float, default=2.0, help='learning rate scale factor'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size'
    )
    parser.add_argument(
        '--d_model', type=int, default=64, help='model dimension'
    )
    parser.add_argument(
        '--d_inner', type=int, default=512, help='inner dimension'
    )
    parser.add_argument(
        '--d_k', type=int, default=64, help='dimension of key and query'
    )
    parser.add_argument(
        '--d_v', type=int, default=64, help='dimension of value'
    )
    parser.add_argument(
        '--n_trf_enc_layer', type=int, default=4, help='number of Transformer encoder layers'
    )
    parser.add_argument(
        '--n_trf_dec_layer', type=int, default=6, help='number of Transformer decoder layers'
    )
    parser.add_argument(
        '--n_src_attn_head', type=int, default=4, help='number of source syntax encoder attention heads'
    )
    parser.add_argument(
        '--n_tmpl_attn_head', type=int, default=4, help='number of template encoder attention heads'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout ratio'
    )
    parser.add_argument(
        '--tgt_emb_prj_weight_sharing', action='store_true',
        help='whether share weights between embedding and projection layers'
    )
    parser.add_argument(
        '--log', type=str, default=os.path.join('logs', 'LogFile'), help='log filepath to save'
    )
    parser.add_argument(
        '--model_save', type=str, default=os.path.join('models', 'model'),
        help='the path of the model to save'
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='disable cuda'
    )
    parser.add_argument(
        '--label_smoothing', action='store_true', help='whether use label smoothing'
    )
    parser.add_argument(
        '--n_warmup_steps', type=int, default=12800, help='number of warm-up steps'
    )
    parser.add_argument(
        '--pin_memory', action='store_true', help='whether pin your cuda memory during training'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42
    )

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def main():
    """ Main Function"""

    args = parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    set_seed_everywhere(args.random_seed, args.cuda)

    # ========= Loading Dataset ========= #
    data_ori = torch.load(args.ori_dir)
    data_ref = torch.load(args.ref_dir)
    w2i_dict = torch.load(args.dict_dir)

    training_data, validation_data = prepare_dataloaders(data_ori, data_ref, w2i_dict, args)

    args.n_syn_token = training_data.dataset.n_syn_token
    args.n_lvl_token = training_data.dataset.n_lvl_token
    args.max_token_src_len = max(
        [len(s) for s in data_ori['train']['src_syntax'] + data_ori['valid']['src_syntax'] + 
         data_ref['train']['src_syntax'] + data_ref['valid']['src_syntax']]
    )
    args.max_token_tmpl_len = max(
        [len(s) for s in data_ori['train']['tmpl_syntax'] + data_ori['valid']['tmpl_syntax'] + 
         data_ref['train']['tmpl_syntax'] + data_ref['valid']['tmpl_syntax']]
    )
    args.max_token_tgt_len = args.max_token_src_len
    print(args)

    # ========= Preparing Model ========= #

    multi_enc_vae_transformer = MultiEncVAETransformer(
        n_syn_token=args.n_syn_token,
        n_lvl_token=args.n_lvl_token,
        max_tmpl_len=args.max_token_tmpl_len,
        max_src_len=args.max_token_src_len,
        max_tgt_len=args.max_token_tgt_len,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_trf_enc_layer=args.n_trf_enc_layer,
        n_trf_dec_layer=args.n_trf_dec_layer,
        n_src_attn_head=args.n_src_attn_head,
        n_tmpl_attn_head=args.n_tmpl_attn_head,
        d_k=args.d_k,
        d_v=args.d_v,
        dropout=args.dropout,
        tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
        device=device
    )

    if args.cuda and torch.cuda.device_count() > 1:
        multi_enc_vae_transformer = nn.DataParallel(multi_enc_vae_transformer).to(device)
    else:
        multi_enc_vae_transformer = multi_enc_vae_transformer.to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, multi_enc_vae_transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        args.lr, args.d_model, args.n_warmup_steps)

    train(multi_enc_vae_transformer, training_data, validation_data, optimizer, device, args)


def prepare_dataloaders(data_ori, data_ref, w2i_dict, args):
    # ========= Preparing DataLoader ========= #
    print("preparing dataloader")

    train_src_syn_ori = data_ori['train']['src_syntax']
    valid_src_syn_ori = data_ori['valid']['src_syntax']
    train_src_lvl_ori = data_ori['train']['src_level']
    valid_src_lvl_ori = data_ori['valid']['src_level']
    train_src_path_ori = data_ori['train']['src_path']
    valid_src_path_ori = data_ori['valid']['src_path']
    train_tmpl_syn_ori = data_ori['train']['tmpl_syntax']
    valid_tmpl_syn_ori = data_ori['valid']['tmpl_syntax']
    train_tmpl_lvl_ori = data_ori['train']['tmpl_level']
    valid_tmpl_lvl_ori = data_ori['valid']['tmpl_level']
    train_tmpl_path_ori = data_ori['train']['tmpl_path']
    valid_tmpl_path_ori = data_ori['valid']['tmpl_path']

    train_src_syn_ref = data_ref['train']['src_syntax']
    valid_src_syn_ref = data_ref['valid']['src_syntax']
    train_src_lvl_ref = data_ref['train']['src_level']
    valid_src_lvl_ref = data_ref['valid']['src_level']
    train_src_path_ref = data_ref['train']['src_path']
    valid_src_path_ref = data_ref['valid']['src_path']
    train_tmpl_syn_ref = data_ref['train']['tmpl_syntax']
    valid_tmpl_syn_ref = data_ref['valid']['tmpl_syntax']
    train_tmpl_lvl_ref = data_ref['train']['tmpl_level']
    valid_tmpl_lvl_ref = data_ref['valid']['tmpl_level']
    train_tmpl_path_ref = data_ref['train']['tmpl_path']
    valid_tmpl_path_ref = data_ref['valid']['tmpl_path']

    train_loader = torch.utils.data.DataLoader(
        Dataset(
            syn_token2idx=w2i_dict['syntax'],
            lvl_token2idx=w2i_dict['level'],
            src_syn_insts=train_src_syn_ori + train_src_syn_ref,
            src_lvl_insts=train_src_lvl_ori + train_src_lvl_ref,
            src_path_insts=train_src_path_ori + train_src_path_ref,
            tmpl_syn_insts=train_tmpl_syn_ref + train_tmpl_syn_ori,
            tmpl_lvl_insts=train_tmpl_lvl_ref + train_tmpl_lvl_ori,
            tmpl_path_insts=train_tmpl_path_ref + train_tmpl_path_ori,
            tgt_syn_insts=train_src_syn_ref + train_src_syn_ori,
            tgt_lvl_insts=train_src_lvl_ref + train_src_lvl_ori
        ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    valid_loader = torch.utils.data.DataLoader(
        Dataset(
            syn_token2idx=w2i_dict['syntax'],
            lvl_token2idx=w2i_dict['level'],
            src_syn_insts=valid_src_syn_ori + valid_src_syn_ref,
            src_lvl_insts=valid_src_lvl_ori + valid_src_lvl_ref,
            src_path_insts=valid_src_path_ori + valid_src_path_ref,
            tmpl_syn_insts=valid_tmpl_syn_ref + valid_tmpl_syn_ori,
            tmpl_lvl_insts=valid_tmpl_lvl_ref + valid_tmpl_lvl_ori,
            tmpl_path_insts=valid_tmpl_path_ref + valid_tmpl_path_ori,
            tgt_syn_insts=valid_src_syn_ref + valid_src_syn_ori,
            tgt_lvl_insts=valid_src_lvl_ref + valid_src_lvl_ori
        ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
