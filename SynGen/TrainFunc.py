import os
import math
import time
import torch
import datetime
import torch.utils.data
import torch.distributions
import Core.Constants as Constants

from Core.Utils import cal_accuracy, cal_nll_loss
from tqdm.auto import tqdm


def train_epoch(model,
                training_data,
                optimizer,
                device: torch.device,
                smoothing: bool
                ):
    """ Epoch operation in training phase """

    model.train()

    total_loss = 0
    kl_total_loss = 0
    n_word_total = 0
    n_syn_correct_total = 0
    n_lvl_correct_total = 0

    for batch in tqdm(training_data, desc=' [Training] ', leave=True):
        # prepare data
        src_syn, src_lvl, src_pos, src_length, src_path_mask, \
            tmpl_syn, tmpl_lvl, tmpl_pos, tmpl_length, tmpl_path_mask, \
            tgt_syn, tgt_lvl, tgt_pos, tgt_length, _ = map(lambda x: x.to(device), batch)
        syn_gold_insts = tgt_syn[:, 1:]
        lvl_gold_insts = tgt_lvl[:, 1:]

        # forward
        optimizer.zero_grad()
        syn_pred, lvl_pred, _ = model(
            src_syn_seqs=src_syn,
            src_lvl_seqs=src_lvl,
            src_pos_seqs=src_pos,
            src_path_masks=src_path_mask,
            tmpl_syn_seqs=tmpl_syn,
            tmpl_lvl_seqs=tmpl_lvl,
            tmpl_pos_seqs=tmpl_pos,
            tmpl_path_masks=tmpl_path_mask,
            tgt_syn_seqs=tgt_syn,
            tgt_lvl_seqs=tgt_lvl,
            tgt_pos_seqs=tgt_pos
        )

        # backward
        syn_reconstruction_loss = cal_nll_loss(syn_pred, syn_gold_insts, smoothing=smoothing)
        lvl_reconstruction_loss = cal_nll_loss(lvl_pred, lvl_gold_insts, smoothing=smoothing)
        kl_loss = 0
        loss = syn_reconstruction_loss + lvl_reconstruction_loss + kl_loss
        loss.backward()

        n_syn_correct = cal_accuracy(syn_pred, syn_gold_insts)
        n_lvl_correct = cal_accuracy(lvl_pred, lvl_gold_insts)

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        kl_total_loss += kl_loss

        non_pad_mask = lvl_gold_insts.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_syn_correct_total += n_syn_correct
        n_lvl_correct_total += n_lvl_correct

    loss_per_word = total_loss / n_word_total
    kl_loss_per_word = kl_total_loss / n_word_total
    syn_accuracy = n_syn_correct_total / n_word_total
    lvl_accuracy = n_lvl_correct_total / n_word_total
    return loss_per_word, kl_loss_per_word, syn_accuracy, lvl_accuracy


def eval_epoch(model,
               validation_data,
               device
               ):
    """ Epoch operation in evaluation phase """

    model.eval()

    total_loss = 0
    kl_total_loss = 0
    n_word_total = 0
    n_syn_correct_total = 0
    n_lvl_correct_total = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, desc=' [Validation] ', leave=True):
            # prepare data
            src_syn, src_lvl, src_pos, src_lengths, src_path_mask, \
                tmpl_syn, tmpl_lvl, tmpl_pos, tmpl_lengths, tmpl_path_mask, \
                tgt_syn, tgt_lvl, tgt_pos, tgt_lengths, _ = map(lambda x: x.to(device), batch)
            syn_gold_insts = tgt_syn[:, 1:]
            lvl_gold_insts = tgt_lvl[:, 1:]

            # forward
            syn_pred, lvl_pred, _ = model(
                src_syn_seqs=src_syn,
                src_lvl_seqs=src_lvl,
                src_pos_seqs=src_pos,
                src_path_masks=src_path_mask,
                tmpl_syn_seqs=tmpl_syn,
                tmpl_lvl_seqs=tmpl_lvl,
                tmpl_pos_seqs=tmpl_pos,
                tmpl_path_masks=tmpl_path_mask,
                tgt_syn_seqs=tgt_syn,
                tgt_lvl_seqs=tgt_lvl,
                tgt_pos_seqs=tgt_pos
            )

            # metrics
            syn_reconstruction_loss = cal_nll_loss(syn_pred, syn_gold_insts)
            lvl_reconstruction_loss = cal_nll_loss(lvl_pred, lvl_gold_insts)
            kl_loss = 0
            loss = syn_reconstruction_loss + lvl_reconstruction_loss + kl_loss

            n_syn_correct = cal_accuracy(syn_pred, syn_gold_insts)
            n_lvl_correct = cal_accuracy(lvl_pred, lvl_gold_insts)

            # note keeping
            total_loss += loss.item()
            kl_total_loss += kl_loss

            non_pad_mask = lvl_gold_insts.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_syn_correct_total += n_syn_correct
            n_lvl_correct_total += n_lvl_correct

    loss_per_word = total_loss / n_word_total
    kl_loss_per_word = kl_total_loss / n_word_total
    syn_accuracy = n_syn_correct_total / n_word_total
    lvl_accuracy = n_lvl_correct_total / n_word_total
    return loss_per_word, kl_loss_per_word, syn_accuracy, lvl_accuracy


def train(model, training_data, validation_data, optimizer, device, args):
    """ Start training """

    log_train_file = None
    log_valid_file = None

    date = datetime.date.today()
    date = date.strftime('.%m.%d')

    if args.log:
        log_dir = os.path.dirname(args.log)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        log_train_file = args.log + date + '.synlvl.train.log'
        log_valid_file = args.log + date + '.synlvl.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch\t\tloss\t\tkl-loss\t\tppl\t\taccuracy\n')
            log_vf.write('epoch\t\tloss\t\tkl-loss\t\tppl\t\taccuracy\n')

    valid_accus = []
    for epoch_i in range(1, args.epoch + 1):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_kl_loss, train_syn_accu, train_lvl_accu = train_epoch(
            model=model,
            training_data=training_data,
            optimizer=optimizer,
            device=device,
            smoothing=args.label_smoothing
        )
        print('[Training] kld: {kld: 8.5f}, ppl: {ppl: 8.5f}, syntax accuracy: {synaccu:3.3f} %, '
              'level accuracy: {lvlaccu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(train_loss, 100)),
                                                 kld=train_kl_loss,
                                                 synaccu=100 * train_syn_accu,
                                                 lvlaccu=100 * train_lvl_accu,
                                                 elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_kl_loss, valid_syn_accu, valid_lvl_accu = eval_epoch(
            model=model,
            validation_data=validation_data,
            device=device
        )
        print('[Validation] kld: {kld: 8.5f}, ppl: {ppl: 8.5f}, syntax accuracy: {synaccu:3.3f} %, '
              'level accuracy: {lvlaccu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(valid_loss, 100)),
                                                 kld=valid_kl_loss,
                                                 synaccu=100 * valid_syn_accu,
                                                 lvlaccu=100 * valid_lvl_accu,
                                                 elapse=(time.time() - start) / 60))

        # use only word accuracy to decide whether the model should be updated
        # since word accuracy is somewhat more important than level accuracy
        valid_accus += [valid_syn_accu]

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'settings': args,
            'epoch': epoch_i
        }

        # ========= Save Model ========= #
        model_dir = os.path.dirname(args.model_save)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # epoch_model_name = args.model_save + date + '.e{:02d}.synlvl.chkpt'.format(epoch_i)
        best_model_name = args.model_save + date + '.best.synlvl.chkpt'
        # torch.save(checkpoint, epoch_model_name)

        if valid_syn_accu >= max(valid_accus):
            torch.save(checkpoint, best_model_name)
            print('[Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(
                    '{epoch}\t\t{loss: 8.5f}\t\t{kld: 8.5f}\t\t{ppl: 8.5f}\t\t{saccu:3.3f}\t\t{laccu:3.3f}\n'.format(
                        epoch=epoch_i, loss=train_loss, kld=train_kl_loss,
                        ppl=math.exp(min(train_loss, 100)), saccu=100 * train_syn_accu, laccu=100 * train_lvl_accu)
                )
                log_vf.write(
                    '{epoch}\t\t{loss: 8.5f}\t\t{kld: 8.5f}\t\t{ppl: 8.5f}\t\t{saccu:3.3f}\t\t{laccu:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss, kld=valid_kl_loss,
                        ppl=math.exp(min(valid_loss, 100)), saccu=100 * valid_syn_accu, laccu=100 * valid_lvl_accu)
                )
