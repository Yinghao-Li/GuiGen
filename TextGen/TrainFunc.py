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
    n_word_total = 0
    n_word_correct_total = 0

    for batch in tqdm(training_data, desc=' [Training] '):
        # prepare data
        src_txt_seqs, tgt_txt_seqs, syn_seqs, lvl_seqs, \
            src_txt_pos_seqs, tgt_txt_pos_seqs, syn_pos_seqs,\
            path_mask = map(lambda x: x.to(device), batch)
        gold_seqs = tgt_txt_seqs[:, 1:]

        # forward
        optimizer.zero_grad()

        txt_seqs_pred, _ = model(
            src_txt_seqs=src_txt_seqs,
            syn_seqs=syn_seqs,
            lvl_seqs=lvl_seqs,
            tgt_txt_seqs=tgt_txt_seqs,
            src_pos_seqs=src_txt_pos_seqs,
            tgt_pos_seqs=tgt_txt_pos_seqs,
            syn_pos_seqs=syn_pos_seqs,
            path_mask=path_mask
        )

        # backward
        loss = cal_nll_loss(txt_seqs_pred, gold_seqs, smoothing=smoothing)
        loss.backward()

        n_word_correct = cal_accuracy(txt_seqs_pred, gold_seqs)

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold_seqs.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct_total += n_word_correct

    ave_loss = total_loss / n_word_total
    word_accuracy = n_word_correct_total / n_word_total
    return ave_loss, word_accuracy


def eval_epoch(model,
               validation_data,
               device: torch.device
               ):
    """ Epoch operation in evaluation phase """

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct_total = 0

    with torch.no_grad():
        for batch in tqdm(validation_data, desc=' [Validation] '):
            # prepare data
            src_txt_seqs, tgt_txt_seqs, syn_seqs, lvl_seqs, \
                src_txt_pos_seqs, tgt_txt_pos_seqs, syn_pos_seqs, path_mask = map(lambda x: x.to(device), batch)
            gold_seqs = tgt_txt_seqs[:, 1:]

            # forward
            text_seqs_pred, _ = model(
                src_txt_seqs=src_txt_seqs,
                syn_seqs=syn_seqs,
                lvl_seqs=lvl_seqs,
                tgt_txt_seqs=tgt_txt_seqs,
                src_pos_seqs=src_txt_pos_seqs,
                tgt_pos_seqs=tgt_txt_pos_seqs,
                syn_pos_seqs=syn_pos_seqs,
                path_mask=path_mask
            )
            loss = cal_nll_loss(text_seqs_pred, gold_seqs)

            n_word_correct = cal_accuracy(text_seqs_pred, gold_seqs)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold_seqs.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct_total += n_word_correct

    ave_loss = total_loss / n_word_total
    word_accuracy = n_word_correct_total / n_word_total
    return ave_loss, word_accuracy


def train(model, training_data, validation_data, optimizer, device, args):
    """ Start training """

    log_train_file = None
    log_valid_file = None

    date = datetime.date.today()
    date = date.strftime('.%m.%d')

    if args.log:
        # create log file direction
        log_dir = os.path.dirname(args.log)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        log_train_file = args.log + date + '.txt.train.log'
        log_valid_file = args.log + date + '.txt.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file
        ))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,\t\tloss,\t\tppl,\t\taccuracy\n')
            log_vf.write('epoch,\t\tloss,\t\tppl,\t\taccuracy\n')

    # create model file direction
    model_dir = os.path.dirname(args.model_save)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    valid_accus = []
    for epoch_i in range(1, args.epoch + 1):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_word_accu = train_epoch(
            model=model,
            training_data=training_data,
            optimizer=optimizer,
            device=device,
            smoothing=args.label_smoothing
        )
        print('[Training] ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(train_loss, 100)),
                                                 accu=100 * train_word_accu,
                                                 elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_word_accu = eval_epoch(
            model=model,
            validation_data=validation_data,
            device=device
        )
        print('[Validation] ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(valid_loss, 100)),
                                                 accu=100 * valid_word_accu,
                                                 elapse=(time.time() - start) / 60))

        # use only word accuracy to decide whether the model should be updated
        # since word accuracy is somewhat more important than level accuracy
        valid_accus += [valid_word_accu]

        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'optimizer': optimizer_state_dict,
            'settings': args,
            'epoch': epoch_i
        }

        # ========= Save Model ========= #
        model_name = args.model_save + date + '.e{:02d}.txt.chkpt'.format(epoch_i)
        torch.save(checkpoint, model_name)
        model_name = args.model_save + date + '.best.txt.chkpt'
        if valid_word_accu >= max(valid_accus):
            torch.save(checkpoint, model_name)
            print('[Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write(
                    '{epoch},\t\t{loss: 8.5f},\t\t{ppl: 8.5f},\t\t{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=train_loss,
                        ppl=math.exp(min(train_loss, 100)), accu=100 * train_word_accu)
                )
                log_vf.write(
                    '{epoch},\t\t{loss: 8.5f},\t\t{ppl: 8.5f},\t\t{accu:3.3f}\n'.format(
                        epoch=epoch_i, loss=valid_loss,
                        ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_word_accu)
                )
