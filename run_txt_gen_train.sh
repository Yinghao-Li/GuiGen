#!/bin/bash

# -------------------------------------
# This script is used to run txt_gen_train.py
# -------------------------------------

# Quit if there's any errors
set -e

SEED=42
EPOCH=30
BATCH_SIZE=256
N_TRF_TXT_ENC_LAYER=4
N_TRF_SYN_ENC_LAYER=3

CUDA_VISIBLE_DEVICES=0,1,2,3,7 python txt_gen_train.py \
	--epoch "$EPOCH" \
	--batch_size "$BATCH_SIZE" \
	--n_trf_txt_enc_layer "$N_TRF_TXT_ENC_LAYER" \
	--n_trf_syn_enc_layer "$N_TRF_SYN_ENC_LAYER" \
	--random_seed "$SEED" \
	--pin_memory \
	--label_smoothing \
	--tgt_emb_prj_weight_sharing

