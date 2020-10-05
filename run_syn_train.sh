#!/bin/bash

# -------------------------------------
# This script is used to run syn_gen_train.py
# -------------------------------------

# Quit if there're any errors
set -e

SEED=42
EPOCH=30
BATCH_SIZE=256
D_MODEL=256
D_INNER=1024
N_TRF_ENC_LAYER=4
N_TRF_DEC_LAYER=6
N_WARMUP_STEPS=6400

CUDA_VISIBLE_DEVICES=5 python syn_gen_train.py \
	--epoch "$EPOCH" \
	--batch_size "$BATCH_SIZE" \
	--n_trf_enc_layer "$N_TRF_ENC_LAYER" \
	--n_trf_dec_layer "$N_TRF_DEC_LAYER" \
	--n_warmup_steps "$N_WARMUP_STEPS" \
	--d_model "$D_MODEL" \
	--d_inner "$D_INNER" \
	--random_seed "$SEED" \
	--pin_memory \
	--label_smoothing \
	--tgt_emb_prj_weight_sharing

