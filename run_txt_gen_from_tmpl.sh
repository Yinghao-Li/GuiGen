#!/bin/bash

# --------------------------------------
# This script is used to run txt_generation.py
# --------------------------------------

# Quit if there're any errors
set -e

TXT_MODEL_PATH=models/model.<date>.best.txt.chkpt
SYN_MODEL_PATH=models/model.<date>.best.synlvl.chkpt

CUDA_VISIBLE_DEVICES=7 python txt_gen_from_tmpl.py \
	--txt_model_path "$TXT_MODEL_PATH" \
	--syn_model_path "$SYN_MODEL_PATH"

