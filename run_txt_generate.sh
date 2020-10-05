#!/bin/bash

# --------------------------------------
# This script is used to run txt_generation.py
# --------------------------------------

# Quit if there're any errors
set -e

TXT_MODEL_PATH=models/model.<date>.best.txt.chkpt

CUDA_VISIBLE_DEVISES=6 python txt_generate.py \
	--txt_model_path "$TXT_MODEL_PATH"

