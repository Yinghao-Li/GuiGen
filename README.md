# Syntactically Guided Text Generation

## Requirements

- [Pytorch](https://pytorch.org/)
- [Numpy](https://numpy.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [sentencepiece](https://github.com/google/sentencepiece)

## Training

### Data Preparation

Download training and test data from [here](https://drive.google.com/drive/folders/1LanFy0BC1qC93vICXk2V2J3zcKpAio7j?usp=sharing) and copy the `train` and `test` folders into the `data` folder.

### Train Syntax Expander

If your data folder are in this directory, you can directly run the script `run_syn_train.sh`: 

```bash
./run_syn_train.sh
```

Otherwise, you need to specify `--ori_dir`, `--ref_dir` and `--dict_dir` parameters.

The trained model will be saved in the `models` folder in the name of `model.<date>.best.synlvl.chkpt`.

### Train Text Generator

Again, if you have `data` folder in your current directory, you can directly run the script `run_txt_gen_train.sh`. Otherwise you need to specify  `--ori_dir`, `--ref_dir` and `--dict_dir` parameters.

## Inference

### Generate Text with Ground Truth Target Parse

You need to edit the script `run_txt_generate.sh` before using it.

First, you need to substitute the `<date>` part in `TXT_MODEL_PATH` to the real value. Then, you may want to specify `--bpe_model_path`, `--test_data_path` and `--dict_path` if you do not have `data` folder in the current directory.

Note that if you train the text generator on **one** or **zero** GPU, you have to delete line 49 and 51 of the file `./TextGen/Generator.py`

The generated text would be saved in the folder `./generations`.

### Expand Template parse and Generate Text with the Expanded Parse

The overall operation is the same as above. The script to run is `run_txt_gen_from_tmpl.sh`.

