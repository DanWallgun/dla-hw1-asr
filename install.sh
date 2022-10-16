#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -n "$conda_env_name" python=3.9
conda activate "$conda_env_name"
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip


mkdir external
mkdir checkpoints

# best checkpoint https://disk.yandex.ru/d/fNwrz8EVluqpDQ
wget "https://cloud-api.yandex.net/v1/disk/public/resources/download?fNwrz8EVluqpDQ"
tar xvf 1011_002417-model_best.tar -C checkpoints
# now you can run `python test.py -r checkpoints/1011_002417/model_best.pth -c hw_asr/configs/ds_testconfig.json`

# LM from mozilla/DeepSpeech
wget \
    -O external/deepspeech-0.9.3-models.scorer \
    https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# [OPTIONAL]
# sentencepiece model from https://bpemb.h-its.org/en/
# wget \
#     -O external/en.wiki.bpe.vs1000.model \
#     https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model