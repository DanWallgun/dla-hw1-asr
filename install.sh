#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -y -n "$conda_env_name" python=3.9
conda activate "$conda_env_name"
conda info | grep "active environment"

pip install --index-url=https://pypi.python.org/simple -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip

mkdir external
mkdir checkpoints

# LM from mozilla/DeepSpeech
wget \
    -nc \
    -O external/deepspeech-0.9.3-models.scorer \
    https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# best checkpoint https://drive.google.com/file/d/175WxNAUunNb5dKhruk5odU9XG8JQJ98E/view?usp=sharing
wget \
    -nc \
    -O 1011_002417-model_best.tar \
    "https://drive.google.com/uc?export=download&id=175WxNAUunNb5dKhruk5odU9XG8JQJ98E"
tar xvf 1011_002417-model_best.tar -C checkpoints

echo "now you can run test script
python test.py -r checkpoints/1011_002417/model_best.pth -c hw_asr/configs/ds_testconfig.json"

# [OPTIONAL]
# sentencepiece model from https://bpemb.h-its.org/en/
# wget \
#     -O external/en.wiki.bpe.vs1000.model \
#     https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model
