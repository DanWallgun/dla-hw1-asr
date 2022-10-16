#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -y -n "$conda_env_name" python=3.9
conda activate "$conda_env_name"
conda info | grep "active environment"

conda install -y -c conda-forge libsndfile
pip install --index-url=https://pypi.python.org/simple -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip

mkdir external

# LM from mozilla/DeepSpeech
wget \
    -nc \
    -O external/deepspeech-0.9.3-models.scorer \
    https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# best checkpoint https://drive.google.com/file/d/175WxNAUunNb5dKhruk5odU9XG8JQJ98E/view?usp=sharing
wget \
    -nc \
    -O 1011_002417-model_best.tar \
    "https://drive.google.com/uc?export=download&id=175WxNAUunNb5dKhruk5odU9XG8JQJ98E&confirm=yes"
tar xvf 1011_002417-model_best.tar
rm 1011_002417-model_best.tar
mv 1011_002417 default_test_model
mv default_test_model/model_best.pth default_test_model/checkpoint.pth

# [OPTIONAL, you don't need it to test the best checkpoint]
# sentencepiece model from https://bpemb.h-its.org/en/
# wget \
#     -O external/en.wiki.bpe.vs1000.model \
#     https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model

echo "Now you can run test.py:
conda activate ${conda_env_name}
python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json \
      -b 5" # default test_data contains 5 samples 
