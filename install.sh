#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -y -n "$conda_env_name" python=3.9
eval "$(conda shell.bash hook)"
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

echo "Now you can run test.py:
conda activate ${conda_env_name}
python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json \
      -b 5" # default test_data contains 5 samples 

# [OPTIONAL, you don't need it to test the best checkpoint]
echo "
Do you want to download BPE ASR model and sentencepiece-model (y/n)?
WARNING. It's NOT REQUIRED to reproduce the best metrics! This option is only for completeness of report."
read answer
if [ "$answer" != "${answer#[Yy]}" ]; then 
    # sentencepiece model from https://bpemb.h-its.org/en/
    wget \
        -O external/en.wiki.bpe.vs1000.model \
        https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model
    # best bpe checkpoint https://drive.google.com/file/d/1N4j62qlWIHey7X4bw1hldOPPQ-9QxQn9/view?usp=sharing
    wget \
        -nc \
        -O 1011_034317.tar \
        "https://drive.google.com/uc?export=download&id=1N4j62qlWIHey7X4bw1hldOPPQ-9QxQn9&confirm=yes"
    tar xvf 1011_034317.tar
    rm 1011_034317.tar
    mv 1011_034317 bpe_test_model
    mv bpe_test_model/checkpoint-epoch100.pth bpe_test_model/checkpoint.pth

    echo "Now you can run test.py:
conda activate ${conda_env_name}
python test.py \
      -c bpe_test_config.json \
      -r bpe_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json \
      -b 5"
fi
