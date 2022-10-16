#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -y -n "$conda_env_name" python=3.9
conda activate "$conda_env_name"
pip install -r requirements.txt
pip install https://github.com/kpu/kenlm/archive/master.zip

mkdir external
mkdir checkpoints

# LM from mozilla/DeepSpeech
wget \
    -O external/deepspeech-0.9.3-models.scorer \
    https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer

# best checkpoint https://disk.yandex.ru/d/fNwrz8EVluqpDQ
wget -O 1011_002417-model_best.tar "https://downloader.disk.yandex.ru/disk/10243375b39bcb8d64b7ef16a00a385a3d0fdd1855e83f0125c3006fbf6366d2/634c334d/M9j18MQZ-8i3RDc98f4DPbP5w1Rj6zqyvhHyIpD4KUvrd6SmdpFdMInffxvEgTI_629DsCj6VJKIQhhUnjBfIA%3D%3D\?uid\=0\&filename\=1011_002417-model_best.tar\&disposition\=attachment\&hash\=AKpCwQNlBDIOY819bkfVm5hBgBTkD1FurULeORBzCji/5xKxYM7S/UC7WpofxP2Rq/J6bpmRyOJonT3VoXnDag%3D%3D%3A\&limit\=0\&content_type\=application%2Fx-tar\&media_type\=compressed\&tknv\=v2"
tar xvf 1011_002417-model_best.tar -C checkpoints

echo "now you can run `python test.py -r checkpoints/1011_002417/model_best.pth -c hw_asr/configs/ds_testconfig.json`"

# [OPTIONAL]
# sentencepiece model from https://bpemb.h-its.org/en/
# wget \
#     -O external/en.wiki.bpe.vs1000.model \
#     https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model
