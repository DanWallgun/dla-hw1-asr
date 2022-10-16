# DLA HW1 ASR project

## Installation guide
Запустить install.sh: ставит либы, скачивает lm и checkpoint. Активация conda environment из bash-скрипта должна пройти гладко, но при неуспехе можно попробовать запустить в интерактивном режиме (`bash -i install.sh`), либо проделать все действия скрипта самостоятельно (он приведён ниже).  
После выполнения install.sh выведутся команды для запуска test.py (активация conda environment и сам запуск test.py на librispeech test-clean с text_encoder, использующим LM).
```shell
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
```
Опциональная часть скрипта про загрузку моделей, относящихся к BPE 
```shell
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
```


## Report

[wandb](https://wandb.ai/danwallgun/asr_project/reports/DLA-HW1-ASR-Report--VmlldzoyODAxMjMz)


## Credits

This repository is based on [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template).
Please make sure to check out its [credits](https://github.com/WrathOfGrapes/asr_project_template/#credits) as well.
