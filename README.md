# ASR project barebones

## Installation guide
Запустить install.sh: ставит либы, скачивает lm и checkpoint. Важно запускать в интерактивном режиме (`bash -i install.sh`), либо можно проделать все действия самостоятельно. Это нужно для того, чтобы успешно активировать conda environment и работать внутри него. После выполнения install.sh выведется две строки для теста (активация conda environment и сам запуск test.py на librispeech test-clean с text_encoder, использующим LM).
```shell
#!/bin/bash

conda_env_name="${1:-dla-env}"
conda create -y -n "$conda_env_name" python=3.9
conda activate "$conda_env_name"
conda info | grep "active environment"

conda install -y -c conda-forge libsndfile
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
    "https://drive.google.com/uc?export=download&id=175WxNAUunNb5dKhruk5odU9XG8JQJ98E&confirm=yes"
tar xvf 1011_002417-model_best.tar -C checkpoints

echo "Now you can run test.py:
conda activate ${conda_env_name}
python test.py -r checkpoints/1011_002417/model_best.pth -c hw_asr/configs/ds_testconfig.json"

# [OPTIONAL]
# sentencepiece model from https://bpemb.h-its.org/en/
# wget \
#     -O external/en.wiki.bpe.vs1000.model \
#     https://bpemb.h-its.org/en/en.wiki.bpe.vs1000.model
```

## Recommended implementation order

You might be a little intimidated by the number of folders and classes. Try to follow this steps to gradually undestand
the workflow.

1) Test `hw_asr/tests/test_dataset.py`  and `hw_asr/tests/test_config.py` and make sure everythin works for you
2) Implement missing functions to fix tests in  `hw_asr\tests\test_text_encoder.py`
3) Implement missing functions to fix tests in  `hw_asr\tests\test_dataloader.py`
4) Implement functions in `hw_asr\metric\utils.py`
5) Implement missing function to run `train.py` with a baseline model
6) Write your own model and try to overfit it on a single batch
7) Implement ctc beam search and add metrics to calculate WER and CER over hypothesis obtained from beam search.
8) ~~Pain and suffering~~ Implement your own models and train them. You've mastered this template when you can tune your
   experimental setup just by tuning `configs.json` file and running `train.py`
9) Don't forget to write a report about your work
10) Get hired by Google the next day

## Before submitting

0) Make sure your projects run on a new machine after complemeting the installation guide or by 
   running it in docker container.
1) Search project for `# TODO: your code here` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```
3) Make sure `test.py` works fine and works as expected. You should create files `default_test_config.json` and your
   installation guide should download your model checpoint and configs in `default_test_model/checkpoint.pth`
   and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash 
docker build -t my_hw_asr_image . 
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_asr_image python -m unittest 
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize

## TODO

These barebones can use more tests. We highly encourage students to create pull requests to add more tests / new
functionality. Current demands:

* Tests for beam search
* README section to describe folders
* Notebook to show how to work with `ConfigParser` and `config_parser.init_obj(...)`
