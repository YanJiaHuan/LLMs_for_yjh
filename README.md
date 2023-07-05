# Introduction
This a self-built repo to reproduce the fine-tuning of some Large Language Models (LLM) based on certain training frameworks like DeepSpeed, bitsandbytes, and QLora.

## Model Reproduction
|                                  Model Name                                   | Parameters | Trainable Parameters Percentage | Methods | Batch Size (train/evaluate) | Training Time | Inference Time |
|:-----------------------------------------------------------------------------:|:----------:|:-------------------------------:|:-------:|:---------------------------:|:-------------:|:--------------:|
|[T5-3B](https://huggingface.co/t5-3b)|     3B     |              100%               |    -    |             2/4             |   1 (base)    |    1 (base)    |
|[llama-65B](https://huggingface.co/huggyllama/llama-65b) |    65B     |             0.0639%             |  QLora  |            4/24             |       -       |      6.1       |
|                                       -                                       |     -      |                -                |    -    |              -              |       -       |       -        |

## Qlora
This is a method to depend on 4bit training to decrease the memory cost of LLMs (derived from int8). By introducing more information loss (which might cause performation loss) and time cost, we saved more memory
space to fit in large models (20B, 65B...).
## Requirements
```bash
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git 
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install datasets
pip install deepspeed
```
## Launch
### T5_3B.py
```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7,8 finaltest_trainer_eval.py
```

### LLAMA_65B.py
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 LLAMA_65B.py
```
