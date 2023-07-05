# Introduction
This a self-built repo to reproduce the fine-tuning of some Large Language Models (LLM) based on certain training frameworks like DeepSpeed, bitsandbytes, and QLora.

## Model Reproduction
|  Model Name  | Parameters  | Trainable Parameters percentage  | Methods |        Hardware         |
|:------------:|:-----------:|:--------------------------------:|:-------:|:-----------------------:|
|  llama-65B   |     65B     |             0.0639%              |  QLora  | 8*A6000 (batch size: 4) |
|      -       |      -      |                -                 |   -     |            -            |

## Qlora
This is a method to depend on 4bit training to decrease the memory cost of LLMs (derived from int8). By introducing more information loss (which might cause performation loss) and time cost, we saved more memory
space to fit in large models (20B, 65B...).
### Requirements
```bash
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git 
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install datasets
```
