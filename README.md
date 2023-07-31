# Introduction
This a self-built repo to reproduce the fine-tuning of some Large Language Models (LLM) based on certain training frameworks like DeepSpeed, bitsandbytes, and QLora.

## Hardware
NVIDIA-SMI 510.108.03   
Driver Version: 510.108.03   
CUDA Version: 11.6  
8 * NVIDIA RTX A6000 (50GB)

## Model Reproduction
|                                             Model Name                                             | Parameters | Trainable Parameters Percentage |      Methods       | Batch Size (train/evaluate) | Training Time | Inference Time |
|:--------------------------------------------------------------------------------------------------:|:----------:|:-------------------------------:|:------------------:|:---------------------------:|:-------------:|:--------------:|
|                               [T5-3B](https://huggingface.co/t5-3b)                                |     3B     |              100%               |         -          |             1/4             |   1 (base)    |    1 (base)    |
|                               [T5-3B](https://huggingface.co/t5-3b)                                |     3B     |              100%               | DeepSpeed (Zero-2) |             2/4             |      0.8      |       1        |
|                     [flan-t5-base](https://huggingface.co/google/flan-t5-base)                     |   0.248B   |               50%               |       QLora        |            36/16            |     0.04      |       -        |
|                     [gpt-neox](https://huggingface.co/EleutherAI/gpt-neox-20b)                     |    20B     |             0.0816%             |       QLora        |            4/24             |       4       |      6.1       |
|                      [llama-65B](https://huggingface.co/huggyllama/llama-65b)                      |    65B     |             0.0639%             |       QLora        |            4/24             |       4       |      6.1       |
|                                                 -                                                  |     -      |                -                |         -          |              -              |       -       |       -        |

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
### Input Examples
```text
What is the average enrollment of schools?
db_id:school_player
Table player, columns = [*,Player_ID,Player,Team,Age,Position,School_ID]
Table school, columns = [*,School_ID,School,Location,Enrollment,Founded,Denomination,Boys_or_Girls,Day_or_Boarding,Year_Entered_Competition,School_Colors]
Table school_details, columns = [*,School_ID,Nickname,Colors,League,Class,Division]
Table school_performance, columns = [*,School_Id,School_Year,Class_A,Class_AA]

foreign key:[school_details.School_ID = school.School_ID,school_performance.School_Id = school.School_ID,player.School_ID = school.School_ID]
primary key:[school.School_ID,school_details.School_ID,school_performance.School_Id,player.Player_ID]
```
Expected Output:
```sql
SELECT avg(Enrollment) FROM school
```
### Notice:
1. As the original paramter size for each model varies, better use larger when training smaller models, and smaller when training larger models.  
2. So far, the model I have tried can be trained smoothly, but the performance is terrible, I don't know whether the problem is out of training or out of this Qlora method.