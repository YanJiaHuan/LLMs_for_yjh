# libs
import torch
import pandas as pd
import transformers
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

# model_name
model_id = "tiiuae/falcon-7b"

# LoRA Config
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=8,  # 理论上调的越高越好，8是一个分界线
    lora_alpha=32, # 这个参数类似lr
    target_modules=["query_key_value"], # 需要影响的层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# load model and tokenizer
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="right",
    use_fast=False,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map=device)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# CUDA_VISIBLE_DEVICES=0 python3 Falcon_7b.py