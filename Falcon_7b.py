# libs
import torch
import pandas as pd
import transformers
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import Dataset
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
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map=device, trust_remote_code=True)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# load data
Data_path = "./Data/1890/background_train.json"
data = pd.read_json(Data_path, orient='records')
train_dataset = data.sample(frac = 0.8, random_state=42)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)
def preprocess_function(example, tokenizer):
    instruction = example['instruction']
    input_text = example['input']
    output_text = example['output']

    # Concatenate instruction and input with special tokens for separation
    instruction_str = f"Instruction: {instruction}"
    input_str = f"Context: {input_text}"

    # Tokenize the concatenated string
    tokenized_input = tokenizer(instruction_str + " " + input_str, return_tensors="pt", padding=True, truncation=True)

    # Tokenize the output string
    tokenized_output = tokenizer(output_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]

    return {
        "input_ids": tokenized_input["input_ids"].squeeze(0),  # Remove batch dimension
        "attention_mask": tokenized_input["attention_mask"].squeeze(0),  # Remove batch dimension
        "labels": tokenized_output.squeeze(0)  # Remove batch dimension
    }

train_dataset = train_dataset.map(lambda e: preprocess_function(e, tokenizer), batched=True)
test_dataset = train_dataset.map(lambda e: preprocess_function(e, tokenizer), batched=True)

# train

trainer = transformers.Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=transformers.Seq2SeqTrainingArguments(
        logging_dir="./logs_for_falcon_7b" ,     # Path to directory to save logs
        logging_strategy='steps',   # Log after every X steps
        logging_steps=10,           # Set X to be 100
        output_dir="./Checkpoints/falcon/1890",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",  # Change evaluation_strategy to "steps"
        save_strategy="steps",
        eval_steps=50,
        save_steps=100,# Add eval_steps parameter need to lower the log/eval/save steps to see the report results
        learning_rate=5e-4,
        fp16=False,
        optim="paged_adamw_8bit",
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=513,
        include_inputs_for_metrics=True,
        # deepspeed=ds_config,
    ),
)
model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
trainer.train()


# CUDA_VISIBLE_DEVICES=0 python3 Falcon_7b.py