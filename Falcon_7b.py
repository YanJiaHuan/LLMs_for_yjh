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
from datasets import Dataset, load_dataset, load_metric
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


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, trust_remote_code=True)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



# load data
Data_path = "./Data/1890/background_train.json"
data = load_dataset('json', data_files=Data_path)
data = data['train'].train_test_split(test_size=0.1)

def preprocess_function(examples):
    inputs = ["### Instruction:\n" + instruction + "\n\n" + "### Input:\n" + context + "\n\n" for instruction, context in
              zip(examples["instruction"], examples["input"])]
    model_inputs = tokenizer(inputs, padding="max_length", max_length=1024, truncation=True)
    labels = tokenizer(examples["output"], padding="max_length", max_length=1024, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

# load metric
bleu_metric = load_metric('sacrebleu')

def compute_metrics(eval_preds):
    labels = eval_preds.label_ids
    preds = eval_preds.predictions

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = [tokenizer.batch_decode(label, skip_special_tokens=True) for label in labels]
    print(decoded_preds[:5])
    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu_score": result['score']}  # Rename the score as 'bleu_score'

# train

training_args = Seq2SeqTrainingArguments(
    logging_dir="./logs_for_falcon_7b",  # Path to directory to save logs
    output_dir="./Checkpoints/falcon/1890",
    evaluation_strategy="steps",
    eval_steps=20,
    learning_rate=1e-4,
    weight_decay=1e-5,
    save_strategy='steps',
    save_steps=600,
    num_train_epochs=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    eval_accumulation_steps=1,
    fp16=True,
    predict_with_generate=True,
    logging_strategy='steps',   # Log after every X steps
    logging_steps=100           # Set X to be 100
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['test'],
    compute_metrics=compute_metrics,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# CUDA_VISIBLE_DEVICES=7 python3 Falcon_7b.py