#### LLAMA-65B ####
import torch
import nltk
import os
import json
import logging
import transformers
from transformers import (
    AutoTokenizer,
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
)
from datasets.arrow_dataset import Dataset
import pandas as pd
import numpy as np
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_config,
    PeftModelForSeq2SeqLM
)
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from typing import Optional, Dict, Sequence
import sys
sys.path.append('./Evaluation_metric/spider/')
from Evaluation_self import evaluate,evaluate_test
import re

##### START #####

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

##### Load model #####
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "EleutherAI/gpt-neox-20b"

model = GPTNeoXForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

##### Load tokenizer #####
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_id)
def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
print('add special tokens...')
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer = tokenizer,
        model = model,
    )
##### Load model as Qlora setup #####
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# max_memory = get_balanced_memory(
#     model,
#     max_memory=None,
#     no_split_module_classes=["GPTNeoXLayer", "GPTNeoXMLP"],
#     dtype='float16',
#     low_zero=False,
# )
# device_map = infer_auto_device_map(
#     model,
#     max_memory=max_memory,
#     no_split_module_classes=["GPTNeoXLayer", "GPTNeoXMLP"],
#     dtype='float16'
# )
# model = dispatch_model(model, device_map=device_map)
#### load data ####
path_to_Spider = "./Data/spider"
Output_path = "./Outputs/Spider"
DATASET_SCHEMA = path_to_Spider + "/tables.json"
DATASET_TRAIN = path_to_Spider + "/train_spider.json"
DATASET_DEV = path_to_Spider + "/dev.json"
OUTPUT_FILE_1 = Output_path + "/predicted_sql.txt"
OUTPUT_FILE_2 = Output_path + "/gold_sql.txt"
DATABASE_PATH = path_to_Spider + "/database"
gold_file = path_to_Spider + "/gold_eval.txt"
def load_data(DATASET):
    return pd.read_json(DATASET)

#### Preprocess ####
def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output
def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output
def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output

def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

print('Creating Schema linking...\n')
spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
train_data = load_data(DATASET_TRAIN)
eval_data = load_data(DATASET_DEV)

test_questions = []
def preprocess_function(example, tokenizer):
    questions = []
    prompt = 'You are an expert in SQL. You are given a question and a database schema. You need to write a SQL query to answer the question.\n'
    for question,db_id in zip(example['question'],example['db_id']):
        schema = "db_id:" + db_id +'\n' + find_fields_MYSQL_like(db_id) + '\n' + "foreign key:" + find_foreign_keys_MYSQL_like(
        db_id) + '\n' + "primary key:" + find_primary_keys_MYSQL_like(db_id)
        question_after = prompt + question + '\n' + schema + '\n' + 'SQL:'
        questions.append(question_after)
        test_questions.append(question_after)
    queries = example['query']
    input_tokenized = tokenizer(questions, return_tensors="pt", max_length=512, truncation=True, padding="max_length",add_special_tokens=False)
    output_tokenized = tokenizer(queries, return_tensors="pt", max_length=512, truncation=True, padding="max_length",add_special_tokens=False)

    return {
        "input_ids": input_tokenized["input_ids"],
        "attention_mask": input_tokenized["attention_mask"],
        "labels": output_tokenized["input_ids"],
        "db_id": example["db_id"],
        "gold_query": example["query"]
    }
db_id_train = []
query_train = []
question_train = []
for index, sample in train_data.iterrows():
    # if index == 8:
    #     break
    db_id_train.append(sample['db_id'])
    query_train.append(sample['query'])
    question_train.append(sample['question'])


dataset_train = Dataset.from_dict({
    "db_id": db_id_train,
    "query": query_train,
    "question": question_train,
})
db_id_eval = []
query_eval = []
question_eval = []
for index,sample in eval_data.iterrows():
    # if index == 8:
    #     break
    db_id_eval.append(sample['db_id'])
    query_eval.append(sample['query'])
    question_eval.append(sample['question'])

dataset_eval = Dataset.from_dict({
    "db_id": db_id_eval,
    "query": query_eval,
    "question": question_eval,
})


# Shuffle and select a subset of the data, if needed
dataset_train = dataset_train.shuffle(seed=42)
dataset_eval = dataset_eval

# Preprocess the data
dataset = dataset_train.map(lambda e: preprocess_function(e, tokenizer), batched=True)
eval_dataset = dataset_eval.map(lambda e: preprocess_function(e, tokenizer), batched=True)

print(test_questions[:4])
#### Custom metric ####
def compute_metric(eval_pred):
    predictions = eval_pred.predictions
    preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = eval_pred.label_ids
    inputs = eval_pred.inputs
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True,clean_up_tokenization_spaces=False)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True,clean_up_tokenization_spaces=False)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True,clean_up_tokenization_spaces=False)
    db_id = []
    for question in decoded_inputs:
        result = re.search(r'db_id:(.+?)\n', question)
        db_id.append(result.group(1).strip())
    genetrated_queries = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]  ###########
    gold_queries_and_db_ids = []
    with open(gold_file, 'r') as file:
        for line in file:
            # Split the line by the tab character '\t'
            query, db_id = line.strip().split('\t')

            # Append the query and db_id as a tuple to the list
            gold_queries_and_db_ids.append((query, db_id))
    db_dir = DATABASE_PATH
    etype = 'all'
    table = DATASET_SCHEMA
    # print("now you see")
    score = evaluate(gold_queries_and_db_ids, genetrated_queries, db_dir, etype, table)
    print(f"Execution Accuracy: {score}")
    return {"exec": score}  # 必须返回字典



#### train ####
# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Seq2SeqTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metric,
    args=transformers.Seq2SeqTrainingArguments(
        output_dir="./Checkpoints/GPT_NeoX/Spider",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",  # Change evaluation_strategy to "steps"
        save_strategy="steps",
        eval_steps=80,
        save_steps=200,# Add eval_steps parameter need to lower the log/eval/save steps to see the report results
        learning_rate=2e-5,
        fp16=True,
        logging_steps=500,
        optim="paged_adamw_8bit",
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=513,
        include_inputs_for_metrics=True,
    ),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()





# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 GPT_NeoX.py