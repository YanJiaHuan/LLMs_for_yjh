# ######### Inference Test #########
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from peft import PeftModel
# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed
#
# # Load the trained model and tokenizer
# model_name = "huggyllama/llama-65b"
# adapter_name = "./Checkpoints/LLAMA_65B/Spider/checkpoint-200"
# reload_path = "./Checkpoints/LLAMA_65B/Spider/reload"
# model = AutoModelForCausalLM.from_pretrained(reload_path)
# model.tie_weights()
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({
#                 "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
#                 "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
#                 "unk_token": tokenizer.convert_ids_to_tokens(
#                     model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
#                 ),
#         })
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
# print(tokenizer.special_tokens_map)
# print(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))
#
# model.resize_token_embeddings(len(tokenizer))
#
# def apply_lora(base_model_path, target_model_path, lora_path):
#     print(f"Loading the base model from {base_model_path}")
#     base = AutoModelForCausalLM.from_pretrained(
#         base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
#     )
#     base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
#
#     print(f"Loading the LoRA adapter from {lora_path}")
#
#     lora_model = PeftModel.from_pretrained(
#         base,
#         lora_path,
#         # torch_dtype=torch.float16
#     )
#
#     print("Applying the LoRA")
#     model = lora_model.merge_and_unload()
#
#     print(f"Saving the target model to {target_model_path}")
#     model.save_pretrained(target_model_path)
#     base_tokenizer.save_pretrained(target_model_path)
#
#
# # print('reloading...')
# # apply_lora(model_name, reload_path, adapter_name)
#
#
# # model.resize_token_embeddings(len(tokenizer))
# # Load a single example from the Spider dataset
# # test_example = eval_data.loc[0]  # replace 0 with the index of the desired test example
# # test_question = test_example['question']
# # test_db_id = test_example['db_id']
# #
# # # Construct the test text with question and schema
# # test_text = test_question + '\n' + "db_id:" + test_db_id +'\n' + find_fields_MYSQL_like(test_db_id) + '\n' + "foreign key:" + find_foreign_keys_MYSQL_like(test_db_id) + '\n' + "primary key:" + find_primary_keys_MYSQL_like(test_db_id)
#
#
# ds_config = {
#     "fp16": {
#         "enabled": False,
#     },
#     "bf16": {
#         "enabled": True,
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "overlap_comm": True,
#         "contiguous_gradients": True,
#         "reduce_bucket_size": model_hidden_size * model_hidden_size,
#         "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
#         "stage3_param_persistence_threshold": 10 * model_hidden_size
#     },
#     "steps_per_print": 2000,
#     "train_batch_size": world_size,
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": False
# }
#
#
# test_text = "What is the name of the person who is the author of the book with the title 'The Shining'?"
# # Define a method for making predictions
#
# device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
# model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
#
# def make_prediction(input_text):
#     # Encode the text
#     encoded_input = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True, padding="max_length",add_special_tokens=False)
#
#     print("Max input value:", torch.max(encoded_input["input_ids"]))
#     print(encoded_input["input_ids"])
#     print("Embedding size:", model.get_input_embeddings().weight.size(0))
#
#     # Generate the prediction
#     output = model.generate(encoded_input["input_ids"], max_new_tokens=514, num_beams=4)
#
#     # Decode the output
#     decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     return decoded_output
#
# # Make a prediction using the model
# print(test_text)
# print(make_prediction(test_text))

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 LLAMA_65B_infer.py








###### New Inference Test ######
"""
Example code to load a PyTorch model across GPUs
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import torch
import pdb
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers


seed = 42
torch.manual_seed(seed)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()


model_hidden_size = 4096  # this is hard-coded to T0pp

ds_config = {
    "fp16": {
        "enabled": False,
    },
    "bf16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": world_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}



# must run before instantiating the model
# ds_config is deepspeed config object or path to the file
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

reload_path = "./Checkpoints/LLAMA_65B/Spider/reload"
model = AutoModelForCausalLM.from_pretrained(reload_path)
model_name = "huggyllama/llama-65b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ds_engine = deepspeed.initialize(model=model,
                                 config_params=ds_config,
                                 model_parameters=None,
                                 optimizer=None,
                                 lr_scheduler=None)[0]
ds_engine.module.eval() # inference


text = "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy"
inputs = tokenizer.encode(text, return_tensors="pt").to(device=local_rank)

with torch.no_grad():
    outputs = ds_engine.module.generate(inputs, synced_gpus=True,min_new_tokens=10, max_new_tokens=512, num_beams=4, early_stopping=True)

print('text:\n')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# deepspeed --num_gpus 8 LLAMA_65B_infer.py