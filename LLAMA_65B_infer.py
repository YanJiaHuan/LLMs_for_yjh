###### New Inference Test ######
"""
Example code to load a PyTorch model across GPUs
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import torch
import pdb
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

##### Load the model #####

quantization_config = BitsAndBytesConfig(
load_in_4bit=False,
load_in_8bit=False,
bnb_4bit_use_double_quant=False,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'huggyllama/llama-65b'
adapter_path = "./Checkpoints/LLAMA_65B/Spider/checkpoint-200"
reload_path = "./Checkpoints/LLAMA_65B/Spider/reload"
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage = True, load_in_4bit = False, return_dict=True, quantization_config=quantization_config, torch_dtype =torch.float16, device_map='auto')

model = PeftModel.from_pretrained(model, adapter_path, offload_folder=reload_path, torch_dtype=torch.float16)
model = model.merge_and_unload()
torch.save(model.state_dict())

# Only use this code snippet for weights emerging
#######################


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
@dataclass
class GenerationConfig:
    max_new_tokens: Optional[int] = field(default=256)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    reptition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

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