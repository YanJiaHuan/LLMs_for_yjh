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

# quantization_config = BitsAndBytesConfig(
# load_in_4bit=False,
# load_in_8bit=False,
# bnb_4bit_use_double_quant=False,
# bnb_4bit_quant_type="nf4",
# bnb_4bit_compute_dtype=torch.bfloat16
# )
# model_id = 'huggyllama/llama-65b'
# adapter_path = "./Checkpoints/LLAMA_65B/Spider/checkpoint-200"
# reload_path = "./Checkpoints/LLAMA_65B/Spider/reload"
# model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage = True, load_in_4bit = False, return_dict=True, quantization_config=quantization_config, torch_dtype =torch.float16, device_map='auto')
#
# model = PeftModel.from_pretrained(model, adapter_path, offload_folder=reload_path, torch_dtype=torch.float16)
# model = model.merge_and_unload()
# torch.save(model.state_dict())

# Only use this code snippet for weights emerging
#######################


os.environ["TOKENIZERS_PARALLELISM"] = "false" # To avoid warnings about parallelism in tokenizers


seed = 42
torch.manual_seed(seed)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()


model_hidden_size = 4096  # this is hard-coded to T0pp

ds_config = {
    "train_micro_batch_size_per_gpu": "auto",
    "fp16": {
        "enabled": True,
        "loss_scale": 0
    },

    "flops_profiler": {
        "enabled": True,
        "profile_step": 10,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": "./tmp.log"
      },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "grad_hooks": True,
        "round_robin_gradients": True
    },

    "gradient_clipping": 1.0,

    "wall_clock_breakdown": True,

    "sparse_attention": {
    "mode": "fixed",
    "block": 16,
    "different_layout_per_head": True,
    "num_local_blocks": 4,
    "num_global_blocks": 1,
    "attention": "bidirectional",
    "horizontal_global_attention": True,
    "num_different_global_patterns": 4
  }
}
@dataclass
class GenerationConfig:
    max_new_tokens: Optional[int] = field(default=256)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=10)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    reptition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

# must run before instantiating the model
# ds_config is deepspeed config object or path to the file
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
reload_path = "./Checkpoints/LLAMA_65B/Spider/reload"
model = AutoModelForCausalLM.from_pretrained(reload_path, quantization_config=bnb_config, device_map="auto")
model_name = "huggyllama/llama-65b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ds_engine = deepspeed.initialize(model=model,
#                                  config_params=ds_config,
#                                  model_parameters=None,
#                                  optimizer=None,
#                                  lr_scheduler=None)[0]
# ds_engine.module.eval() # inference


text = "What is the average enrollment of schools?\ndb_id:school_player\nTable player, columns = [*,Player_ID,Player,Team,Age,Position,School_ID]\nTable school, columns = [*,School_ID,School,Location,Enrollment,Founded,Denomination,Boys_or_Girls,Day_or_Boarding,Year_Entered_Competition,School_Colors]\nTable school_details, columns = [*,School_ID,Nickname,Colors,League,Class,Division]\nTable school_performance, columns = [*,School_Id,School_Year,Class_A,Class_AA]\n\nforeign key:[school_details.School_ID = school.School_ID,school_performance.School_Id = school.School_ID,player.School_ID = school.School_ID]\nprimary key:[school.School_ID,school_details.School_ID,school_performance.School_Id,player.Player_ID]\n', 'Find the first names of the teachers that teach first grade.\ndb_id:student_1\nTable list, columns = [*,LastName,FirstName,Grade,Classroom]\nTable teachers, columns = [*,LastName,FirstName,Classroom]\n\nforeign key:]\nprimary key:[list.LastName,teachers.LastName]\n"
inputs = tokenizer.encode(text, return_tensors="pt").to(device=local_rank)

outputs = model.generate(input_ids=inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# deepspeed --num_gpus 8 LLAMA_65B_infer.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python LLAMA_65B_infer.py