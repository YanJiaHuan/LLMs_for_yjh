import transformers
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
import torch
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    get_peft_config,
    PeftModelForSeq2SeqLM
)
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
model = MT5ForConditionalGeneration.from_pretrained(model_id, low_cpu_mem_usage = True, load_in_4bit = False, return_dict=True, quantization_config=quantization_config, torch_dtype =torch.float16, device_map='auto')

model = PeftModelForSeq2SeqLM.from_pretrained(model, adapter_path, offload_folder=reload_path, torch_dtype=torch.float16)
model = model.merge_and_unload()
torch.save(model.state_dict())

# Only use this code snippet for weights emerging
#######################
