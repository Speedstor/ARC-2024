import torch
from platform import system as platform_system
from triton import __version__ as triton_version
import json

from .quantization_config import BitsAndBytesConfig
from . import Peft
from .device_type import (
    DEVICE_TYPE,
    DEVICE_COUNT,
)

from transformers.models.mistral.modeling_mistral import MistralForCausalLM

FILE_NAME_FOR_SAFETENSOR = "model.safetensors"
FILE_NAME_FOR_CONFIG = "config.json"

class LlamaModel:
    def get_config_dict(self, model_path):
        config_dict = json.loads


    def from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        max_seq_length=None,
        load_in_4bit=False,
        device_map         = "sequential",
    ):
        
        if DEVICE_TYPE == "cuda":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_version = torch.version.cuda
            gpu_stats_snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."
        elif DEVICE_TYPE == "hip":
            gpu_stats = torch.cuda.get_device_properties(0)
            gpu_version = torch.version.hip
            gpu_stats_snippet = f"ROCm Toolkit: {gpu_version}."
        elif DEVICE_TYPE == "xpu":
            gpu_stats = torch.xpu.get_device_properties(0)
            gpu_version = torch.version.xpu
            gpu_stats_snippet = f"Intel Toolkit: {gpu_version}."
        else:
            raise ValueError(f"Unsloth: Unsupported device type: {DEVICE_TYPE}")

        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)


        statistics = \
        f"{gpu_stats.name}. Num GPUs = {DEVICE_COUNT}. Max memory: {max_memory} GB. Platform: {platform_system()}.\n"\
        f"Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"
        print(statistics)


        assert(dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.float32)

        with open(f"{model_path}/{FILE_NAME_FOR_CONFIG}", "r") as f:
            model_config = json.loads(f.read())
        model_config.model_path = model_path
        
        IS_FALCON_H1 = model_config.model_type.startswith("falcon_h1")

        if max_seq_length is None:
            max_seq_length = model_config.max_position_embeddings 

        rope_scaling = None
        if "rope_scalling" in model_config:
            rope_scaling = {"type": "linear", "factor": max_seq_length / model_config.max_position_embeddings,}

        bnb_config = None
        if load_in_4bit:
            llm_int8_skip_modules =  Peft.SKIP_QUANTIZATION_MODULES.copy()
            if IS_FALCON_H1:
                # we cannot quantize out_proj layer due to mamba kernels: https://github.com/tiiuae/Falcon-H1/issues/13#issuecomment-2918671274
                llm_int8_skip_modules.append("out_proj")
                
            bnb_config = BitsAndBytesConfig(
                load_in_4bit              = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type       = "nf4",
                bnb_4bit_compute_dtype    = dtype,
                llm_int8_skip_modules     = llm_int8_skip_modules,
            )

        model = MistralForCausalLM.from_pretrained(
            model_path,
            device_map              = device_map,
            dtype                   = dtype,
            quantization_config     = bnb_config,
            max_position_embeddings = model_config.max_position_embeddings,
            attn_implementation     = "eager",
        )
        
        return model