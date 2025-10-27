
# Skip some modules sensitive to quantization
SKIP_QUANTIZATION_MODULES = [
    "lm_head",
    "multi_modal_projector", # Llama 3.2 Vision, Pixtral, Llava
    "merger",                # Qwen2 VL
    "modality_projection",   # Idefics, SmolVLM
]