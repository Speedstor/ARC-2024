
def for_training(model, use_gradient_checkpointing = True):
    if not hasattr(model, "parameters"):
        raise TypeError("mldev: I think you're passing a tokenizer, not the model to for_training!")

    # Delete all fast inference loras
    for param in model.parameters():
        if hasattr(param, "_fast_lora"):
            del param._fast_lora
    pass

    def _for_training(m):
        if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = use_gradient_checkpointing
        if hasattr(m, "training"): m.training = True
        # Pad tokenizer to the left
        if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "right"
        # Set a flag for generation!
        if hasattr(m, "_flag_for_generation"): del m._flag_for_generation
    pass
    m = model
    while hasattr(m, "model"):
        _for_training(m)
        m = m.model
    _for_training(m)
    model.train() # to turn on training on modules deeper in

    # Since transformers 4.53, must turn on explicitly
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = use_gradient_checkpointing
    pass

    # Also re-enable training for embeddings for NEFTune
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if hasattr(embeddings, "training"): embeddings.training = True
    pass
    if hasattr(model, "get_output_embeddings"):
        embeddings = model.get_output_embeddings()
        if hasattr(embeddings, "training"): embeddings.training = True
    pass
    return model
pass




def for_inference(model):
    if not hasattr(model, "parameters"):
        raise TypeError("mldev: I think you're passing a tokenizer, not the model to for_inference!")

    def _for_inference(m):
        if hasattr(m, "gradient_checkpointing"): m.gradient_checkpointing = False
        if hasattr(m, "training"): m.training = False
        # Pad tokenizer to the left
        if hasattr(m, "_saved_temp_tokenizer"): m._saved_temp_tokenizer.padding_side = "left"
        # Set a flag for generation!
        m._flag_for_generation = True
    pass
    m = model
    while hasattr(m, "model"):
        _for_inference(m)
        m = m.model
    _for_inference(m)
    model.eval() # to turn off training on modules deeper in

    # Since transformers 4.53, must turn off explicitly
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    pass

    # Also disable training for embeddings for NEFTune
    if hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        if hasattr(embeddings, "training"): embeddings.training = False
    pass
    if hasattr(model, "get_output_embeddings"):
        embeddings = model.get_output_embeddings()
        if hasattr(embeddings, "training"): embeddings.training = False
    pass
    return model