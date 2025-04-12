## 1. Pseudo-code 
1. load test data set
```python
arc_challenge_file = './input/arc-prize-2025/arc-agi_test_challenges.json'
with open(arc_challenge_file, 'r') as f:
    arc_data = json.load(f)
```
2. randomize test order and split to 4 GPUs
```python
task_ids = list(arc_data.keys())
random.shuffle(task_ids)  # Shuffle task order

gpu_tasks = {}
for gpu_id in range(4):
    start_idx = gpu_id * len(task_ids) // 4
    end_idx = (gpu_id + 1) * len(task_ids) // 4
    gpu_tasks[gpu_id] = task_ids[start_idx:end_idx]
```
3. **PREPARE Base Model --**  Load & Shrink Model, tokenizer and formater, detailed in [[#^ddfedd]]
4. **RUN TRAINING --** either with unsloth or trl, with `train_args` =  [[#^128607]]. Train the GPU on the test data using (RCL ?). This fine-tunes the `base_model`
```python
if is_unsloth_model(model):
	from unsloth import UnslothTrainer as Trainer
else:
	from trl import SFTTrainer as Trainer

trainer = Trainer(
	model=model,
	tokenizer=formatter.tokenizer,
	data_collator=formatter.get_data_collator(),
	train_dataset=Dataset.from_list(dataset.as_list(formatter)),
	dataset_text_field="text",
	max_seq_length=max_seq_length,
	dataset_num_proc=None,
	packing=packing,  # Can make training 5x faster for short sequences.
	**add_args,
	args=TrainingArguments(
		**add_train_args,
		**train_args
	),
)

print('*** Start training run...')
if grad_acc_fix and is_unsloth_model(model):
	trainer_stats = unsloth_train(trainer)
else:
	trainer_stats = trainer.train()
```
5. store trained model
```python
if model is not None: model.save_pretrained(store_path)
if tokenizer is not None: tokenizer.save_pretrained(store_path)
```
6. **PREPARE Trained Model --** Load & Shrink Model, using slightly different parameters than preparing base model
7. **RUN INFERENCE --** Run inference with a (Retrainer ?) TODO::
```python

```
8. calculate (augmented scores?)
9. **SELECT & SUBMIT --** 

<br>

## 2. Baseline Model
1. Base Model: [Mistral 8B](https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base)
2. 2024 winner's model: [Daniel's wb55L](https://www.kaggle.com/models/dfranzen/wb55l_nemomini_fulleval/)

<br>

## 3. Files in Baseline
- [[Files in Baseline]]

<br>

## 9. Others
##### a) Pseudo-code to load, shrink the model and tokenizer 

^ddfedd

4. load model
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(report_to=None, model_name=model, dtype=None, load_in_4bit=True, local_files_only=local_files_only, **kwargs)

# ~or_
#########################################################

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, **kwargs)
model = AutoModelForCausalLM.from_pretrained(model, **model_load_args) if mode!='tokenizer_only' else None
if tf_grad_cp and model is not None: model.gradient_checkpointing_enable()
```
5. create formatter instance, which is the `ArcFormatter` class written in `arc_loader.py`
```python
ArcFormatter_premix_3 = lambda **kwargs: ArcFormatter(masking=1, inp_prefix='I', out_prefix='O', arr_sep='\n', arr_end='\n', pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', pre_out=['+/-=']*99, pretext_corpus_split='\n', **kwargs)
formatter = ArcFormatter_premix_3(tokenizer=tokenizer)
```
6. Shrink Embeddings (of the model ?)
```python
def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order):
	...
    return mapping, keep_indices

def shrink_model_embeddings(model, keep_indices, mapping):
    with torch.no_grad():
        row_select = torch.tensor(list(keep_indices))
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select.to(model.get_input_embeddings().weight.data.device))
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select.to(model.get_output_embeddings().weight.data.device))
        model.resize_token_embeddings(len(keep_indices))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def shrink_embeddings(model, tokenizer, corpus=None, keep_token_ids=[], keep_tokens=[], remove_token_ids=[], keep_model_tokens=True, keep_special_tokens=True, keep_normalizer=False, keep_token_order=True):
    if not keep_normalizer: remove_tokenizer_normalizer(tokenizer)
    from collections import OrderedDict  # use as OrderedSet
    keep_indices = OrderedDict()
    keep_indices.update({k: None for k in keep_token_ids})
    keep_indices.update({tokenizer.vocab[t]: None for t in keep_tokens})
    if corpus is not None: keep_indices.update({k: None for k in tokenizer(corpus)['input_ids']})
    if keep_model_tokens:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update({k: None for k in (v if isinstance(v, list) else [v])})
    keep_indices.pop(None, None)
    for idx in remove_token_ids: keep_indices.pop(idx, None)
    mapping, keep_indices = shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order)
    shrink_model_embeddings(model, keep_indices, mapping=mapping)
    return mapping
```
7. dequantize model
```python
model = model.dequantize()
```
8. peft: Parameter Efficient Fine-tuning, using the parameters in `state_dict`, which is detailed below [[#^adc198]]
```python
from peft import set_peft_model_state_dict
res = set_peft_model_state_dict(model, state_dict)
assert not res.unexpected_keys

# ~ or
#####################################################
print(f"*** Load peft model from '{m}'...")
# be careful when using unsloth - using PeftModel to load the model will not apply unsloth optimizations
from peft import PeftModel
model, peft_trained = PeftModel.from_pretrained(model, m, trainable=peft_trainable), True
```
9. format dataset to feed into model, where `arc_test_set` = `ArcDataset.from_file(arc_challenge_file)`
```python
ds = arc_test_set
ds = ds.remove_replies()
ds = ds.augment(tp=True, rot=True, perm=perm_aug, n=(2 if arc_test_set.is_fake else train_epochs), shfl_ex=True, shfl_keys=True)
ds = ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length_train, max_new_tokens=0)
if arc_test_set.is_fake: ds = ds.sorted_by_len(formatter=formatter, name='text', reverse=True)
```
##### a) peft Parameters (General)

^adc198

```python
dict(
	r=64,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
	target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
	lora_alpha=16,
	lora_dropout=0,  # Supports any, but = 0 is optimized
	bias="none",  # Supports any, but = "none" is optimized
	use_gradient_checkpointing=True,  # True or "unsloth" for very long context
	random_state=42,
	use_rslora=True,  # We support rank stabilized LoRA
	loftq_config=None,  # And LoftQ
)
```
##### b) peft Parameters (LoRA)
```python
from peft import LoraConfig, get_peft_model
my_get_peft_model = lambda model, **kwargs: get_peft_model(model, LoraConfig(**kwargs))
```
##### b) Fine-tune Training Parameters

^128607

```python
dict(
	per_device_train_batch_size=8, #x4
	gradient_accumulation_steps=1, #half
	warmup_steps=100, 
	num_train_epochs=1,
	max_steps=5 if arc_test_set.is_fake else -1,
	#max_steps=10 if arc_test_set.is_fake else 20,
	learning_rate=2e-4, #double
	embedding_learning_rate=1e-5, 
	logging_steps=10,
	optim="adamw_8bit",
	weight_decay=0.01,  # 0.01,
	lr_scheduler_type='cosine',  # "linear", "cosine",
	seed=42,
	output_dir=os.path.join(tmp_dir, 'checkpoints'),
	save_strategy="no",
	report_to='none',
)
```
##### b) Modifications Implemented from 2024 ARC to 2025 ARC
1. **4-GPU Support**
   - Extended multi-GPU implementation from 2 to 4 GPUs
   - Modified dataset splitting logic in `prepare_dataset` to evenly distribute work
   - Added training and inference processes for GPU 2 and GPU 3
   - Updated subprocess monitoring to wait for all 8 processes (4 training + 4 inference)
   - Improved resource utilization and inference throughput by 2x

2. **Enhanced Reproducibility**
   - Added global seed control (GLOBAL_SEED = 42) with per-GPU deterministic seeding
   - Applied consistent seed values to all randomized operations for reproducible results
   - Disabled non-deterministic algorithms to ensure consistent outputs across runs
   - Implemented seed-based task distribution for consistent GPU workloads

1. **Comprehensive Visualization**
   - Implemented data visualization for both training and inference phases:
     - Color-coded grid displays for ARC tasks with intuitive color mapping
     - Side-by-side comparisons of inputs, ground truth, and model predictions
   - Added multi-GPU result comparison showing prediction quality across all GPUs
   - Created task-specific visualizations showing training examples, test inputs, and prediction attempts
   - Calculated detailed accuracy metrics with statistical breakdowns:
     - Per-attempt success rates for first and second predictions
     - Overall accuracy percentages for both individual attempts
     - Combined success rate for either prediction attempt
     - Shape and value distribution analysis for predictions vs ground truth
     - Non-zero prediction completion rate and zero-prediction filtering