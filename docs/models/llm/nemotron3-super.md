# Nemotron 3 Super

```{note}
⚠️ **Work In Progress** - This documentation is under active development and may be incomplete or subject to change.
```
## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, run the following command.

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/output/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints.py import \
--hf-model $HF_MODEL \
--megatron-path $MEGATRON_PATH \
--tp 1 \
--ep 8
```

Notes:
- The default parallelism is TP=1, EP=8 (Expert Parallel)
- Adjust `--nproc-per-node` based on your available GPUs

### Export Megatron → HF
```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/trained/megatron/ckpt
OUTPUT_PATH=/path/to/output/hf/ckpt

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints.py export \
--hf-model $HF_MODEL \
--megatron-path $MEGATRON_PATH \
--hf-path $OUTPUT_PATH \
--tp 1 \
--ep 8
```

### Roundtrip Testing
To verify the correctness of import/export conversions:

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
--hf-model-id $HF_MODEL \
--megatron-load-path $MEGATRON_PATH \
--tp 1 \
--ep 8 \
--trust-remote-code
```

### Compare HF and Megatron Outputs
To compare outputs between HF and Megatron models:

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/compare_hf_and_megatron/compare.py \
--hf_model_path $HF_MODEL \
--megatron_model_path $MEGATRON_PATH \
--prompt "Hello who are " \
--tp 8 \
--ep 8 \
--trust_remote_code
```

## Pretraining Examples

### Pretraining with Real Data
```bash
BLEND_PATH=/path/to/dataset/blend.json
CHECKPOINT_DIR=/path/to/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/pretrain_nemotron_3_super.py \
--per-split-data-args-path=${BLEND_PATH} \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=8 \
train.micro_batch_size=1 \
train.train_iters=1280 \
scheduler.lr_warmup_iters=128 \
scheduler.lr_decay_iters=1152 \
scheduler.lr_wsd_decay_iters=1152 \
model.tensor_model_parallel_size=8 \
model.context_parallel_size=1 \
model.expert_model_parallel_size=8 \
model.sequence_parallel=True
```

Notes:
- **GPU Requirements**: Requires B200 GPUs for NVFP4 support. Minimum of 8 nodes (64 GPUs) required
- The default parallelism settings are TP=8, EP=8, PP=1, CP=1 with sequence parallel enabled
- Expert parallelism (EP) is set to 8 for the MoE architecture
- Adjust batch sizes and iteration counts based on your training requirements
- Make sure to set up WandB credentials if using WandB logging

### Pretraining with Mock Data
For quick testing without a dataset:

```bash
CHECKPOINT_DIR=/path/to/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/pretrain_nemotron_3_super.py \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=128 \
train.train_iters=100 \
scheduler.lr_warmup_iters=10 \
model.hybrid_override_pattern="MEME*ME" \
model.num_layers=7
```

Notes:
- If `BLEND_PATH` is not specified, mock dataset will be used
- The `hybrid_override_pattern` can be used to customize the MoE layer pattern
- Useful for debugging and testing the training pipeline


## Finetuning Recipes

### Full Parameter Fine-Tuning
```bash
MEGATRON_PATH=/path/to/pretrained/megatron/ckpt
CHECKPOINT_DIR=/path/to/finetuned/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_super.py \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=50 \
train.global_batch_size=16 \
train.train_iters=200 \
scheduler.lr_warmup_iters=10 \
model.tensor_model_parallel_size=4 \
model.sequence_parallel=True \
checkpoint.pretrained_checkpoint=$MEGATRON_PATH
```

Notes:
- Default parallelism TP=4, EP=8, PP=1, CP=1 with sequence parallel enabled
- By default, the [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) dataset is used.
- Fine-tuning requires a pretrained Megatron checkpoint, which can be obtained from the "Import HF → Megatron" section above
- Adjust `global_batch_size` and parallelism settings based on your GPU memory and requirements


### LoRA Fine-Tuning
To enable LoRA fine-tuning, pass `--peft lora` to the script:

```bash
MEGATRON_PATH=/path/to/pretrained/megatron/ckpt
CHECKPOINT_DIR=/path/to/lora/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/finetune_nemotron_3_super.py \
--peft lora \
logger.wandb_project=your_project \
logger.wandb_entity=nvidia \
logger.log_interval=5 \
checkpoint.load=${CHECKPOINT_DIR} \
checkpoint.save=${CHECKPOINT_DIR} \
checkpoint.save_interval=100 \
train.global_batch_size=4 \
train.train_iters=200 \
model.tensor_model_parallel_size=4 \
model.context_parallel_size=2 \
model.sequence_parallel=True \
scheduler.lr_warmup_iters=30 \
checkpoint.pretrained_checkpoint=$MEGATRON_PATH
```

Notes:
- By default, the target modules are linear layers `["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]` in the model
- LoRA fine-tuning uses less memory and can work with smaller batch sizes
- Consider using Context Parallel (CP) for longer sequences