# Nemotron 3 Super
[Nemotron 3 Super](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3)is a large language model (LLM) trained by NVIDIA, designed to deliver strong agentic, reasoning, and conversational capabilities. It is employs a hybrid **Latent Mixture-of-Experts (LatentMoE)** architecture, utilizing interleaved Mamba-2 and MoE layers, along with select Attention layers. Distinct from the Nano model, the Super model incorporates **Multi-Token Prediction (MTP)** layers for faster text generation and improved quality, and it is trained using **NVFP4** quantization to maximize compute efficiency. The model has **12B active parameters** and **120B parameters in total**.

NeMo Megatron Bridge supports pretraining, full parameters finetuning, and LoRA finetuning this model. The finetuned model can be converted back to the 🤗 Hugging Face format for downstream evaluation.

```{important}
Please use the custom container `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` when working with this model.

Run all commands from `/opt/Megatron-Bridge` (e.g. `docker run -w /opt/Megatron-Bridge ...`)
```

## Getting the Latest Code

For the best experience, it is recommended to use the latest code from the `super-v3` branch. There are two ways to do this:

### Option 1: Update the Code Inside the Container

Launch the container and update the code in-place:

```bash
# Pull the latest changes from the super-v3 branch
cd /opt/megatron
git pull origin super-v3
```

### Option 2: Mount the Repo from Host

This approach lets you work with the code on your host machine and mount it into the container at runtime.

**Step 1 — Pull the latest `super-v3` branch on the host:**

```bash
git checkout super-v3 && git pull origin super-v3
```

**Step 2 — Mount the repo when launching the container:**

```bash
MEGATRON_BRIDGE_PATH=/path/to/Megatron-Bridge  # set this to your local clone

docker run --rm -it \
  -v $MEGATRON_BRIDGE_PATH:/opt/Megatron-Bridge \
  -w /opt/Megatron-Bridge \
  nvcr.io/nvidia/nemo:26.02.nemotron_3_super \
  bash
```

---

## Conversion with 🤗 Hugging Face

### Import HF → Megatron
To import the HF model to your desired `$MEGATRON_MODEL_PATH`, use the distributed
conversion script because this model uses expert parallelism. The single-process
`examples/conversion/convert_checkpoints.py` script is limited to single-GPU conversion
without model parallelism.

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_PATH=/path/to/output/megatron/ckpt

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints_multi_gpu.py import \
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

torchrun --nproc-per-node=8 examples/conversion/convert_checkpoints_multi_gpu.py export \
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
model.tensor_model_parallel_size=4 \
model.context_parallel_size=1 \
model.expert_model_parallel_size=64 \
model.sequence_parallel=True
```

Notes:
- **GPU Requirements**: Requires B200 GPUs for NVFP4 support. Minimum of 8 nodes (64 GPUs) required
- The default parallelism settings are TP=4, EP=64, PP=1, CP=1 with sequence parallel enabled
- Expert parallelism (EP) is set to 64 for the MoE architecture
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


## Quantization (PTQ and QAT)

```{important}
Quantization support requires the latest code from the `super-v3` branch. See [Getting the Latest Code](#getting-the-latest-code) for instructions.
```

Nemotron 3 Super supports four quantization configurations:

| Config Name | Format | Description |
|---|---|---|
| `mamba_moe_fp8_aggressive` | FP8 | Aggressive FP8 quantization for Mamba-MoE |
| `mamba_moe_fp8_conservative` | FP8 | Conservative FP8 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_aggressive` | NVFP4 | Aggressive NVFP4 quantization for Mamba-MoE |
| `mamba_moe_nvfp4_conservative` | NVFP4 | Conservative NVFP4 quantization for Mamba-MoE |

Pass the desired config name via `--export-quant-cfg` to `quantize.py`.

### Quantize
```bash
export HF_MODEL=/path/to/hf/model
export MEGATRON_SAVE_PATH=/path/to/quantized/megatron/ckpt

torchrun --nproc_per_node=8 examples/quantization/quantize.py \
    --hf-model-id $HF_MODEL \
    --export-quant-cfg mamba_moe_nvfp4_conservative \
    --megatron-save-path $MEGATRON_SAVE_PATH \
    --pp 1 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

### Verify with PTQ Generate
```bash
torchrun --nproc_per_node=8 examples/quantization/ptq_generate.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_SAVE_PATH \
    --pp 1 \
    --tp 8 \
    --ep 8 \
    --trust-remote-code
```

Notes:
- For multi-node setups (e.g. 2 nodes with 8× H100), increase `--pp` accordingly (e.g. `--pp 2`) and use a job scheduler like SLURM to launch across nodes.

### Export Quantized Megatron Checkpoint → HF

After quantization, export the Megatron checkpoint back to Hugging Face format:

```bash
HF_MODEL=/path/to/hf/model
MEGATRON_LOAD_PATH=/path/to/quantized/megatron/ckpt
EXPORT_DIR=/path/to/output/hf/ckpt

torchrun --nproc_per_node=8 examples/quantization/export.py \
    --hf-model-id $HF_MODEL \
    --megatron-load-path $MEGATRON_LOAD_PATH \
    --export-dir $EXPORT_DIR \
    --pp 8 \
    --dtype bfloat16 \
    --trust-remote-code
```

### Quantization-Aware Training (QAT)

After quantization, further improve model quality with QAT by continuing training from a quantized Megatron checkpoint.

```bash
MEGATRON_PATH=/path/to/quantized/megatron/ckpt
CHECKPOINT_DIR=/path/to/qat/checkpoints

torchrun --nproc-per-node=8 examples/models/nemotron_3/qat_nemotron_3_super.py \
--megatron-load-path=${MEGATRON_PATH} \
--seq-length=8192 \
--packed-sequence \
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
model.sequence_parallel=True
```