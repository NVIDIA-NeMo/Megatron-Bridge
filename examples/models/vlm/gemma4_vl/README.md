# Gemma 4 VL Examples

This directory contains example scripts for the Gemma 4 26B-A4B vision-language model.

Gemma 4 26B-A4B is a Mixture-of-Experts (MoE) VLM with a SigLIP vision encoder and a 26B sparse language model (4B active parameters). It requires dedicated handling compared to dense VLMs due to its MoE architecture and expert parallelism requirements.

## Requirements

Gemma 4 requires `transformers>=5.5.0`. To upgrade:

```bash
uv pip install -q --upgrade 'transformers>=5.5.0' mistral_common
```

All scripts in this directory run `uv run --no-sync` to prevent `uv` from reverting the upgrade.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted Megatron checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

### Import HF → Megatron

```bash
uv pip install -q --upgrade 'transformers>=5.5.0'
uv run --no-sync python examples/conversion/convert_checkpoints.py import \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B
```

### Export Megatron → HF

```bash
uv run --no-sync python examples/conversion/convert_checkpoints.py export \
    --hf-model google/gemma-4-26B-A4B \
    --megatron-path ${WORKSPACE}/models/gemma-4-26B-A4B/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gemma-4-26B-A4B-hf-export
```

See the [conversion.sh](conversion.sh) script for more examples including multi-GPU round-trip validation.

## Inference

Gemma 4 uses VLM inference script (`hf_to_megatron_generate_vlm.py`) as other models. The script auto-detects the bridge type and switches to Gemma4-specific input preprocessing, attention mask handling, and stop tokens.

### Text-only

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --prompt "The capital of France is" \
    --max_new_tokens 20 \
    --tp 4 --pp 2
```

### Vision + Text (HF weights)

Use the instruction-tuned model (`-it`) for image+text queries — the base model has no chat template and requires manual image token injection.

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg" \
    --prompt "What is shown in this image?" \
    --max_new_tokens 50 \
    --tp 4 --pp 2
```

### Vision + Text (imported Megatron checkpoint)

```bash
uv run --no-sync python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path google/gemma-4-26B-A4B \
    --megatron_model_path ${WORKSPACE}/models/gemma-4-26B-A4B/iter_0000000 \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG" \
    --prompt "What animal is on the candy?" \
    --max_new_tokens 50 \
    --tp 4 --pp 2
```

Note: when loading a Megatron checkpoint for VLM inference, use the base model (`gemma-4-26B-A4B`, not `-it`) as the `--hf_model_path` to match the checkpoint's tokenizer.

See the [inference.sh](inference.sh) script for all three steps.

**Expected output (cat image):**
```
======== GENERATED TEXT OUTPUT ========
Image: https://.../pipeline-cat-chonk.jpeg
Prompt: What is shown in this image?
New tokens: The image shows a large, fluffy orange and white cat sitting inside
what appears to be a wire cage or kennel. The cat looks quite large and appears
relaxed, with its paws tucked underneath its body.
=======================================
```

## Finetune Recipes

Available recipes:
- `gemma4_vl_26b_sft_config` — Full supervised fine-tuning
- `gemma4_vl_26b_peft_config` — LoRA parameter-efficient fine-tuning

Before training, ensure the following environment variables are set:
1. `WORKSPACE`: base directory for checkpoints and results (default: `/workspace`)
2. `HF_TOKEN`: to download models from HF Hub (if required)
3. `HF_HOME`: (optional) to avoid re-downloading models and datasets
4. `WANDB_API_KEY`: (optional) to enable WandB logging; set `WANDB_MODE=disabled` to turn off

### Supervised Fine-Tuning (SFT)

For single-node interactive runs, see [sft.sh](sft.sh).

For multi-node Slurm jobs, see [slurm_sft.sh](slurm_sft.sh). Default configuration: TP=2, PP=1, EP=8 on 2 nodes (16 GPUs).

```bash
# Override defaults via environment variables
PRETRAINED_CHECKPOINT=${WORKSPACE}/models/gemma-4-26B-A4B \
TP=2 PP=1 EP=8 \
sbatch --nodes=2 slurm_sft.sh
```

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA

For single-node interactive runs, see [peft.sh](peft.sh).

For multi-node Slurm jobs, see [slurm_peft.sh](slurm_peft.sh). Default configuration: TP=2, PP=1, EP=4 on 1 node (8 GPUs).

```bash
sbatch slurm_peft.sh
```

### Recommended Configurations

| Mode | TP | PP | EP | Nodes | Global Batch Size | Learning Rate | Notes |
|------|----|----|----|----|-------------------|---------------|-------|
| Full SFT | 2 | 1 | 8 | 2 | 32 | 5e-5 | Max EP=DP=8; vision unfrozen; no activation recompute |
| Full SFT | 4 | 2 | 1 | 1 | 32 | 5e-5 | `recompute_granularity="selective"`; freeze vision |
| LoRA | 2 | 1 | 4 | 1 | 32 | 2e-4 | EP=4 required (see note above) |

> **Note:** Do not use `recompute_granularity="full"`. Megatron's `CheckpointFunction` does not support non-tensor (tuple) arguments, causing a `TypeError` at runtime. Use `"selective"` instead.


### Expected Training Dynamics

We provide a [Weights & Biases report](https://api.wandb.ai/links/nvidia-nemo-fw-public/r7dgbroo) for the expected loss curves and grad norms.

## Evaluation

After training, use [eval_sft_cord_v2.py](eval_sft_cord_v2.py) to verify the fine-tuned checkpoint on CORD-v2. It feeds the full conversation (image + prompt + ground-truth response) through the model in a single forward pass and reports per-example cross-entropy loss, token accuracy, and GT vs. predicted text.

Example invocation (single node, 8 GPUs). Replace `<JOB_ID>` with your training job ID:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/models/vlm/gemma4_vl/eval_sft_cord_v2.py \
    --hf_model_path google/gemma-4-26B-A4B-it \
    --megatron_model_path ${WORKSPACE}/results/gemma4_vl_sft_tp2_pp1_ep8_<JOB_ID> \
    --tp 2 --pp 1 --ep 4 \
    --num_examples 20
```

For batch evaluation on Slurm, see [slurm_eval_sft.sh](slurm_eval_sft.sh).

After 100 SFT iterations on CORD-v2, expected teacher-forced token accuracy is ~98%.

> **These scripts run one sample at a time and are intended only as sanity checks of the trained checkpoint.** For production inference, re-export the checkpoint to HF format using the export step in [conversion.sh](conversion.sh) and run with vLLM.

## LoRA Merge

After LoRA training, merge adapter weights back into the base Megatron checkpoint. The script reads the base checkpoint path from `run_config.yaml` inside the LoRA checkpoint directory, so `--pretrained` is usually not required. Match `--tp` and `--ep` to the parallelism used during training.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/peft/merge_lora.py \
    --lora-checkpoint ${WORKSPACE}/results/gemma4_vl_lora_tp2_pp1_ep4_<JOB_ID>/iter_<NNNNNNNN> \
    --hf-model-path google/gemma-4-26B-A4B-it \
    --output ${WORKSPACE}/results/gemma4_vl_lora_tp2_pp1_ep4_<JOB_ID>_merged \
    --tp 2 --pp 1 --ep 4
```

The merged checkpoint is a standard Megatron checkpoint that can be used for inference or re-exported to HF format using the export step in [conversion.sh](conversion.sh).

If the node does not have enough GPU memory, add `--cpu` to load and merge entirely on CPU (no GPU required, but slower).

## LoRA Adapter Export

Export LoRA adapter weights to HuggingFace PEFT format (`adapter_config.json` + `adapter_model.safetensors`). This lightweight format can be shared and loaded with the `peft` library without distributing the full base model. No GPU required — runs entirely on CPU.

```bash
uv run python examples/conversion/adapter/export_adapter.py \
  --hf-model-path google/gemma-4-26B-A4B-it \
  --lora-checkpoint ${WORKSPACE}/results/gemma4_vl_lora_tp2_pp1_ep4_<JOB_ID>/iter_<NNNNNNNN> \
  --output ${WORKSPACE}/results/gemma4_vl_lora_tp2_pp1_ep4_<JOB_ID>_adapter
```

The output directory contains:

- `adapter_config.json` — LoRA configuration (rank, alpha, target modules)
- `adapter_model.safetensors` — adapter weights only (~3.7 GB for rank-32 on all linear layers)

> **Note:** Gemma 4 global-attention layers use K=V tying — there is no `v_proj` in the HF global-attention modules. The bridge automatically skips exporting the V adapter for those layers, so the exported adapter is compatible with `peft.PeftModel.from_pretrained`.

## Architecture Notes

### Global Attention K=V Tying

Gemma 4 26B-A4B uses an interleaved pattern of sliding-window (local) and full (global) attention
layers. The global attention layers have a unique property: in the HF checkpoint there is no `v_proj`
weight — key and value share the same projection matrix. At checkpoint import the bridge copies the
K rows into the V rows of Megatron's fused QKV weight.

During fine-tuning, K=V tying must be enforced in every forward pass to keep the model
architecturally consistent. This is handled in `Gemma4SelfAttention.get_query_key_value_tensors`
by setting `value = key` after the parent class returns the split Q/K/V tensors.

