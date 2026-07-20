# Nemotron 3.5 Examples

This directory contains example scripts for Nemotron 3.5 language models:

| Model | Parameters | Active Parameters | Subdirectory |
|-------|-----------|-------------------|--------------|
| Nemotron 3.5 Nano | 30B | A3B | [nano/](nano/) |

Nemotron 3.5 is the next iteration of the Nemotron 3 family. It uses the same
hybrid Mamba+Attention+MoE (`model_type=nemotron_h`) architecture as Nemotron 3
with the addition of Multi-Token Prediction (MTP) heads.

The examples target `nvidia/nemotron-nano-3.5-ea2`. Authenticate with Hugging
Face before running them because the EA2 repository is private.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See [nano/conversion.sh](nano/conversion.sh) for the import (HF → Megatron), export (Megatron → HF), and roundtrip-validation commands.
See [nano/inference.sh](nano/inference.sh) for distributed Megatron text generation directly from the Hugging Face checkpoint or from an imported Megatron checkpoint.

## Training Recipes

Available recipes ([source](../../../../src/megatron/bridge/recipes/nemotronh/nemotron_3_5_nano.py)):

**Nano**:
- `nemotron_3_5_nano_pretrain_config`: Pretraining
- `nemotron_3_5_nano_sft_config`: Supervised fine-tuning
- `nemotron_3_5_nano_peft_config`: PEFT with LoRA support
- `nemotron_3_5_nano_sft_openmathinstruct2_config`: SFT on `nvidia/OpenMathInstruct-2` (packed sequences, SEQ=4096)
- `nemotron_3_5_nano_peft_openmathinstruct2_config`: LoRA on `nvidia/OpenMathInstruct-2` (packed sequences, SEQ=4096)

Before training, ensure the following are configured:
1. **Runtime environment**: Launch from a Megatron Bridge development container or environment with the project dependencies installed.
2. **Environment variables**:
   - `HF_TOKEN`: to download models from HF Hub (if required)
   - `HF_HOME`: (optional) to avoid re-downloading models and datasets
   - `NEMO_HOME`: **required for multi-node SFT/PEFT with packed sequences** — set to a shared-filesystem path (e.g. `${WORKSPACE}/cache/nemo`) so packed data prepared on one node is visible to all others
   - `WANDB_API_KEY`: (optional) to enable WandB logging

Launch recipes with `scripts/training/run_recipe.py` inside an existing distributed environment. The concise recipe names above are compatibility aliases for the canonical H100 recipes exported from `megatron.bridge.recipes.nemotronh.h100`.

### Pre-pack SFT data (one-time, before submitting packed SFT/PEFT jobs)

Packed-sequence recipes require the JSONL + packed parquet cache to exist before training. Run once per dataset on a CPU node:

```bash
export NEMO_HOME=${WORKSPACE}/cache/nemo  # shared FS, use the same value for preparation and training
uv run --no-sync python scripts/training/prepare_gpt_sft_packed_data.py \
    --recipe nemotron_3_5_nano_sft_openmathinstruct2_config
```

The packed `.parquet` lands at `${NEMO_HOME}/datasets/<dataset>/packed/<tokenizer-hash>_pad_seq_to_mult1/training_<seq>.idx.parquet`. The matching PEFT recipe shares the same cache (identical dataset config + tokenizer + `pad_seq_to_mult=1`) — no need to re-pack.

> **CP > 1 note:** the cache is keyed on `pad_seq_to_mult`. For `CP=2` training you need to re-pack with `pad_seq_to_mult = 2 * CP = 4`. The current OpenMath cache only supports `CP=1`.

## LoRA Merge

After LoRA training, export Hugging Face weights with the adapter weights merged into the base model. The script reads the base checkpoint path from `run_config.yaml` inside the LoRA checkpoint directory, so `--pretrained` is usually not required. **Match `--tp` and `--ep` to the parallelism used during training** — otherwise the distributed-checkpoint reshard fails.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/peft/merge_lora.py \
    --lora-checkpoint ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k/iter_0001000 \
    --hf-model-path nvidia/nemotron-nano-3.5-ea2 \
    --output ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k_merged \
    --tp 2 --pp 1 --ep 4
```

The output is a merged Hugging Face checkpoint that can be used for downstream inference or serving.

If the node does not have enough GPU memory, add `--cpu` to load and export entirely on CPU (no GPU required, but slower).

## LoRA Adapter Export

Export LoRA adapter weights to HuggingFace PEFT format (`adapter_config.json` + `adapter_model.safetensors`). This lightweight format can be shared and loaded with the `peft` library without distributing the full base model. No GPU required — runs entirely on CPU.

```bash
uv run python examples/conversion/adapter/export_adapter.py \
  --hf-model-path nvidia/nemotron-nano-3.5-ea2 \
  --lora-checkpoint ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k/iter_0001000 \
  --output ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k_adapter
```

The output directory contains:

- `adapter_config.json` — LoRA configuration (rank, alpha, target modules)
- `adapter_model.safetensors` — adapter weights only

> **Note on PEFT targets:** Nemotron-H has Mamba layers in addition to standard attention/MLP. The PEFT recipe targets attention + MLP + Mamba projections (`in_proj` and `out_proj`) but **not** the expert parameters. The merged checkpoint is therefore a partial fine-tune by design; expert weights pass through unchanged. To extend coverage, add expert modules to `target_modules` in `nemotron_3_5_nano_peft_config`.

## Verify Adapter Export

Verify the exported HuggingFace PEFT adapter behaves identically to the Megatron-merged checkpoint by comparing top-k logits. CPU mode shown; for GPU mode see the script header.

```bash
uv run python examples/conversion/adapter/verify_adapter.py \
  --hf-model-path nvidia/nemotron-nano-3.5-ea2 \
  --hf-adapter-path ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k_adapter \
  --lora-checkpoint ${WORKSPACE}/results/lora-squad-tp2-ep4-cp1-1n-1k/iter_0001000 \
  --cpu
```

The script asserts (a) the PEFT-loaded model's logits differ from the base model (adapter has effect), and (b) top-k predicted tokens match the Megatron checkpoint with merged weights.

## Architecture Notes

- **Hybrid layer pattern** (`hybrid_override_pattern`): mix of Mamba (`M`), Attention (`*`), and Expert/MLP (`E`) layers — see the model's `config.json` for the exact sequence.
- **MTP**: one Multi-Token Prediction head (`num_nextn_predict_layers=1`, pattern `*E`). Loss tracked as `mtp_1 loss` / `mtp_2 loss` in training logs.
- **MoE**: 128 routed experts, top-6 routing, 1 shared expert. Expert-parallel training is essential — recipes default to `EP=8` for pretrain/SFT and `EP=4` for PEFT.
- **CUDA graphs**: enabled by default for pretrain (TE-scoped on `["attn","mamba"]`), disabled for SFT/PEFT (incompatible with `packed_seq_params`).
