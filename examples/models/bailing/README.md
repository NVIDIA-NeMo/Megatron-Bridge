# Bailing (Ling) Examples

This directory contains example scripts for [Ling 2.0](https://github.com/inclusionAI/Ling-V2) MoE language models by inclusionAI.

Ling 2.0 uses a high-sparsity Mixture of Experts (MoE) architecture with sigmoid routing, QK-Norm, and Half RoPE.

| Model | HF ID | Architecture | Params | Active Params |
|---|---|---|---|---|
| Ling-flash-2.0 | `inclusionAI/Ling-flash-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-flash-base-2.0 | `inclusionAI/Ling-flash-base-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-mini-2.0 | `inclusionAI/Ling-mini-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |
| Ling-mini-base-2.0 | `inclusionAI/Ling-mini-base-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable for the base directory. Default: `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for checkpoint conversion examples.

### Import HF → Megatron

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model inclusionAI/Ling-flash-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-flash-2.0 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model inclusionAI/Ling-flash-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-flash-2.0/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Ling-flash-2.0-hf-export \
    --trust-remote-code
```

### Round-trip Validation

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id inclusionAI/Ling-flash-2.0 \
    --megatron-load-path ${WORKSPACE}/models/Ling-flash-2.0/iter_0000000 \
    --tp 1 --ep 8 \
    --trust-remote-code
```

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`inclusionAI/Ling-flash-2.0`)
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint (after conversion export)

The default parallelism for 8 GPUs is `--tp 2 --ep 4`.
TP×PP×EP must equal `--nproc_per_node`.

> **Note**: `--tp 1 --ep 8` works for conversion round-trip but may cause issues during autoregressive inference with single-token batches (empty token dispatch to some EP ranks). Use `--tp 2 --ep 4` for inference.

> **Note**: All Ling 2.0 models use custom HuggingFace code, so `--trust-remote-code` is required for conversion and inference.
