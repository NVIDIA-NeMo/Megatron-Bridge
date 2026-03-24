# Sarvam Examples

This directory contains example scripts for Sarvam language models.

Sarvam models use a Mixture of Experts (MoE) architecture with QKV layernorm and Grouped Query Attention (GQA).

| Model | HF ID | Architecture | Params |
|---|---|---|---|
| Sarvam 30B | `sarvamai/sarvam-30b` | MoE (128 experts, top-6) | 30B total, 3B active |
| Sarvam 105B | `sarvamai/sarvam-105b` | MoE (128 experts, top-8) | 105B total, 10.3B active |

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
    --hf-model sarvamai/sarvam-30b \
    --megatron-path ${WORKSPACE}/models/sarvam-30b \
    --trust-remote-code
```

### Export Megatron → HF

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model sarvamai/sarvam-30b \
    --megatron-path ${WORKSPACE}/models/sarvam-30b/iter_0000000 \
    --hf-path ${WORKSPACE}/models/sarvam-30b-hf-export
```

### Round-trip Validation

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id sarvamai/sarvam-30b \
    --megatron-load-path ${WORKSPACE}/models/sarvam-30b/iter_0000000 \
    --tp 2 --pp 2 --ep 2 \
    --trust-remote-code
```

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`sarvamai/sarvam-30b`)
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint (after conversion export)

The default parallelism for 8 GPUs is `--tp 2 --pp 1 --ep 4 --etp 1`.
TP×PP×EP must equal `--nproc_per_node`.

> **Note**: `sarvamai/sarvam-30b` is a base pretrained model published in float32.
> The Bridge conversion is correct (verified by matching outputs between HF and Megatron checkpoints).
> For coherent text generation, use an instruction-tuned variant if available.
