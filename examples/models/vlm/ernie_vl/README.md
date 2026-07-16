# ERNIE 4.5 VL Examples

This directory contains example scripts for ERNIE 4.5 Vision-Language (VL) MoE models.

## Supported Models

| Model | Parameters | Active Parameters | Type |
|-------|-----------|-------------------|------|
| ERNIE-4.5-VL-28B-A3B-Instruct | 28B | 3B | VL MoE |
| ERNIE-4.5-VL-28B-A3B-Thinking | 28B | 3B | VL MoE |

## Prerequisites

- `--trust-remote-code` is required for the custom HuggingFace model class.
- All scripts use a `WORKSPACE` environment variable for checkpoints. Default: `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

## Checkpoint Conversion

### Import HF → Megatron

```bash
./scripts/conversion/convert.sh import \
    --hf-model baidu/ERNIE-4.5-VL-28B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/ERNIE-4.5-VL-28B-A3B-Instruct \
    --torch-dtype bfloat16 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
./scripts/conversion/convert.sh export \
    --hf-model baidu/ERNIE-4.5-VL-28B-A3B-Instruct \
    --megatron-path ${WORKSPACE}/ERNIE-4.5-VL-28B-A3B-Instruct/iter_0000000 \
    --hf-path ${WORKSPACE}/ERNIE-4.5-VL-28B-A3B-Instruct-hf-export \
    --trust-remote-code
```

See [conversion.sh](conversion.sh) for the full pipeline including multi-GPU round-trip validation.

## Inference

ERNIE 4.5 VL uses a processor API that differs from other VLMs (e.g., Qwen), so a
dedicated inference script is provided instead of the generic `hf_to_megatron_generate_vlm.py`.

### Run Inference from HF Checkpoint

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/models/vlm/ernie_vl/hf_to_megatron_generate_ernie_vl.py \
    --hf_model_path baidu/ERNIE-4.5-VL-28B-A3B-Instruct \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 100 \
    --tp 2 --pp 1 --ep 4 \
    --trust_remote_code
```

See [inference.sh](inference.sh) for a ready-to-use launch script.

## Architecture Notes

ERNIE 4.5 VL uses a **dual-pool MoE** architecture:
- Text and vision experts reside in separate pools within each MoE layer.
- Each pool has its own router and routes tokens independently.
- This design uses `SequentialMLP` (per-expert execution) rather than `GroupedMLP`
  (batched GEMM), since the two pools cannot be merged into a single expert group.
