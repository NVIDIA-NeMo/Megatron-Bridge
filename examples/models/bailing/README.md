# Bailing (Ling) Examples

This directory contains checkpoint-conversion examples for the Ling 2.0 and Ling 3.0 model families from inclusionAI. All supported variants use custom Hugging Face code and require `--trust-remote-code`.

## Shared Checkpoint Conversion

[conversion.sh](conversion.sh) wraps the repository-wide `scripts/conversion/convert.sh` launcher. It imports a Hugging Face checkpoint into Megatron, exports the persisted DCP back to Hugging Face, and runs the distributed round-trip example.

The wrapper stores checkpoints under `WORKSPACE`, which defaults to `/workspace`:

```bash
export WORKSPACE=/your/shared/workspace
```

The following environment variables select the model and parallel topology:

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_MODEL_ID` | `inclusionAI/Ling-3.0-tiny` | Hub model ID or local Hugging Face directory |
| `TP`, `PP`, `EP`, `ETP` | `1` | Megatron parallelism sizes |
| `NPROC_PER_NODE` | `TP * PP * EP` | Local GPU process count |
| `MEGATRON_PATH` | `${WORKSPACE}/models/<model>` | Imported Megatron checkpoint directory |
| `HF_EXPORT_PATH` | `${WORKSPACE}/models/<model>-hf-export` | Exported Hugging Face directory |
| `ROUNDTRIP_OUTPUT_DIR` | `${WORKSPACE}/models/<model>-roundtrip` | Round-trip Hugging Face output directory |

## Ling 2.0

| Variant | Hugging Face ID | Architecture notes |
|---------|-----------------|--------------------|
| Ling-mini-2.0 | `inclusionAI/Ling-mini-2.0` | 16B MoE, 1.5B active |
| Ling-mini-base-2.0 | `inclusionAI/Ling-mini-base-2.0` | Base checkpoint, 16B MoE |
| Ling-flash-2.0 | `inclusionAI/Ling-flash-2.0` | 100B MoE, 6.1B active |
| Ling-flash-base-2.0 | `inclusionAI/Ling-flash-base-2.0` | Base checkpoint, 100B MoE |

### Conversion

For example, convert Ling-mini-2.0 with two-way tensor parallelism and four-way expert parallelism:

```bash
HF_MODEL_ID=inclusionAI/Ling-mini-2.0 \
TP=2 EP=4 \
bash examples/models/bailing/conversion.sh
```

### Inference

[inference.sh](inference.sh) generates text from the original Hugging Face checkpoint, the imported Megatron checkpoint, and the exported Hugging Face checkpoint. It defaults to `inclusionAI/Ling-mini-2.0` with `TP=2` and `EP=4`:

```bash
bash examples/models/bailing/inference.sh
```

## Ling 3.0

| Variant | Hugging Face ID | Architecture notes |
|---------|-----------------|--------------------|
| Ling-3.0-tiny | `inclusionAI/Ling-3.0-tiny` | Hybrid KDA/MLA, 128 routed experts |
| Ling-3.0-flash | `inclusionAI/Ling-3.0-flash` | Hybrid KDA/MLA with MTP, 512 routed experts |

### Conversion

Ling 3.0 Tiny is the wrapper default and runs on one GPU:

```bash
bash examples/models/bailing/conversion.sh
```

For Ling 3.0 Flash, select the public Flash checkpoint and a distributed topology appropriate for the full model. The following example uses eight-way expert parallelism:

```bash
HF_MODEL_ID=inclusionAI/Ling-3.0-flash \
EP=8 \
bash examples/models/bailing/conversion.sh
```

The wrapper is optional. The equivalent Ling 3.0 Tiny import and export commands use the shared launcher directly:

```bash
export WORKSPACE=${WORKSPACE:-/workspace}

./scripts/conversion/convert.sh import \
    --executor local --device gpu --gpus-per-node 1 \
    --hf-model inclusionAI/Ling-3.0-tiny \
    --megatron-path "${WORKSPACE}/models/Ling-3.0-tiny" \
    --trust-remote-code

./scripts/conversion/convert.sh export \
    --executor local --device gpu --gpus-per-node 1 \
    --hf-model inclusionAI/Ling-3.0-tiny \
    --megatron-path "${WORKSPACE}/models/Ling-3.0-tiny/iter_0000000" \
    --hf-path "${WORKSPACE}/models/Ling-3.0-tiny-hf-export" \
    --trust-remote-code
```

### Tiny Training

`ling_v3_tiny_pretrain_8gpu_h100_bf16_config` is a full-size, eight-GPU, fresh-initialization training smoke using mock data. It does not load the public Hugging Face checkpoint. The public Tiny checkpoint has no MTP tensors; the training smoke enables one MTP depth to exercise that training path.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --recipe ling_v3_tiny_pretrain_8gpu_h100_bf16_config \
    --mode pretrain --dataset mock \
    --seq_length 128
```

Set `LING_V3_TINY_HF_PATH` to a trusted local Hugging Face reference containing `config.json` and the custom modeling files when recipe construction must run offline.

## Related Documentation

- [Ling 2.0](../../../docs/models/bailing/ling-2.md)
- [Ling 3.0](../../../docs/models/bailing/ling-3.md)
