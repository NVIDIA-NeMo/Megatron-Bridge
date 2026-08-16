# Bailing (Ling) Examples

This directory contains example scripts for [Ling 2.0](https://github.com/inclusionAI/Ling-V2) MoE language models by inclusionAI.

It also contains the temporary Ling 3.0 Tiny/Flash HF-to-DCP conversion entrypoint.
Both variants use the architecture-level `BailingMoeV3Bridge`; the bridge dispatches
by the public HF configuration rather than by model size.

Ling 2.0 uses a high-sparsity Mixture of Experts (MoE) architecture with sigmoid routing, QK-Norm, and Half RoPE.

| Model | HF ID | Architecture | Params | Active Params |
|---|---|---|---|---|
| Ling-flash-2.0 | `inclusionAI/Ling-flash-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-flash-base-2.0 | `inclusionAI/Ling-flash-base-2.0` | MoE (256 experts, top-8) | 100B | 6.1B |
| Ling-mini-2.0 | `inclusionAI/Ling-mini-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |
| Ling-mini-base-2.0 | `inclusionAI/Ling-mini-base-2.0` | MoE (256 experts, top-8) | 16B | 1.5B |

## Ling 3.0 Tiny

The pinned public artifact is `inclusionAI/Ling-3.0-tiny` at revision
`a2ee06c0f2de5b171701aee7f73f70a1da75483b`. It has 24 logical blocks, 18 KDA
blocks, 6 MLA blocks, 128 routed experts, top-8 routing, and no MTP layers.

The model-only conversion wrapper expects a local HF directory so that the exact
revision and custom modeling files are explicit:

```bash
python examples/models/bailing/convert_ling3_tiny.py \
    --hf-path /workspace/models/ling_v3_tiny_hf \
    --revision a2ee06c0f2de5b171701aee7f73f70a1da75483b \
    --output /workspace/models/ling_v3_tiny_dcp \
    --low-memory-save
```

The resulting `iter_0000000` directory is a native Megatron `torch_dist` DCP and
can be exported with the shared conversion launcher:

```bash
./scripts/conversion/convert.sh export \
    --device gpu --gpus-per-node 1 \
    --hf-model /workspace/models/ling_v3_tiny_hf \
    --megatron-path /workspace/models/ling_v3_tiny_dcp/iter_0000000 \
    --hf-path /workspace/models/ling_v3_tiny_hf_roundtrip \
    --trust-remote-code
```

For the architecture contract, mapping semantics, topology validation, and known
current MCore draft limitations, see
[`docs/models/bailing/ling-3-tiny-design.md`](../../../docs/models/bailing/ling-3-tiny-design.md).

## Ling 3.0 Flash

The public `inclusionAI/Ling-3.0-flash` variant has 42 logical blocks, 512 routed
experts, `intermediate_size=6144`, direct-Q MLA, and one physical MTP layer. The
same conversion wrapper selects the Flash mappings from `config.json`:

```bash
python examples/models/bailing/convert_ling3_tiny.py \
    --hf-path /workspace/models/ling_v3_flash_hf \
    --output /workspace/models/ling_v3_flash_dcp \
    --low-memory-save
```

The full public Flash checkpoint has been converted to native DCP and strictly
reloaded in the AIStudio runtime. The remaining Flash gates are HF round-trip
weight parity, direct-Q/MTP logit parity, and one GPU forward/backward/save/reload
smoke. Validation uses the official Ling-capable MCore draft revision
`f62b8bf20ee5a03c2fd77a28362e568a0451257e`; the Bridge submodule development pin
matches this commit while `.main.commit` remains on the production MCore pin.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable for the base directory. Default: `/workspace`.

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for checkpoint conversion examples. The
script defaults to `inclusionAI/Ling-mini-2.0`, the 16B checkpoint, and accepts
the other variants through `MODEL_NAME` or a complete `HF_MODEL_ID`:

```bash
MODEL_NAME=Ling-flash-2.0 bash examples/models/bailing/conversion.sh
```

### Import HF → Megatron

```bash
./scripts/conversion/convert.sh import \
    --hf-model inclusionAI/Ling-mini-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-mini-2.0 \
    --trust-remote-code
```

### Export Megatron → HF

```bash
./scripts/conversion/convert.sh export \
    --hf-model inclusionAI/Ling-mini-2.0 \
    --megatron-path ${WORKSPACE}/models/Ling-mini-2.0/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Ling-mini-2.0-hf-export \
    --trust-remote-code
```

### Round-trip Validation

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id inclusionAI/Ling-mini-2.0 \
    --megatron-load-path ${WORKSPACE}/models/Ling-mini-2.0/iter_0000000 \
    --tp 2 --ep 4 \
    --trust-remote-code
```

## Inference

See [inference.sh](inference.sh) for text generation with:
- Hugging Face checkpoint (`inclusionAI/Ling-mini-2.0` by default)
- Imported Megatron checkpoint (after [conversion.sh](conversion.sh) import)
- Exported HF checkpoint (after conversion export)

Both scripts default to 8 GPUs with `--tp 2 --ep 4`. Override `TP`, `PP`,
`EP`, `ETP`, and `NPROC_PER_NODE` together for another valid layout.
TP×PP×EP must equal `--nproc_per_node`.

> **Note**: `--tp 1 --ep 8` works for conversion round-trip but may cause issues during autoregressive inference with single-token batches (empty token dispatch to some EP ranks). Use `--tp 2 --ep 4` for inference.

> **Note**: All Ling 2.0 models use custom HuggingFace code, so `--trust-remote-code` is required for conversion and inference.
