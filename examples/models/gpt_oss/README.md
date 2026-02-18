# GPT-OSS Examples

This directory contains example scripts for GPT-OSS 20B language models.

For model introduction and architecture details, see the GPT-OSS documentation.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints and results. By default, this is set to `/workspace`. You can override it:

```bash
export WORKSPACE=/your/custom/path
```

Directory structure:
- `${WORKSPACE}/models/` - Converted checkpoints
- `${WORKSPACE}/results/` - Training outputs and experiment results

## Checkpoint Conversion

See the [conversion.sh](conversion.sh) script for checkpoint conversion examples. The Hugging Face GPT-OSS model uses mxfp4; exported Megatron checkpoints are typically bf16. Use `--not-strict` on export to allow key/dtype differences.

### Import HF → Megatron

To import the HF model to your desired Megatron path:

```bash
python examples/conversion/convert_checkpoints.py import \
    --hf-model openai/gpt-oss-20b \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b \
    --trust-remote-code
```

### Export Megatron → HF

Use `--not-strict` when the original HF model uses a different format (e.g. mxfp4) than the Megatron export (e.g. bf16):

```bash
python examples/conversion/convert_checkpoints.py export \
    --hf-model openai/gpt-oss-20b \
    --megatron-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --hf-path ${WORKSPACE}/models/gpt-oss-20b-hf-export \
    --not-strict
```

### Round-trip Validation

Multi-GPU round-trip validation between formats:

```bash
python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id openai/gpt-oss-20b \
    --megatron-load-path ${WORKSPACE}/models/gpt-oss-20b/iter_0000000 \
    --tp 2 --pp 2 \
    --trust-remote-code \
    --not-strict
```