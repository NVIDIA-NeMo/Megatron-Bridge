# Qwen3-VL MegatronMIMO

This guide describes how to use MegatronMIMO with Qwen3-VL Dense and MoE models.
Qwen3-VL uses `Qwen3VLMIMOModel` to handle DeepStack features, while the
conversion and training procedures are the same as for Qwen3.5-VL.
For an overview of MegatronMIMO, see the [Qwen3.5-VL README](../qwen35_vl/README.md).

## Conversion

Import an HF checkpoint into MIMO format or export a MIMO checkpoint back to HF:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py import \
  --hf-model Qwen/Qwen3-VL-2B-Instruct \
  --megatron-path /workspace/models/qwen3-vl-2b-mimo \
  --component language=tp=1,dp=1,rank_offset=0 \
  --component images=tp=1,dp=1,rank_offset=1 \
  --torch-dtype bfloat16

uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/conversion/convert_megatron_mimo.py export \
  --hf-model Qwen/Qwen3-VL-2B-Instruct \
  --megatron-path /workspace/models/qwen3-vl-2b-mimo \
  --hf-path /workspace/models/qwen3-vl-2b-export \
  --component language=tp=1,dp=1,rank_offset=0 \
  --component images=tp=1,dp=1,rank_offset=1 \
  --torch-dtype bfloat16
```

### MoE configuration

For `Qwen/Qwen3-VL-30B-A3B-Instruct`, use four language ranks and one image
rank on an allocation with at least five GPUs:

```text
language=tp=1,pp=1,cp=1,dp=4,ep=4,etp=1,rank_offset=0
images=tp=1,pp=1,cp=1,dp=1,rank_offset=4
```

EP subdivides the language data-parallel ranks; it does not multiply the
component's rank count.

## Training

Use the [Qwen3.5-VL training script](../qwen35_vl/finetune_qwen35_vl.py)
unchanged for Qwen3-VL Dense and MoE models. Set `--hf-model`,
`--processor-path`, `--pretrained-checkpoint`, and `--component` to match the
Qwen3-VL model and converted checkpoint layout.

## Limits

- The first language pipeline stage's first chunk must contain all DeepStack injection layers(the first three decoder layers for published Qwen3-VL configurations). This is checked against the configured layer partition,　including uneven and virtual splits.
- Use one common CP layout for linear and attention operations. Dual CP layouts are not supported for DeepStack.
- MTP is disabled for MIMO.
