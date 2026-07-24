# Mistral 7B examples

These scripts run the full checkpoint-conversion and generation smoke workflow
for the original Mistral text models. They default to
`mistralai/Mistral-7B-v0.1` and can target another compatible variant by
setting `MODEL_NAME` or a complete `HF_MODEL_ID`.

## Conversion

Set a checkpoint workspace, then run the import, export, and multi-GPU
round-trip verification:

```bash
WORKSPACE=/workspace bash examples/models/mistral/mistral/conversion.sh
```

The default layout is two GPUs with tensor parallelism 2. To use another
valid layout, set `TP`, `PP`, `EP`, `ETP`, and `NPROC_PER_NODE` together.

## Inference

After conversion, run generation from the original Hugging Face checkpoint,
the imported Megatron checkpoint, and the exported Hugging Face checkpoint:

```bash
WORKSPACE=/workspace bash examples/models/mistral/mistral/inference.sh
```

`PROMPT` and `MAX_NEW_TOKENS` are configurable. For example:

```bash
PROMPT="Once upon a time" MAX_NEW_TOKENS=64 \
  bash examples/models/mistral/mistral/inference.sh
```
