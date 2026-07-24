# MiMo 7B examples

These scripts run the full checkpoint-conversion and generation smoke workflow
for XiaomiMiMo text models. They default to `XiaomiMiMo/MiMo-7B-Base`; set
`MODEL_NAME` or a complete `HF_MODEL_ID` for another compatible variant.

MiMo uses custom Hugging Face configuration code, so the scripts pass
`--trust-remote-code` for conversion and inference.

## Conversion

Run the single-process import and export followed by multi-GPU verification:

```bash
WORKSPACE=/workspace bash examples/models/mimo/conversion.sh
```

The default layout is two GPUs with tensor parallelism 2. To use another
valid layout, set `TP`, `PP`, `EP`, `ETP`, and `NPROC_PER_NODE` together.

## Inference

After conversion, run generation from the original Hugging Face checkpoint,
the imported Megatron checkpoint, and the exported Hugging Face checkpoint:

```bash
WORKSPACE=/workspace bash examples/models/mimo/inference.sh
```

`PROMPT` and `MAX_NEW_TOKENS` are configurable. During generation, the MiMo
multi-token-prediction layer is disabled by the shared text-generation helper.
