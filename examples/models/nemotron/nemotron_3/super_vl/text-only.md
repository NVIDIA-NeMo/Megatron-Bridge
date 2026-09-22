# Nemotron 3.5 Super VL as a text-only Nemotron-H checkpoint

Use `text_only=True` to import the language model and MTP weights directly from
the VL checkpoint. This selects the existing Nemotron-H bridge and plain hybrid
model provider: no vision encoder, projector, video embedder, or audio module is
built. The default remains full VL. For a standalone Nemotron-H checkpoint,
`text_only=True` is a no-op. Other families, including Qwen 3.5 VL, do not
yet implement this option and reject it explicitly.

The VL bridge selects the language config and filters `language_model.*` weights
through the ordinary `PreTrainedCausalLM` wrapper. The existing `NemotronHBridge`
performs conversion; there is no separate text-only model or pretrained subclass.

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B",
    text_only=True,
    trust_remote_code=True,
    revision="<checkpoint-commit>",
)
provider = bridge.to_megatron_provider(load_weights=False)
```

`load_weights=False` constructs the architecture only. Use the default
`load_weights=True` when building a model directly from HF weights, or point a
training recipe's `checkpoint.pretrained_checkpoint` at the converted Megatron
checkpoint. Setting a recipe's `hf_path` alone does not initialize its weights.
The standalone HF text export can also be used as `checkpoint.pretrained_checkpoint`;
it loads through the native text bridge without extracting a language subtree again.

## Conversion

The shared CLI accepts `--text-only` for import and export. For example, in a
GPU-enabled environment with sufficient aggregate memory:

```bash
HF_MODEL=nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B
HF_REVISION='<checkpoint-commit>'

bash scripts/conversion/convert.sh import \
    --executor local --device gpu --gpus-per-node 8 \
    --hf-model "$HF_MODEL" --hf-revision "$HF_REVISION" \
    --text-only --trust-remote-code --tp 1 --pp 1 --ep 8 \
    --megatron-path /workspace/super-text

bash scripts/conversion/convert.sh export \
    --executor local --device gpu --gpus-per-node 8 \
    --hf-model "$HF_MODEL" --hf-revision "$HF_REVISION" \
    --text-only --trust-remote-code --tp 1 --pp 1 --ep 8 \
    --megatron-path /workspace/super-text/iter_0000000 \
    --hf-path /workspace/super-text-hf
```

Export produces a standalone `NemotronHForCausalLM` checkpoint with unprefixed
language weights and a text config. It can subsequently use the normal text
bridge, without `text_only=True`. The original tokenizer and special-token IDs
are retained; keeping media tokens in the vocabulary does not retain the media
encoder. The lazy source wrapper itself is for conversion, not HF generation;
use the Megatron model or the standalone HF export for inference.

Super VL stores one shared attention+MoE MTP block. The text config expresses
two runtime prediction depths, matching the existing Super training convention,
with `mtp_use_repeated_layer=True`. This does not duplicate the serialized
weights. Native Transformers inference may ignore the MTP weights; a successful
HF generation test alone therefore does not validate MTP training.

## Reuse the Super recipes

All Super library recipe constructors (pretrain, SFT, long-context SFT, PEFT,
H100/GB200 variants and legacy aliases) accept the same source options:

```python
from megatron.bridge.recipes.nemotronh import nemotron_3_super_sft_config

cfg = nemotron_3_super_sft_config(
    hf_path="nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B",
    text_only=True,
    revision="<checkpoint-commit>",
    trust_remote_code=True,
)
cfg.checkpoint.pretrained_checkpoint = "/workspace/super-text/iter_0000000"
```
