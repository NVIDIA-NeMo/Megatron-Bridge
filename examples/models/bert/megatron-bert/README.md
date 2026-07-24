# Megatron-Style BERT (`MegatronBertForMaskedLM`) Examples

This directory contains example scripts for the Megatron-Bridge `BertBridge`, which converts
between Hugging Face's `MegatronBertForMaskedLM` architecture and Megatron-Core's
`megatron.core.models.bert.bert_model.BertModel`.

The examples target `nvidia/megatron-bert-uncased-345m` (24 layers, 1024 hidden size, 345M
parameters), NVIDIA's original Pre-LayerNorm BERT checkpoint.

## Supported architecture: Pre-LayerNorm only

`BertBridge` targets HF's **`MegatronBertForMaskedLM`** architecture (`transformers` model type
`megatron-bert`), which is Pre-LayerNorm — LayerNorm is applied *before* self-attention and the
MLP, matching Megatron-Core's `BertModel` exactly.

> [!WARNING]
> This bridge does **not** support vanilla `transformers.BertModel`-based checkpoints (e.g.
> `bert-base-uncased`, `bert-large-uncased`, most BioBERT/SciBERT/etc. variants). Those are
> **Post-LayerNorm** (LayerNorm applied after the residual add, as in the original BERT paper),
> which is architecturally different from Megatron-Core's `BertModel`. Converting a Post-LN
> checkpoint through this bridge would silently load weights into the wrong computational graph
> and produce incorrect outputs with no error. If a HF config's `model_type` is `bert` rather than
> `megatron-bert`, it is not supported here.

`MegatronBertForMaskedLM` also has no pooler/NSP head in this bridge (`add_binary_head=False`),
matching how the masked-LM checkpoints are published.

The bridge requires `hidden_act="gelu"` because Megatron-Core's BERT masked-LM head hardcodes
GELU. Decoder and cross-attention configurations are not supported.

## Obtaining a convertible checkpoint

The `nvidia/megatron-bert-uncased-345m` Hugging Face repo only hosts tokenizer files — the
weights must be downloaded from NVIDIA GPU Cloud (NGC) and converted to HF format first, using
HF's own conversion script (this is a one-time step, unrelated to Megatron-Bridge):

    wget --content-disposition \
      https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip \
      -O checkpoint.zip
    git clone https://github.com/huggingface/transformers.git
    python3 transformers/src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py checkpoint.zip

This produces a `config.json` + weights directory that can be used as `HF_MODEL_PATH` below. See
the [`nvidia/megatron-bert-uncased-345m` model card](https://huggingface.co/nvidia/megatron-bert-uncased-345m)
for full details, including the `-cased` variant.

## Workspace Configuration

All scripts use a `WORKSPACE` environment variable to define the base directory for checkpoints
and results. By default, this is set to `/workspace`.

    export WORKSPACE=/your/custom/path
    export HF_MODEL_PATH=/path/to/converted/megatron-bert-uncased-345m-hf

## Checkpoint Conversion

See [conversion.sh](conversion.sh) for Hugging Face to Megatron import, Megatron to Hugging Face
export, and round-trip validation.

    ./examples/models/bert/megatron-bert/conversion.sh

## Inference (fill-mask)

`MegatronBertForMaskedLM` has no `generate()` method, so "inference" here means masked-token
prediction rather than text generation. See [inference.sh](inference.sh), which runs
[fill_mask.py](fill_mask.py) against the original Hugging Face checkpoint and, if present, the
round-tripped export from `conversion.sh`:

    TEXT="Paris is the [MASK] of France." ./examples/models/bert/megatron-bert/inference.sh

`fill_mask.py` uses the Hugging Face `fill-mask` pipeline directly (no Megatron-Core forward
pass) — its purpose is to sanity-check that a converted checkpoint still produces sensible
predictions, not to benchmark Megatron inference throughput.
