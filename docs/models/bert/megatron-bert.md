# Megatron-Style BERT

[Megatron BERT](https://huggingface.co/nvidia/megatron-bert-uncased-345m) is NVIDIA's original
Pre-LayerNorm BERT-family encoder, published to Hugging Face as `MegatronBertForMaskedLM`
(`transformers` model type `megatron-bert`). Megatron Bridge supports it through `BertBridge`,
the first encoder-only (masked-LM) model bridge in this repository.

## Supported Variants

The public examples target NVIDIA's original 345M-parameter checkpoint:

- `nvidia/megatron-bert-uncased-345m`: https://huggingface.co/nvidia/megatron-bert-uncased-345m
- `nvidia/megatron-bert-cased-345m`: https://huggingface.co/nvidia/megatron-bert-cased-345m

The bridge is registered for the `MegatronBertForMaskedLM` architecture and can auto-detect
compatible checkpoints when their Hugging Face config uses the `megatron-bert` model type.

## Architecture Notes: Pre-LayerNorm Only

`BertBridge` targets **Pre-LayerNorm** BERT, matching Megatron-Core's
`megatron.core.models.bert.bert_model.BertModel` exactly (LayerNorm is applied before
self-attention/MLP, and the residual carries the un-normalized input).

> [!WARNING]
> Vanilla `transformers.BertModel`-based checkpoints (`bert-base-uncased`, `bert-large-uncased`,
> most BioBERT/SciBERT/etc. derivatives) are **Post-LayerNorm** — architecturally different from
> Megatron-Core's `BertModel`. This bridge does **not** support them: forcing a Post-LN checkpoint
> through it would silently load weights into the wrong computational graph and produce incorrect
> outputs with no error raised. Check that a checkpoint's `model_type` is `megatron-bert`, not
> `bert`, before converting.

Other implementation notes:

- `add_binary_head=False` — masked-LM checkpoints have no pooler/NSP head.
- `hidden_act="gelu"` is required because Megatron-Core's BERT masked-LM head hardcodes GELU.
- Decoder and cross-attention configurations are not supported.
- HF ties `cls.predictions.decoder.bias` to `cls.predictions.bias` internally. The bridge
  synthesizes the alias when the export target expects both keys; safe serialization normally
  stores only the canonical `cls.predictions.bias` key.
- Megatron-Core's `BertModel.forward()` requires `tokentype_ids` to be passed explicitly whenever
  `num_tokentypes > 0`, and expects a raw (non-extended) `[batch, seq_len]` attention mask — it
  builds the extended mask internally.

## Examples

For checkpoint conversion, round-trip validation, and fill-mask inference, see the
[Megatron-Style BERT examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/bert/megatron-bert/README.md).

Note that `nvidia/megatron-bert-uncased-345m` only hosts tokenizer assets on the Hub; the weights
must first be downloaded from NGC and converted with Hugging Face's own conversion script (see
the examples README for the exact commands) before they can be imported with Megatron Bridge.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/bert`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/bert)
- Examples: [`examples/models/bert/megatron-bert`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bert/megatron-bert)
