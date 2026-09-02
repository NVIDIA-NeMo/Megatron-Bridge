# Muse Glimmer 30B

[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a dense
vision-language model with a roughly 2B-parameter vision encoder and a 28B-parameter
decoder. Megatron Bridge models the complete checkpoint: the vision tower, adapter,
projection, and text decoder all participate in checkpoint conversion.

## Modeling

Muse Glimmer uses the builder-backed API. `MuseGlimmerBridge` translates the Hugging
Face configuration directly into `MuseGlimmerModelConfig`, and
`MuseGlimmerModelBuilder` constructs `MuseGlimmerModel` as a native Megatron-Core
`HybridModel`; no model provider or `GPTModel` wrapper is involved.

The text decoder uses a native Megatron-Core `HybridStack` whose layer pattern selects
the Muse transformer-layer spec at every depth. It preserves GQA,
tensor/pipeline/context parallelism, 3:1 sliding/full attention, per-layer NoPE,
weightless Q/K RMSNorm, and the full-head sigmoid attention output gate. Muse-specific
residual-branch RMSNorms wrap the attention and MLP outputs without replacing
Megatron-Core's `TransformerLayer` execution path. The embedding norm, final scaled
RMSNorm, output multiplier, and tanh logit soft-cap reproduce the published decoder
math.

The replicated vision path implements the published 50-layer encoder, half-pixel
bilinear learned-position lookup with zero padding, 2-D RoPE, 3:1 window/full attention,
2x2 pixel shuffle, two-layer GELU adapter, text-width projection, and weightless
perception RMSNorm. Its parameters are marked for tensor-parallel gradient averaging.

## Builder-backed configuration

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "meta-models/Muse-Glimmer-30B",
    revision="f84ecc3a0ea984a4c04542a84269e3d065350a6e",  # pragma: allowlist secret
)
model_config = bridge.get_model_config()

# Apply deployment-specific parallelism before construction.
model_config.tensor_model_parallel_size = 8
models = bridge.get_model(model_config)
```

The published configuration needs Transformers 5.15 for native Hugging Face model
execution. Megatron Bridge includes only a configuration compatibility class so that
the lazy safetensors import and builder path continue to work with the repository's
currently pinned Transformers release; it does not vendor or execute the Hugging Face
model implementation.

## Verification status

See the [model verification card](../../model_verification_cards/muse-glimmer-30b/card.yaml)
for the pinned source revision and current evidence. The card keeps import, export,
forward correlation, inference, and training claims separate.
