# Inkling2 support

Supported checkpoint: **Inkling 2 Small**,
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(`Inkling2ForConditionalGeneration`, model type `inkling2`), an image-and-text
mixture-of-experts model. The release ships its configuration, tokenizer, chat template, and
processor configuration but no modeling code. Released Transformers versions do not implement
it (their `inkling` model is the first Inkling generation, a different architecture);
huggingface/transformers#50682 adds Inkling 2. Bridge registers the `inkling2_model`
configuration type, and `inkling2` when Transformers has no class for it, so that
`AutoConfig` reads the checkpoint; their defaults are the release's values.

The model API is `Inkling2Bridge`, `Inkling2Model`, `Inkling2ModelBuilder`, and
`Inkling2ModelConfig`, under `megatron.bridge.models.inkling`. The configuration is
builder-backed: `AutoBridge.get_model_config()` returns an `Inkling2ModelConfig` whose
nested `Inkling2TransformerConfig` holds the Megatron-Core settings and the
`inkling2_*` architecture fields.

## Architecture

`Inkling2Model` extends Megatron-Core `HybridModel` without positional embeddings, as
Muse Spark Open does. Each decoder block is a Hybrid attention layer followed by a MoE
layer, so the Hybrid pattern repeats `*E`: Hybrid layers `2 * i` and `2 * i + 1` hold
block `i`, and the pattern's `|` separators set the pipeline stages. The embedding stage
normalizes the token embeddings with an RMSNorm and replaces image placeholders with
encoder features; the output layer applies a tanh soft-cap and masks the padded
vocabulary rows. Every block's MoE layer is the existing Megatron-Core MoE layer: it runs
a sigmoid router whose bias only selects experts,
routing weights renormalized over the selected experts and scaled, grouped expert GEMMs,
and expert-parallel token dispatch (HybridEP in the recipes). The experts use the SiTU activation, soft-capped gate and up
projections with a sigmoid gate, through Transformer Engine's activation function hook.

The `modeling_inkling2/` package adds the computations that Megatron-Core lacks:

- `layers.py`: Block Attention Residuals. Instead of a residual stream, every
  sublayer reads a softmax mixture of the completed block outputs and the current
  partial block, weighted by a learned FP32 pseudo-query against the RMS-normalized
  sources. Full-attention blocks mix separate query, key, and value inputs, each with
  its own norm. A short depthwise convolution follows the attention and the MoE
  outputs, and the final output is one more mixture. HybridStack passes one tensor
  between layers, so each layer packs the bank of completed blocks next to the partial
  block along the hidden dimension; the width grows by one hidden size per completed
  block, and pipeline communication exchanges the shapes.
- `attention.py`: separate query, key, value, relative-query, gate, and sink
  projections; L2 clipping of queries, keys, and relative queries; length scaling of
  full-attention queries; a weightless value RMSNorm; short convolutions on keys and
  values, static in sliding-window layers and input-dependent in full-attention
  layers; learned relative logits over the last 1024 positions; an attention sink;
  and a sigmoid output gate. Attention runs in compiled PyTorch FlexAttention with
  the relative logits in its score modification and a causal, per-document,
  sliding-window block mask. The sink stays out of the kernel: it scales each output
  row by `sigmoid(lse - sink)`, from the log-normalizer FlexAttention returns, because
  an extra key block that every query attends to would serialize the backward pass
  over all query blocks.
- `vision.py`: the image encoder, a vision transformer with a class token, a
  learned position grid, 2D rotary embeddings, and LayerScale, followed by a 2x2
  patch merge and a two-layer projection. Released Transformers versions have no
  implementation, so this is a parameter-for-parameter port of the reference
  encoder. It takes patches already scaled to [-1, 1] (mean and standard deviation 0.5),
  as the release's `processor_config.json` specifies.
- `ops.py`: the AttnRes mixture, FP32-accumulated residual additions, the static and
  dynamic short convolutions, L2 clipping, the length-scaling factor, and the SiTU
  activation. With `inkling2_compile_ops` the layers run these operators through
  `torch.compile`, one fused graph per call signature, instead of a chain of FP32 passes;
  the context-parallel short convolutions, which exchange rows between ranks, stay eager.

## Checkpoint conversion

The released checkpoint stores the routed experts in NVFP4 (packed E2M1 values, an
FP8 E4M3 scale per 16 values, and a per-expert amax with a shared denominator) or in
MXFP8 (E4M3 values with a power-of-two scale per 32 values), and their gate and up
rows interleaved. Import dequantizes each expert to BF16 when it is loaded, so only one
layer's experts are materialized at a time, and splits the interleaved rows. Export
re-quantizes each expert against the source scales and writes the scale tensors, which
reproduces the stored bytes of an unchanged model; with an explicit export weight dtype
the experts are written unquantized instead. The AttnRes queries and router biases stay
FP32 through training and conversion.

The speculative-decoding draft network in `mtp.safetensors` is not part of the
Megatron model. Export copies it unchanged, with the quantization config, processor
config, chat template, and tokenizer files. Export writes `config.json` from the
checkpoint's saved model config, so a model whose architecture differs from the
`--hf-model` reference keeps its own settings; the reference supplies only what the
Megatron model does not hold, such as `mtp_config` and `eos_token_id`. Transformers (from
v5.23.0) applies the release's MXFP8 quantization to a `config.json` without
`transformers_version`, as the release's is, so a default export, whose experts keep the
source's formats, leaves the version out, and a plain export keeps it.

## Parallelism and limits

Supported: tensor parallelism with sequence parallelism, pipeline parallelism,
expert parallelism, context parallelism, and data parallelism with the distributed
optimizer. A pipeline stage sends the completed AttnRes blocks together with its
partial block, so stage boundaries may fall inside an AttnRes block; pipeline
communication exchanges tensor shapes (`variable_seq_lengths`).

Transformer Engine's context parallelism does not apply to the FlexAttention core, so
`context_parallel.py` provides Inkling2's own. It keeps the training step's split:
every packed document is cut into 2 x CP chunks, two per rank. Each rank's queries
attend over the whole sequence: keys and values are all-gathered over the CP group and
put back in global order, and the score modification and block mask use each query's
global position and document. The short convolutions on keys, values, and outputs
gather the three rows before each local chunk from the neighboring ranks, and length
scaling uses document positions from the same layout. Image inputs and hierarchical
context parallelism are rejected with CP above 1.

Not yet supported, and rejected by `Inkling2ModelConfig.finalize()`: hierarchical
context parallelism, virtual pipeline parallelism, CUDA graphs, expert-parallel
communication overlap, FP8 and FP4 training, Megatron MTP layers, dropout, and linear
biases. Megatron's KV-cache inference is not implemented, so generation
(`scripts/inference/vlm_generation.py`) runs the whole sequence at every step.

Documents stay apart only in packed sequences: the attention and the short convolutions
take document boundaries from `packed_seq_params` and never read `attention_mask`. The
model therefore rejects an unpacked batch whose mask hides earlier tokens, as the GPT
dataset's `reset_attention_mask` and left padding produce; pack the documents instead.
Causal masks and padding at the end lose nothing.

FlexAttention is compiled for fixed shapes. The packed SFT recipes pad every pack to
the full sequence length so that the attention kernel compiles once.

The checkpoint's tokenizer configuration assigns no BOS or EOS token, which pretraining
datasets and SFT padding need. The recipes set the reference tokenizer's roles,
`<|begin_of_text|>` and `<|endoftext|>`, through `tokenizer.hf_special_tokens`.

## Training recipes

The GB200 recipes in `megatron.bridge.recipes.inkling` warm-start from an imported
checkpoint: bounded pretraining, full SFT on packed 8K Tulu 3 and on packed 32K Tulu 3
with context parallelism, LoRA on the
attention projections (`linear_q`, `linear_k`, `linear_v`, and `linear_proj`), and full
SFT and LoRA on image-text CORD-v2. They
dispatch the experts with HybridEP inside one NVL72 domain per expert-parallel group and
freeze the image encoder on text data; image-text full SFT trains it.

The release's `processor_config.json` names an `Inkling2Processor` and an
`Inkling2ImageProcessor` but ships no implementation, and released Transformers versions have
none, so `data/` provides them. `Inkling2ImageProcessor` implements the release's settings:
images keep their native size, are padded to a multiple of 40 pixels, rescaled by 1/255 and
normalized with mean and standard deviation 0.5, and cut into 40x40-pixel patches repeated
twice in time, one image token each. With `max_image_tokens`, a larger image is encoded at
the largest same-aspect size within that budget, resized with a Lanczos3 kernel in FP32
arithmetic that takes its sine from the C library's single-precision `sinf`; on smooth
images it stays within one level of Pillow's `LANCZOS` filter.
`Inkling2Processor` renders conversations token for token as the release's chat template
does, including a row's `tools` and the `reasoning_effort` in its `chat_template_kwargs`, with
one `<|unused_200054|>` image placeholder per patch where the template writes
one per image, and marks the assistant turns for the loss; it is registered with
`AutoProcessor` for `Inkling2Config`. `Inkling2Config.image_token_id`, which the release's
`config.json` leaves out, defaults to that placeholder (token 200054).
`inkling2_collate_fn` builds the padded batches, and image-text
recipes use the VLM forward step. For full-parameter runs, the model
config supplies the training FLOPs estimate: it counts the sliding-window layers by
their window and includes the relative-logit, gate, sink, and dynamic-convolution
projections.
