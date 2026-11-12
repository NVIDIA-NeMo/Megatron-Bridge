# Inkling2

Inkling2 is an image-and-text mixture-of-experts model with 594B total and about 17B
active parameters. Its 48 decoder layers replace the residual stream with Block
Attention Residuals, interleave 1024-token sliding-window attention with full attention
on every sixth layer, and route each token to 8 of 384 experts. The release stores the
routed experts in NVFP4 and MXFP8. Megatron Bridge converts the complete checkpoint,
including the image encoder and its projection, and keeps the speculative-decoding draft
network outside the Megatron model. The release,
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small),
ships no modeling code, and released Transformers versions do not implement Inkling 2
(huggingface/transformers#50682 adds it). The
[model README](../../../src/megatron/bridge/models/inkling/README.md) describes the
implementation, the conversion of the quantized experts, and the supported parallelism.

## Conversion

Import the checkpoint into a Megatron checkpoint; the import dequantizes one expert at
a time. Megatron checkpoints reshard on load, so later runs may use another layout.

```bash
./scripts/conversion/convert.sh import --executor slurm --device gpu \
  --nodes 8 --gpus-per-node 4 --hf-model thinkingmachines/Inkling-2-Small \
  --megatron-path work/inkling2/megatron --torch-dtype bfloat16 --tp 1 --pp 4 --ep 32 --etp 1
```

Export writes the routed experts back in their source format and scales, so exporting an
unchanged model reproduces the released tensors byte for byte:

```bash
./scripts/conversion/convert.sh export --executor slurm --device gpu \
  --nodes 8 --gpus-per-node 4 --hf-model thinkingmachines/Inkling-2-Small \
  --megatron-path work/inkling2/megatron/iter_0000000 --hf-path work/inkling2/hf-export \
  --torch-dtype bfloat16 --tp 1 --pp 4 --ep 32 --etp 1 --distributed-save
```

After training, the export quantizes the trained experts with the original scales, so
updates smaller than one quantization step of an expert block do not survive the export.

## Training recipes

The GB200 recipes are defined in
[inkling2.py](../../../src/megatron/bridge/recipes/inkling/gb200/inkling2.py) and warm-start
from the imported checkpoint with `--pretrained_checkpoint`. Each expert-parallel group
stays inside one NVL72 NVLink domain and dispatches the experts with HybridEP, and the
image encoder stays frozen on text data.

| Recipe | Workload | Nodes | Layout | Recompute |
| --- | --- | ---: | --- | --- |
| `inkling2_small_pretrain_128gpu_gb200_bf16_config` | Text pretraining, GBS 1024, sequence length 4096 | 32 | TP1/PP4/EP32, 13/13/12/10 blocks per stage | Full |
| `inkling2_small_sft_128gpu_gb200_bf16_config` | Full SFT on 10K Tulu 3 conversations packed to 8K, GBS 8 | 32 | TP4/PP4/EP32 | None |
| `inkling2_small_sft_96gpu_gb200_bf16_config` | The same full SFT run on fewer GPUs | 24 | TP4/PP3/EP32 | First 20 of 32 layers per stage |
| `inkling2_small_sft_128gpu_gb200_bf16_long_context_config` | Full SFT on packed 32K Tulu 3, GBS 8 | 32 | TP4/PP4/EP32/CP4 | First 20 of 24 layers per stage |
| `inkling2_small_peft_32gpu_gb200_bf16_config` | LoRA on the attention projections, packed 8K Tulu 3, GBS 8 | 8 | TP4/PP1/EP32 | None |
| `inkling2_small_peft_16gpu_gb200_bf16_config` | The same LoRA run on fewer GPUs | 4 | TP2/PP1/EP16 | Full |
| `inkling2_small_sft_128gpu_gb200_bf16_multimodal_config` | Full SFT on image-text CORD-v2, image encoder trained, GBS 64 | 32 | TP4/PP4/EP32 | Full |
| `inkling2_small_peft_32gpu_gb200_bf16_multimodal_config` | LoRA on image-text CORD-v2, image encoder frozen, GBS 8 | 8 | TP4/PP2/EP16 | None |

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./scripts/training/train.sh --nodes 32 --gpus-per-node 4 \
  --env PYTORCH_CUDA_ALLOC_CONF --additional-slurm-params 'segment=8' \
  --recipe inkling2_small_sft_128gpu_gb200_bf16_config --mode sft \
  --pretrained_checkpoint work/inkling2/megatron
```

HybridEP needs every expert-parallel group inside one NVL72 domain, and each group is a run
of consecutive nodes: 8 for EP32 and 4 for EP16. On a cluster with Slurm's block topology,
`segment=8` places every 8 consecutive nodes in one NVL72 rack; use `segment=4` for the
4-node LoRA recipe.

Launch every recipe with expandable segments set as above. The recipes declare the same
allocator setting, but `run_recipe.py` applies recipe environment variables only after
Transformer Engine's import has initialized CUDA, and PyTorch reads its allocator settings
once, when CUDA initializes. Without expandable segments the caching allocator fills the
GPUs: a 16-GPU LoRA run with all-to-all dispatch reserved 183 GB for a 130 GB peak, and NCCL
ran out of memory creating a communicator after the first step.

The pretraining recipe splits the 48 decoder blocks unevenly over its four pipeline
stages. A block's cost grows with depth, as the AttnRes state it reads grows, and the last
stage also computes the output layer and loss, so with 12 blocks per stage the last stage
sets the pace and the step is about 11% slower.

The packed SFT and LoRA recipes train GBS 8 on eight data-parallel replicas, one pack
each, so a step is a single pass through the 48 decoder blocks: more pipeline stages add
GPUs without shortening the step. Variants that train the same run on another layout
change only the parallel layout and activation recompute; they keep the data,
batch, optimizer, schedule, and precision. The 96-GPU SFT recipe is the smallest layout
that holds the full-SFT optimizer state, and the 16-GPU LoRA recipe the smallest that
holds the frozen base. The recipes are short verification runs whose learning-rate decay
ends at the last step, so raise `scheduler.lr_decay_iters` together with the step count to
train longer.

Inkling2 attention runs in compiled FlexAttention, which compiles once per sequence
shape. The packed SFT recipes therefore pad every pack to the full sequence length
(`pad_to_max_length` in `dataset.dataset_kwargs`). The recipes also compile Inkling2's
elementwise operators, the AttnRes mixtures, short convolutions, L2 clips, SiTU
activations, and residual additions (`model.inkling2_compile_ops`): the first step compiles
about 90 graphs per rank, roughly two minutes, and later steps run them as fused kernels.
Set it to false to debug in eager mode. The recipes collect Python garbage every 100 steps
(`train.manual_gc_interval`), because a full collection stalls the step it runs in by seconds. The 32K recipe adds context
parallelism over four ranks. With the global batch fixed at 8, that leaves two
data-parallel replicas that each run four packs per step, which keeps the four pipeline
stages busy. Context-parallel SFT needs every packed sequence aligned to
lcm(2 x CP, CP x TP) tokens (16 at TP4/CP4), per-token loss
(`model.calculate_per_token_loss`), and `ddp.average_in_collective=False`; the recipe
sets all three, and a different CP size needs its own packed-data directory. Image
inputs are not supported with context parallelism yet. Conversations
are rendered with `INKLING2_CHAT_TEMPLATE` from
`megatron.bridge.models.inkling.inkling2_chat_template`: the release's `chat_template.jinja`,
which it renders identically, plus two additions. An assistant's reasoning may also come in
the `thinking` field of Bridge's dataset adapters, and assistant turns are marked as
`{% generation %}` spans for the loss mask. As in the release's template, tool declarations
come first, a `Thinking effort level` system line precedes the first non-system message
(the `reasoning_effort` template argument, 0.9 by default), reasoning is its own block, each
tool call is a structured `<|content_invoke_tool_json|>` block, and tool results are named
after the call they answer. The supervised span of an assistant turn includes its message
headers, tool calls, and end-of-sampling token.

The image-text recipes read CORD-v2 receipts through `Inkling2Processor`, which
`AutoProcessor` resolves from the checkpoint directory, so point
`dataset.hf_processor_path` at the same checkpoint as `--pretrained_checkpoint`'s source.
An image takes one token per 40x40-pixel patch at its native size, as the release's
`processor_config.json` specifies. CORD-v2 phone photos reach 7,676 image tokens, so
both recipes set `dataset.hf_processor_kwargs={"max_image_tokens": 3072}`: the 106 of
800 training receipts above that budget are encoded at the largest same-aspect size
within it, resized with a Lanczos3 filter. The longest
training sample is then 3,904 tokens, so no answer is cut at the 4096-token sequence
length; a sample whose image tokens do not fit is rejected rather than cut. Batches are
padded to the full sequence length, without packing.

## Inference

Megatron's KV-cache inference is not implemented for Inkling2, so generation must
recompute the full prefix at every step. Logit comparisons use the full-prefix forward
(`--legacy-full-prefix`).
