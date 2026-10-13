# Beam

Beam is a 52-layer sparse mixture-of-experts language model with about 501B parameters. Three dense layers are
followed by 49 MoE layers with 256 routed experts each (top-8). Attention alternates five sliding-window layers
(window 512, RoPE) with one full-attention NoPE layer, and the last layer uses full attention.

## Modeling

Beam uses the builder-backed API. `BeamBridge` translates the Hugging Face configuration directly into
`BeamModelConfig`, and `BeamModelBuilder` constructs a native Megatron-Core `HybridModel`: each Beam layer is an
attention/MLP (or attention/MoE) pair in the hybrid layer pattern. `BeamModelProvider` remains only as a compatibility
path for provider-based callers.

The Beam stack spec preserves the published decoder math: GQA (48 query heads, 8 KV heads) with Q/K RMSNorm, a
full-width sigmoid attention output gate with a per-head bias, sandwich RMSNorms after attention and after the MLP,
depth-scaled residual connections, SSMax query scaling on the NoPE layers, grouped sigmoid routing (8 groups, top-4
groups, normalized top-8 weights, routed scaling 2.5) with an expert bias and FP32 router parameters, and a tanh
soft cap of 30 on the output logits. The bridge maps the expert-bias update rate to zero, so fine-tuning and the
bounded pretraining recipe keep the published router bias fixed and add no auxiliary load-balancing loss.

## Builder-backed configuration

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("<beam-hf-id>", revision="<immutable-revision>")
model_config = bridge.get_model_config()

# Flat assignment routes to the nested transformer config.
model_config.tensor_model_parallel_size = 2
model_config.expert_model_parallel_size = 16
models = bridge.get_model(model_config)
```

Command-line overrides of model fields must use the nested path, for example
`model.transformer.expert_model_parallel_size=16`. The `train.sh` shortcuts `-tp`, `-pp`, `-cp` and `-ep` write flat
`model.*` keys, which builder-backed configurations reject.

## Recipes

| Recipe | Layout | Purpose |
| --- | --- | --- |
| `beam_pretrain_128gpu_h200_bf16_config` | 16 nodes, TP2 / EP128, SP | Bounded pretraining; BF16 gradients and Adam moments with FP32 master weights |
| `beam_sft_128gpu_h200_bf16_config` | 16 nodes, TP8 / PP2 / EP64, SP | Full SFT under `qwen3_30b_a3b_convergence_v2` (8K packs, GBS 8, DP 8) |
| `beam_peft_16gpu_h200_bf16_config` | 2 nodes, TP2 / EP16, SP | LoRA (rank 8, alpha 16) on `linear_qkv` and `linear_proj` under the same contract |

All three recipes resolve the model from a pinned Hub revision; pass `hf_path` to use a local checkpoint instead. Use
expert parallelism first: the experts hold about 493B of the 501B parameters, so pipeline parallelism does not lower
the memory floor. All-to-all context parallelism needs the KV heads per tensor-parallel rank to be divisible by the
context-parallel size, so CP8 runs at TP1. With context parallelism, SFT switches to per-token loss normalization,
which is why the convergence-contract SFT layout reaches DP8 without it.

## Conversion

The BF16 checkpoint is about 1 TB. GPU import and export on H200 use two nodes at EP16, which places about 69 GiB of
weights on each GPU; EP8 on one node would need about 127 GiB per GPU. Distributed CPU export works across nodes. CPU
import runs in a single process and needs a host with well over 1 TB of memory.

## Verification status

See the [model verification card](../../model_verification_cards/beam/card.yaml) for the pinned source revision,
commands, and current evidence.
