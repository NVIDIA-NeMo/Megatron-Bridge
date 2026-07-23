---
name: parity-testing
description: Structured framework for exact HF↔MCore weight verification, forward-pass logit correlation, and optional strict numerical diagnostics. Use when debugging weight mismatches, verifying HF↔MCore checkpoint round-trips, choosing verification tools, or investigating conversion commits that caused parity failures. References existing tools and the add-model-support skill.
---

# Parity Testing for Megatron Bridge

This skill provides the decision framework for choosing the right
verification tool and interpreting results. For the full model onboarding
workflow (which includes parity testing as milestones 1 and 2), see the
`add-model-support` skill.

## Quick Decision: Which Tool to Run

| What you want to verify | Tool | GPU? | When to use |
|---|---|---|---|
| All weights round-trip exactly (single GPU) | `hf_megatron_roundtrip.py` | No | First check after writing a bridge |
| Weights round-trip with TP/PP/EP | `hf_megatron_roundtrip_multi_gpu.py` | Yes | After single-GPU passes |
| Forward-pass logit correlation | `compare_hf_and_megatron/compare.py` | Yes | After round-trip passes |
| Text generation sanity | `hf_to_megatron_generate_text.py` | Yes | Separate inference evidence |
| Programmatic weight check | `weights_verification_table()` | Yes | Inside Python scripts |
| VLM generation sanity | `hf_to_megatron_generate_vlm.py` | Yes | VLM models |

All tools live under `examples/conversion/`.

## 3-Level Test Strategy

### Level 1: State Dict Round-Trip (exact match)

The fastest and most fundamental check. If mappings can't perfectly
round-trip weights, nothing else will work.

```bash
# Single-GPU round-trip
uv run python examples/conversion/hf_megatron_roundtrip.py \
    --hf-model-id <org>/<model>

# Multi-GPU with TP=2
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id <org>/<model> --tp 2

# Multi-GPU with PP=2
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id <org>/<model> --pp 2
```

**Expected:** Every weight shows "Matches Original: checkmark". Any "X"
means the param mapping has an error.

**Tolerance:** Exact match (`max_diff == 0.0`). Round-trip conversions are
pure tensor reshaping — no floating-point arithmetic is involved.

For programmatic verification inside scripts, use the built-in verifier:

```python
from megatron.bridge.models.conversion.utils import weights_verification_table
weights_verification_table(bridge, hf_pretrained, megatron_model)
```

### Level 2: Forward-Pass Correlation (GPU / bfloat16)

After round-trip passes, verify that the converted model produces strongly
correlated logits and the same next-token prediction. This is a functional
correlation gate, not a claim of bitwise-identical floating-point arithmetic.

```bash
# Compare logits (loads both HF and Megatron models)
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path <org>/<model> --tp 2 \
    --prompt "The capital of France is"
```

**Expected:** Matching next-token predictions and cosine similarity at least
0.99. Equivalently, cosine distance (`1 - cosine_similarity`) must be at most
1%. Record numeric maximum and mean absolute logit differences for diagnosis,
but do not use either absolute difference as a pass/fail guard. A fixed
absolute threshold is not scale- or ULP-aware across BF16 logits and
implementations.

For large models that OOM `compare.py` (which loads both models), use text
generation as a separate inference sanity check:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
    examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path <org>/<model> --tp 2 \
    --prompt "The capital of France is" --max_new_tokens 50
```

Generation does not measure logit correlation and therefore cannot by itself
verify the manual forward-pass item.

### Level 3: Training Parity (optional)

Verify that a few training steps produce decreasing loss. This catches
gradient computation issues that forward-pass tests miss. Use a toy model
with 2 layers and small dimensions. See the functional test pattern in the
`add-model-support` skill (Milestone 3, Phase 6).

## Tolerance Table

| Test Level | Dtype | Device | Required Gate | Report Only |
|---|---|---|---|---|
| Round-trip | float32 | CPU | `max_diff == 0.0` and cosine `== 1.0` | None |
| Forward pass | bfloat16 | GPU | next token matches and cosine `>= 0.99` | max/mean absolute logit difference |
| Forward pass | float16 | GPU | next token matches and cosine `>= 0.99` | max/mean absolute logit difference |

Keep stricter cosine or absolute-difference thresholds as optional diagnostic
targets when investigating arithmetic differences. Do not use them to downgrade
a model-support card after the correlation gate passes.

## Comparison Utilities

These functions are useful when writing custom verification scripts or
debugging failures. They are not part of the Bridge library — copy them
into your script as needed.

```python
import torch


def compare_tensors(a, b, name=""):
    """Compare two tensors and report similarity metrics."""
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0,
    ).item()
    print(f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cosine_sim={cos_sim:.8f}")
    return max_diff, mean_diff, cos_sim


def compare_state_dicts(sd_a, sd_b, prefix=""):
    """Compare two state dicts key-by-key, reporting per-parameter differences."""
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    missing, extra = keys_a - keys_b, keys_b - keys_a
    if missing:
        print(f"{prefix}Missing keys: {sorted(missing)}")
    if extra:
        print(f"{prefix}Extra keys: {sorted(extra)}")
    max_diffs = {}
    for key in sorted(keys_a & keys_b):
        diff = (sd_a[key].float() - sd_b[key].float()).abs().max().item()
        if diff > 0:
            max_diffs[key] = diff
            print(f"{prefix}{key}: max_diff={diff:.6e}")
    if not max_diffs and not missing and not extra:
        print(f"{prefix}All {len(keys_a & keys_b)} parameters match exactly.")
    return missing, extra, max_diffs
```

## Debugging Workflow

When a parity test fails, follow this sequence:

1. **Run single-GPU round-trip** — if this fails, the mapping itself is
   wrong. Check the `mapping_registry()` in the bridge file.

2. **If single-GPU passes but multi-GPU fails** — the TP/PP scatter/gather
   is wrong. Compare the TP=1 result against each TP shard. See the
   `nccl-contiguous-tensors` skill for NCCL-specific issues.

3. **If round-trip passes but forward correlation fails** — the weights loaded
   correctly, but the runtime architectures may differ. A failure means the
   next token differs or cosine similarity is below 0.99; a large absolute
   difference alone is diagnostic, not a failure. Check `provider_bridge()`
   config mapping (normalization, activation, RoPE, etc.).

4. **Use the debugging script template** from the `add-model-support` skill
   to inspect runtime vs safetensors key naming and bridge config mapping.

For the full catalog of pitfalls (QKV interleaving, MoE fused exports, tied
embeddings, FP8 dequantization, TE LayerNorm aliases, etc.), see the
Pitfalls section of the `add-model-support` skill.

## Code Anchors

| Component | Path |
|---|---|
| Single-GPU round-trip | `examples/conversion/hf_megatron_roundtrip.py` |
| Multi-GPU round-trip | `examples/conversion/hf_megatron_roundtrip_multi_gpu.py` |
| Forward-pass comparison | `examples/conversion/compare_hf_and_megatron/compare.py` |
| Text generation | `examples/conversion/hf_to_megatron_generate_text.py` |
| VLM generation | `examples/conversion/hf_to_megatron_generate_vlm.py` |
| Checkpoint CLI | `scripts/conversion/convert.sh` |
| Toy model creator | `examples/conversion/create_hf_toy_model.py` |
| Verification utility | `src/megatron/bridge/models/conversion/utils.py` |
| Adapter verification | `examples/conversion/adapter/verify_adapter.py` |
