# Intra-Microbatch Reordering for MegatronMIMO

`MegatronMIMOFeatureConfig` adds an optional path that cuts the multimodal
data-parallel straggler in non-colocated MegatronMIMO training (e.g. Qwen3.5-VL: a
separate vision encoder and language model wired by `BridgeCommunicator`).
Per-micro-batch vision load is uneven across samples, so a data-parallel rank that draws
a heavy-image shard stalls the whole group every step. The feature rebalances that load
across the module DP group before compute.

All knobs below are off by default. Engine code:
`src/megatron/bridge/data/megatron_mimo/reorder_buffer.py`; unit tests:
`tests/unit_tests/data/megatron_mimo/`.

## What It Is

The feature has three layers, each enabling the next:

| Layer | Config / flag | Effect |
|---|---|---|
| Scalable data parallelism | `scalable_dp` / `--scalable-dp` | Each rank reads only its disjoint `1/dp` shard instead of every rank reading the full micro-batch and slicing locally. Removes per-rank read/IO overhead and enables vision DP > 1. |
| Intra-microbatch reordering | `intra_microbatch_reorder` (on by default once `scalable_dp` is set) | Rebalances per-sample vision load across the module DP group via a per-sample cost all-gather plus a ragged all-to-all, run GPU-resident. Removes the straggler tail. |
| In-batch sequence packing | `pack_sequences_in_batch` / `--pack-sequences-in-batch` | Packs each language shard's real tokens into a single `[1, T]` THD sequence so the LM skips padding compute. |

Balancing is driven by an intrinsic, collation-independent per-sample cost derived from
the image-placeholder token count in `input_ids` (proportional to the vision patch count).
Because that count is present and identical on the vision and language modules and on every
pipeline stage, both modules derive the same assignment with no cross-module communication,
and `BridgeCommunicator` keeps vision replica *r* paired with language replica *r*. The
padded sequence length is never used as a cost — it is collation-dependent and would
mispair the vision/language fan-out.

The exchange supports a variable number of images per sample (0 = text-only, 1, or N) and
heterogeneous DP (`vision_dp != language_dp`). By default it overlaps with compute on a side
stream (`overlap_intra_microbatch_reorder`, requires `CUDA_DEVICE_MAX_CONNECTIONS != 1`),
which is what turns the read-sharding win into a net throughput win instead of paying the
transfer on the critical path.

## When to Use It

Use it for non-colocated MegatronMIMO VLM finetuning where vision load is the per-step
straggler — receipt/document datasets (e.g. CORD-v2) with uneven image counts per sample
are the canonical case. The win grows with per-rank batch size and with sequence packing,
since a packed rank's length is dominated by its samples' image-placeholder tokens.

It is not useful for text-only training or colocated single-module setups, where there is
no cross-module DP straggler to remove.

## Configuration

- `scalable_dp=True` — required for reordering and packing.
- `cost_linear_vit > 0` or `cost_linear_lm > 0` — a non-degenerate per-sample cost
  `cost_linear_vit · patches + cost_linear_lm · real_tokens` (`finalize()` enforces it).
  `cost_linear_lm` defaults to `0.0` (patch-only cost).
- `overlap_intra_microbatch_reorder=True` (default) — overlap the exchange; set
  `--no-overlap-intra-microbatch-reorder` to run it synchronously for debugging.
- `reorder_window_size` — micro-batches per windowed exchange (default 1).

## Constraints and Support

| Configuration | Status |
|---|---|
| Homogeneous and heterogeneous DP (`vision_dp != language_dp`) | Supported |
| `PP > 1` | Supported with untied checkpoints. Tied-embedding + `PP > 1` is blocked by the upstream #3905-family cross-PP embedding all-reduce; use an untied checkpoint (LM head = copy of the input embedding). |
| In-batch packing + `PP > 1` | Supported |
| Non-`single` sampler (cyclic/batch) | Guarded with `NotImplementedError` — the exchange assumes contiguous sharding; implementable later by all-gathering each rank's real global indices |
| `TP > 1` | Untested |
| `CP > 1` | Blocked upstream (`bridge_communicator` asserts language-grid CP size 1) |

The on-device exchange needs module `dp ≥ 2` (≥ 4 ranks non-colocated), which exceeds the
2-GPU functional-test cap, so CI coverage is the emulated all-to-all unit tests plus a 2-GPU
smoke (`test_reorder_exchange.py`); the full path is validated manually.

## Validation

**Correctness.** `scalable_dp` (with and without reorder) tracks the full-batch-read
baseline per iteration to within float/reduction-order noise (`|Δ loss|` ≈ 1e-4–9e-4,
non-growing), and processes the identical set of images (byte-identical global patch count) —
it redistributes load, it does not skip work. Per-vision-rank patch spread tightens from
≈1.55× to ≈1.04× at dp8/dp8. Reorder + `PP > 1` (untied) and packing + `PP > 1` both reach
`lm loss < 2`, tracking their no-reorder baselines.

**Throughput.** Single 8×A100-80GB node, vision dp4 / language dp4, Qwen3.5-0.8B (VL) +
CORD-v2, seq 2048, sequence packing on, MBS/GBS 32 (8 examples/rank), patch-only cost. 500
iters, per-iteration time with the first 10 (warmup) excluded
(`expandable_segments:True`, `CUDA_DEVICE_MAX_CONNECTIONS=8`):

| config | p50 | p90 | p99 | mean | max |
|---|---|---|---|---|---|
| base + pack (full-batch read) | 968 | 1523 | 2342 | 1075 | 2659 |
| scalable read only + pack | 793 | 864 | 910 | 798 | 1493 |
| scalable + reorder + pack (default) | 752 | 786 | 863 | 755 | 1410 |

Scalable data parallelism is the larger win: reading only the `1/dp` shard instead of the full
batch on every rank cuts the mean iter time 1.35× and collapses the full-read tail (p99 2342 →
910 ms). On top of that, reorder removes the residual per-rank straggler for a further ~5–9%
(mean 798 → 755 ms, p90 −9%) with a tighter distribution, since packing makes each rank's `[1,
T]` length scale with its image load. The reorder gain is realized with overlap on (the
synchronous exchange otherwise lands on the critical path), and grows with the per-rank load
imbalance — larger DP, larger image-size variance, and larger per-rank batch.

## Related

- [Packed Sequences](packed-sequences.md) — the sequence packing this builds on.
