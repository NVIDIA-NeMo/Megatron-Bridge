# MegatronMIMO Intra-Microbatch Reordering

Intra-microbatch reordering rebalances the per-sample vision load of each
micro-batch across a module's data-parallel ranks in non-colocated
MegatronMIMO training. It removes the straggler that appears when one
data-parallel rank happens to draw the heaviest images every step.

This page is the stable overview of what the feature does, how to enable it,
and which constraints are durable. The implementation lives in
`megatron.bridge.data.megatron_mimo.reorder_buffer`; the Qwen3.5-VL example
under `examples/megatron_mimo/qwen35_vl/` wires it end to end.

## What It Is

In a non-colocated MegatronMIMO run the vision encoder and the language model
run on disjoint rank groups, each with its own data-parallel size. With
scalable data parallelism (`megatron_mimo_scalable_dp`), every rank reads only
its own shard of each global micro-batch, so the images a rank must encode are
whatever the sampler happened to assign to it. Image sizes vary widely, so the
per-rank vision work varies too, and the slowest rank gates every step.

The reorder buffer wraps the training data iterator and, for each micro-batch
(or window of micro-batches), does three things:

1. Computes a per-sample cost on every rank from fields that are identical on
   the vision and language modules: the image-placeholder token count in
   `input_ids` (proportional to the number of vision patches) and, optionally,
   the real token count from `attention_mask`.
2. All-gathers the costs over the module's data-parallel group and computes one
   balanced assignment of samples to ranks. The assignment is a deterministic
   function of the costs, so the vision and language modules compute the same
   one without talking to each other.
3. Exchanges samples with a single ragged `all_to_all` so each rank ends up
   holding its assigned samples in a canonical order.

Because both modules apply the same assignment, sample *i* on vision replica
*r* is still sample *i* on language replica *r*, and the `BridgeCommunicator`
fan-out stays aligned. Balancing is done over the canonical grid (the least
common multiple of the module data-parallel sizes), so modules with different
data-parallel sizes, including a module with data-parallel size 1, agree on
the same order. A data-parallel-1 module has nothing to exchange and simply
permutes its whole micro-batch locally.

By default the exchange for the next window runs on a background thread with a
dedicated NCCL process group, overlapping with the current window's compute.

## When to Use It

Reordering is a good fit when all of the following are true:

- the run is non-colocated MegatronMIMO with `megatron_mimo_scalable_dp=True`
- at least one module has data-parallel size 2 or more
- image load varies across samples (document, receipt, or mixed text-and-image
  SFT data), so per-rank vision time is the per-step straggler

It does not help text-only training, colocated single-module runs, or data
where every sample carries the same number of similarly sized images.

## Bridge Configuration

All fields live on `DirectHFSFTDatasetConfig` and default to off.

| Field | Default | Meaning |
|---|---|---|
| `megatron_mimo_scalable_dp` | `False` | Read only this rank's canonical-grid shard. Required by reordering. |
| `megatron_mimo_intra_microbatch_reorder` | `False` | Enable the reorder buffer. |
| `megatron_mimo_reorder_encoder_cost_weight` | `1.0` | Cost weight per vision patch. |
| `megatron_mimo_reorder_language_cost_weight` | `0.0` | Cost weight per real language token; `0.0` balances on vision load only. |
| `megatron_mimo_reorder_overlap` | `True` | Exchange the next window on a background thread. |
| `megatron_mimo_reorder_window_size` | `1` | Micro-batches balanced together in one exchange. |

Only the ratio of the two cost weights matters; at least one must be positive.

The Qwen3.5-VL example exposes the same fields as `--scalable-dp`,
`--intra-microbatch-reorder`, `--reorder-encoder-cost-weight`,
`--reorder-language-cost-weight`, `--no-overlap-intra-microbatch-reorder`, and
`--reorder-window-size`:

```bash
python examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py \
  --component "language=tp=1,pp=1,dp=2,rank_offset=0" \
  --component "images=tp=1,dp=2,rank_offset=2" \
  --scalable-dp --intra-microbatch-reorder \
  ...
```

Overlap is refused under `CUDA_DEVICE_MAX_CONNECTIONS=1`: with a single
hardware queue the prefetch thread's all-to-all and the DDP all-reduce execute
in issue order, that order differs across ranks, and training deadlocks
(reproduced on a 4-rank run and on a 6-rank TP=2 run). Use
`--no-overlap-intra-microbatch-reorder` in that case; it issues both
collectives from the training thread and completed 1500 iterations under the
same setting. Megatron Bridge does not require the variable for tensor
parallelism; it is a Megatron-LM sequence-parallel performance setting, so TP
users choose between that setting with the non-overlapped reorder, or overlap
with the variable unset.

## Stable Constraints

| Configuration | Status |
|---|---|
| Heterogeneous module data parallelism (for example vision dp 1 or 2 with language dp 2 or 4) | Supported. Balancing uses the canonical grid; a dp-1 module reorders locally. |
| Language pipeline parallelism > 1 | Supported. Every language stage receives `input_ids` so all stages compute the same assignment. |
| Encoder pipeline parallelism > 1 | Not supported; the example raises before training starts. |
| `--pad-to-seq-length false` (variable batch width) | Not supported with reordering; the example raises before training starts. Exchanged samples are concatenated back into one micro-batch, so all ranks must pad to the same length. |
| `micro_batch_size` not divisible by the canonical grid size | Rejected by the canonical loader. |
| `drop_last=False`, or a dataloader type other than `single` / `cyclic` | Rejected by scalable data parallelism. |
| Expert tensor parallelism > 1 | Rejected by scalable data parallelism. |
| Tensor parallelism > 1, context parallelism > 1 | Untested. |

A real exchange needs a module with data-parallel size 2 or more, which
exceeds the two-GPU functional-test budget. CI covers the exchange arithmetic
with CPU-emulated all-to-all unit tests and a two-GPU smoke test; multi-rank
runs are validated manually.

## Feature Interactions

- **In-batch sequence packing** (`enable_in_batch_packing`): reordering runs in
  the data iterator, before the forward step packs each language shard, so the
  two compose. Packing drops padding, so the exchange transfers padded rows but
  model compute is unaffected.
- **Checkpoint resume**: reordering changes which rank holds a sample, not how
  many samples are consumed, so the resume arithmetic of the canonical sampler
  is unchanged.
- **Cost weights and the language term**: setting
  `megatron_mimo_reorder_language_cost_weight > 0` counts real tokens from
  `attention_mask`; the example ships `attention_mask` on every stage for this
  purpose and the forward step drops it before the model.

## Related Docs

- [Packed Sequences](packed-sequences.md)
- Example: `examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py`
