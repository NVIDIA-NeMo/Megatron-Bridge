---
name: nemo-mbridge-perf-vision-dp-over-cp
description: Operational guide for sharding a VLM vision encoder across the language model's context-parallel ranks in Megatron-Bridge, including config knobs, code anchors, load-balance pitfalls, and measured impact.
license: Apache-2.0
when_to_use: The vision tower replicates work and activations on every CP rank while the language model is already sharded, or a VLM run at CP>1 OOMs in the encoder; 'vision_dp_over_cp', 'vision CP', 'vision DP over CP', 'image sharding across CP ranks', 'encoder replicated across CP'.
---

# Vision DP Over CP Skill

This skill covers `vision_dp_over_cp`: reusing the language model's
context-parallel process group to shard the **images** (or temporal tubelets) of
a microbatch across CP ranks, so each rank encodes `1/CP` of them instead of all
of them.

It is **not** ring attention and shares nothing with
`nemo-mbridge-perf-hierarchical-context-parallel` beyond the process group.
Vision encoders such as RADIO already attend *within* each image via
`cu_seqlens`, so there is no long sequence to split and no cross-rank attention
communication. The correct mental model is **data parallelism over images,
borrowing the CP group** — a rank encodes whole images, never partial ones.

Shipped for Nemotron Omni. The MCore split/gather helpers are model-agnostic, so
the pattern generalizes to any Bridge VLM with a dynamic-resolution encoder.

## Do Not Confuse With `dist_train.vision_context_parallel_size`

Bridge already has a similarly named and completely different knob:
`vision_context_parallel_size` on the dist-train path
(`src/megatron/bridge/models/qwen_vl/qwen3_vl_provider.py`, consumed in
`src/megatron/bridge/training/initialize.py`).

| | `model.vision_dp_over_cp` (bool) | `dist_train.vision_context_parallel_size` (int) |
|---|---|---|
| Ranks | Reuses the language model's CP ranks | Gives the vision tower its **own** ranks |
| Splits | Whole images across those ranks | The vision encoder's own sequence dimension |
| Encoder `context_parallel_size` | Stays `1` | Set to this value |
| Models | Nemotron Omni | Qwen3-VL dist-train |

They are not alternatives and do not compose in any tested configuration.

## Enablement

```python
cfg.model.context_parallel_size = 2      # the flag is a no-op at CP=1
cfg.model.vision_dp_over_cp = False      # opt out; on by default
```

Defaults to `True`. It self-disables when `context_parallel_size == 1`, so a
recipe that does not raise CP behaves exactly as it did before the flag existed
and reports no error. Opting out is only useful for A/B measurement or to
isolate a suspected split/gather bug.

## Code Anchors

Provider field, `src/megatron/bridge/models/nemotron_omni/nemotron_omni_provider.py`:

```python
vision_dp_over_cp: bool = True
```

Gating, in `NemotronOmniModel.__init__`
(`src/megatron/bridge/models/nemotron_omni/modeling_nemotron_omni.py`):

```python
self.context_parallel_lm = language_transformer_config.context_parallel_size
self.vision_dp_over_cp = vision_dp_over_cp and self.context_parallel_lm > 1
```

The split itself wraps an upstream MCore helper:

```python
return split_to_context_parallel_ranks_dynamic_res(
    images,
    imgs_sizes,
    vision_packed_seq_params,
    patch_dim=self.patch_dim,
    fp8_enabled=False,
    fp8_recipe=getattr(self.config, "fp8_recipe", None),
    num_frames=num_frames,
    temporal_patch_size=temporal_patch_size,
)
```

Project-then-gather, at the end of `_encode_images`:

```python
if shard_vision:
    # Project before the gather so the projector stays sharded, then
    # restore the global feature set every CP rank's media merge needs.
    projected = _project_multimodal_embeddings(self.vision_projection, encoded)
    gathered = gather_from_context_parallel_ranks_dynamic_res(projected, num_padded_ranks)
    return gathered.contiguous()
```

Upstream helpers (do not modify — changes go through the MCore repo):

- `3rdparty/Megatron-LM/megatron/core/models/multimodal/context_parallel.py`
  — `split_to_context_parallel_ranks_dynamic_res`,
  `gather_from_context_parallel_ranks_dynamic_res`
- `3rdparty/Megatron-LM/megatron/core/models/multimodal/llava_model.py`
  — performs the identical split/encode/gather on its temporal path

## Implementation Map

1. **Split.** The MCore helper partitions the patched image tensor on image (or
   tubelet) boundaries and rewrites `imgs_sizes` and the vision
   `PackedSeqParams` (`cu_seqlens`, `max_seqlen`) to describe only the rank-local
   share. Sharding is by **image count**, not patch count; the remainder goes to
   the **last** rank.
2. **Placeholder padding.** When a microbatch has fewer images than CP ranks,
   the helper injects `num_padded_imgs = max(0, cp_size - num_imgs)` dummy `1x1`
   images so no rank runs an empty encoder, and returns `num_padded_ranks` so the
   gather can drop them again.
3. **Encode.** Each rank runs the full vision tower on its own images only.
   Activations, not just compute, are what drop by `1/CP`. The vision encoder's
   own `context_parallel_size` stays hardcoded to `1` in
   `_build_vision_config` — that is correct and unchanged by this feature, since
   the sharding happens *outside* the encoder, over its inputs.
4. **Even-grid padding.** `_pad_patch_grid_to_even` zero-pads an odd patch grid
   before pixel shuffle. Only the injected `1x1` placeholders hit this path;
   real frames are already even in both extents.
5. **Project, then gather.** The vision projector runs on the rank-local shard,
   so it stays sharded too, and only the smaller post-projector features cross
   the wire. The gather is the **only** collective this feature adds.
6. **Merge.** After the gather, every rank holds the full global feature set,
   which is exactly what the unchanged media-merge path expects.

## Measured Impact

Nemotron-3 Nano Omni 30B-A3B, 2 nodes / 16 GPUs, PP=1 DP=2 EP=8 ETP=1 MBS=2
GBS=64 GA=16, with only `vision_dp_over_cp` flipped:

| Dataset     | TP | CP | Step time     | Peak memory |
|-------------|----|----|---------------|-------------|
| InfoVQA     | 4  | 2  | 13.2 -> 12.5 s | 81 -> 73 GB |
| Mantis      | 2  | 4  | 12.2 -> 11.6 s | 81 -> 77 GB |
| llava_video | 4  | 2  | 13.6 -> 13.0 s | 84 -> 75 GB |

It is both a memory win (-5% to -11% peak) and a throughput win (-4% to -5% step
time). Do not describe it as memory-only; a single early video A/B suggested step
time was flat, and the three-dataset sweep contradicted that.

The Mantis row is the weakest memory result, and that is the load-balance
signature described under Pitfalls, not noise.

## Pitfalls

1. **Silent no-op at CP=1.** The flag ANDs with `context_parallel_lm > 1`. Both
   checked-in Nemotron Omni recipes ship CP=1, so the on-by-default value changes
   nothing there. Always confirm CP>1 before attributing a result to this flag,
   and note that raising CP now silently enables image sharding too.
2. **Count-based split leaves the tail rank hot.** Images are divided by count
   with the remainder on the last rank. Multi-image data with a wide per-sample
   image-count spread (e.g. Mantis, 1-5 images/sample at CP=4) balances poorly:
   peak memory is set by the busiest rank, so the win shrinks. A
   patch-count-balanced greedy split on image boundaries is the fix, and it
   belongs **upstream in MCore**, not reimplemented in Bridge. Since the flag is
   on by default, this imbalance is what a multi-image CP>1 run gets unless the
   user opts out.
3. **Process-group divergence is checked, not assumed.** The MCore splitter
   resolves the CP group from global parallel state, so a `ProcessGroupCollection`
   whose `cp` size disagrees with `context_parallel_size` would shard against the
   wrong ranks. Bridge raises instead:
   `"Nemotron Omni vision_dp_over_cp does not match its process group"`.
4. **Odd patch grids break pixel shuffle.** Omni's factor-2 pixel shuffle rejects
   an odd grid, and MCore's `1x1` placeholder images are odd by construction.
   Without `_pad_patch_grid_to_even`, the CP>images case fails only for the
   specific microbatches that trigger placeholder injection — an intermittent
   failure that looks data-dependent.
5. **Do not gather before projecting.** Gathering encoder output first would
   replicate the projector on every rank and move a larger tensor. The ordering
   is deliberate.
6. **The placeholder path is unvalidated end-to-end.** A two-rank unit test
   proves the sharded encoder matches the unsharded one when a microbatch holds
   fewer images than ranks, but no multi-node training run has exercised it, and
   nothing covers CP=8.

## Verification

Unit tests. The second file self-launches a two-rank `torch.distributed` job and
asserts the sharded tower matches the unsharded one for fewer images than ranks,
an exact split, and a remainder; it skips when fewer than two GPUs are visible:

```bash
uv run python -m pytest tests/unit_tests/models/nemotron_omni/test_nemotron_omni_model.py -q
uv run python -m pytest tests/unit_tests/models/nemotron_omni/test_vision_dp_over_cp_distributed.py -q
```

Multi-GPU A/B. Run the same recipe twice at CP>1, flipping only the flag, and
compare reported step time and peak allocated memory:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  scripts/training/run_recipe.py \
  --recipe nemotron_omni_cord_v2_sft_config \
  model.context_parallel_size=2 \
  model.vision_dp_over_cp=false \
  train.train_iters=20
```

Success criteria:

- Peak memory drops relative to the `false` arm; step time does not regress.
- Loss curves of the two arms track each other. The gather makes this
  mathematically equivalent, so a divergence means the split or gather is wrong,
  not that the feature is lossy.
- If nothing changes at all, check that `context_parallel_size > 1` — the flag
  silently self-disables at CP=1.

## Related

- @skills/nemo-mbridge-perf-moe-vlm-training/SKILL.md — VLM training strategy,
  FSDP vs 3D parallel, and why mock image data invalidates VLM benchmarks
- @skills/nemo-mbridge-perf-memory-tuning/SKILL.md
- @skills/nemo-mbridge-perf-hierarchical-context-parallel/SKILL.md — the
  *sequence*-splitting use of the same process group; unrelated mechanism
- @skills/nemo-mbridge-perf-vision-dp-over-cp/card.yaml
