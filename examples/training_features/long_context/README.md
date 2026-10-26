# Long-Context Examples

This directory contains small examples for long-context training features.

## Dynamic Context Parallel Packing

`dynamic_context_parallel.py` is a local packing demo for Megatron-Core Dynamic
Context Parallelism (DCP v2). It does not train a model, build a model, or launch
distributed workers. The goal is to show what DCP does to variable-length
samples before the forward pass.

It runs on the Megatron-Core `main` pin this repository ships, so no submodule
switch is needed:

```bash
uv run python examples/training_features/long_context/dynamic_context_parallel.py
```

The example uses Megatron-Core's `DefaultDynamicCPScheduler` on toy sequence
lengths:

```text
[128, 96, 64, 48, 32, 24, 16, 8]
```

with `max_seqlen_per_rank=64`, `dp_size=2`, and `cp_size=2`.

### What It Prints

The first block shows the Bridge config knobs used in a real DCP run:

```python
cfg.dataset.enable_global_batch_packing = True
cfg.model.dynamic_context_parallel = True  # selects the default_dynamic_cp scheduler
cfg.model.max_seqlen_per_dp_cp_rank = 64
cfg.model.min_dynamic_context_parallel_size = 1
cfg.train.micro_batch_size = 1
```

The second block shows the CP group sizes the `dp x cp` pool can form and the
smallest group each input sample needs:

```text
Valid group sizes for the dp x cp = 4 pool: [1, 2, 4]
sample 0: length=128, min_group_size=2
sample 2: length=64, min_group_size=1
```

A length-128 sample needs two ranks because each rank is configured for at most
64 sequence tokens. Length-64 and shorter samples fit on one rank.

The scheduled microbatch block shows the packed THD-style batch metadata that
the scheduled data iterator would yield, together with the scheduler's own
estimate of the work each rank receives (the sum of `length ** 2 / group size`,
in relative units):

```text
microbatch 1:
  dpxcp_rank 2: sample_ids=[3, 6], lengths=[48, 16], local_cp_size=1, estimated_work=2560
    tokens.shape=(64,), cu_seqlens=[0, 48, 64], max_seqlen=48, tokens[:8]=[3000, 3001, ...]
  dpxcp_rank 3: sample_ids=[5, 7], lengths=[24, 8], local_cp_size=1, estimated_work=640
    tokens.shape=(32,), cu_seqlens=[0, 24, 32], max_seqlen=24, tokens[:8]=[5000, 5001, ...]
  critical path (busiest rank, relative units): 2560
```

This means short samples were packed onto single DPxCP ranks:

- `tokens.shape=(64,)`: the packed token tensor has `48 + 16` tokens.
- `cu_seqlens=[0, 48, 64]`: sequence boundaries inside the flat token tensor.
- `max_seqlen=48`: longest individual sequence in this packed microbatch.
- `local_cp_size=1`: this packed microbatch does not need a multi-rank CP
  group.

The scheduler is packing-aware: it may place a short sample on a larger group
than it needs when that lowers the busiest rank's workload. In the default run
the 64-token sample fits one rank but runs on a two-rank group, for that reason.
In the real forward step Megatron-Core reads `local_cp_size`, selects the
matching pre-created dynamic CP group, and slices the packed THD batch with:

```python
get_batch_on_this_rank_for_sequence_packing(..., dynamic_cp=True)
```

### Useful Variations

Change the toy lengths:

```bash
uv run python examples/training_features/long_context/dynamic_context_parallel.py \
  --lengths 256 128 72 64 32 16
```

Change the rank capacity:

```bash
uv run python examples/training_features/long_context/dynamic_context_parallel.py \
  --max-seqlen-per-rank 128
```

Change the toy DP/CP topology:

```bash
uv run python examples/training_features/long_context/dynamic_context_parallel.py \
  --dp-size 4 --cp-size 2
```

Restrict the scheduler to a subset of the valid group sizes:

```bash
uv run python examples/training_features/long_context/dynamic_context_parallel.py \
  --dp-size 2 --cp-size 2 --cp-group-sizes 1 4
```

## Long-Context SFT Launch Scripts

The `qwen3_600m_sft_128k.sh` and `qwen3_600m_sft_yarn_128k.sh` scripts are
separate long-context SFT launch examples. They are real training launch
scripts, unlike the DCP packing demo above.

## Dynamic Context Parallel Training on Mock Data

`qwen35_35b_a3b_dynamic_cp.py` is a real training launch script. It builds the
`qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config` recipe, applies
dotted config overrides, and trains. It runs on the Megatron-Core `main` pin
this repository ships.

`--tiny-model` is the supported way to run it today. It shrinks the recipe to a
Qwen3.5-shaped attention and MoE model (4 layers, 8 experts, 8K cap) and
replaces the dataset with a generated variable-length SFT file, so it needs no
download and no checkpoint. The model runs `world_size / 2` data-parallel
replicas of CP2 with EP2, uses full-attention layers only, and drops the
recipe's multi-token-prediction layer. Mock runs write and resume no
checkpoints, so repeated runs and dynamic-versus-static runs start fresh.

Dynamic CP on two GPUs:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model
```

Static-CP baseline on the same packing path (`dp_balanced`), for comparisons:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model --static-cp
```

The generated data has lognormal lengths: most samples fit within one rank's
4K-token share and a tail needs a larger group. `dataset.enable_global_batch_packing=True`
packs the global batch once per step, and `model.dynamic_context_parallel=True`
runs every packed microbatch on a context-parallel group sized for its
longest sequence.

### Limits of dynamic CP on Megatron-Core `main`

Dynamic CP on `main` covers full-attention and MoE layers. Two things in the
full Qwen3.5-35B-A3B recipe are not supported under it yet, and
`ConfigContainer.validate` rejects the recipe with an explanation instead of
letting the run fail later:

- **Gated DeltaNet layers.** Megatron-Core sizes their context-parallel
  exchange from the static CP group, so a bin placed on a smaller runtime group
  fails in the forward pass.
- **Multi-token prediction.** With microbatches on different context-parallel
  groups the MTP token roll hangs. Static CP with MTP is unaffected.

### Measured on GB200 with mock data

Four GB200 GPUs in one node, NeMo 26.10 rc5 container, Megatron-Core `main`
pin, `--tiny-model` data, with the first 12 steps dropped as warm-up (one-time
compilation, communicator setup and shape-dependent kernel setup). Both modes
process identical global batches in the same order and reach the same loss
(final losses agree to four decimals), so the comparison is step time over the
same steps. The first two rows are single 40-step runs. The CP4 / 128 row is two
runs in opposite order (dynamic first, then static first), compared over steps 13
to 37 and 13 to 30.

| Topology | Global batch | Static CP median step | Dynamic CP median step | Static / dynamic total time |
|---|---|---|---|---|
| DP2 x CP2 | 32 | 702 ms | 638 ms | 0.89 (dynamic 11% slower) |
| CP4 | 32 | 771 ms | 645 ms | 1.10 |
| CP4 | 128 | 2.82 s | 2.10 to 2.15 s | 1.41 and 1.46 (1.31 on the matched 13 to 30 window) |

- Dynamic CP pays off when static CP is expensive. With CP2 inside two
  data-parallel replicas on one NVLink domain, communication is cheap and the
  tiny model is dominated by per-step overhead, so dynamic CP is slightly slower.
  With one CP4 group across all four GPUs and a larger global batch, short
  samples run on single GPUs with no context-parallel exchange and the gain
  grows.
- The gain depends on the length mix. This mock stream has many samples that fit
  one rank. A stream where nearly every sample needs the whole group gains
  nothing from dynamic CP.
- Steps vary because each step draws a different mix of lengths, and each newly
  seen packed shape pays a one-time kernel setup cost, so short runs overstate
  per-step variance. Compare totals over the same steps, as above.

See `docs/training/dynamic-context-parallel.md` for when dynamic CP helps and the
configuration rules.
