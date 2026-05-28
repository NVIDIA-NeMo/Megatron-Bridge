# Long-Context Examples

This directory contains small examples for long-context training features.

## Dynamic Context Parallel Packing

`dynamic_context_parallel.py` is a local packing demo for Megatron-Core Dynamic
Context Parallelism (DCP). It does not train a model, build a model, or launch
distributed workers. The goal is to show what DCP does to variable-length
samples before the forward pass.

Run it after switching the Megatron-Core submodule to a dev commit that contains
DCP:

```bash
./scripts/switch_mcore.sh dev
uv sync
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
cfg.model.dynamic_context_parallel = True
cfg.model.sequence_packing_scheduler = "default_dynamic_cp"
cfg.model.max_seqlen_per_dp_cp_rank = 64
cfg.model.min_dynamic_context_parallel_size = 1
cfg.train.micro_batch_size = 1
```

The second block shows each input sample length and how many DPxCP ranks it
needs:

```text
sample 0: length=128, gpus_needed=2
sample 2: length=64, gpus_needed=1
```

A length-128 sample needs two ranks because each rank is configured for at most
64 sequence tokens. Length-64 and shorter samples fit on one rank.

The scheduled microbatch block shows the packed THD-style batch metadata that
the scheduled data iterator would yield:

```text
dpxcp_rank 3: sample_ids=[5, 6, 7], lengths=[24, 16, 8], local_cp_size=1
  tokens.shape=(48,), cu_seqlens=[0, 24, 40, 48], max_seqlen=24
```

This means three short samples were packed onto one DPxCP rank:

- `tokens.shape=(48,)`: the packed token tensor has `24 + 16 + 8` tokens.
- `cu_seqlens=[0, 24, 40, 48]`: sequence boundaries inside the flat token
  tensor.
- `max_seqlen=24`: longest individual sequence in this packed microbatch.
- `local_cp_size=1`: this packed microbatch does not need a multi-rank CP
  group.

For longer samples, the output may show the same `sample_id` on multiple DPxCP
ranks with `local_cp_size=2`. In the real forward step, MCore reads
`local_cp_size`, selects the matching dynamic CP group, and slices the packed
THD batch with:

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

## Long-Context SFT Launch Scripts

The `qwen3_600m_sft_128k.sh` and `qwen3_600m_sft_yarn_128k.sh` scripts are
separate long-context SFT launch examples. They are real training launch
scripts, unlike the DCP packing demo above.
