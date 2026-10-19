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

## Qwen3.5-35B-A3B with Dynamic Context Parallel

`qwen35_35b_a3b_dynamic_cp.py` is a real training launch script. It builds the
`qwen35_text_35b_a3b_sft_8gpu_gb200_bf16_dynamic_cp_config` recipe, applies
dotted config overrides, and trains. The recipe fine-tunes Qwen3.5-35B-A3B on
CoderForge-Preview SWE-Rebench trajectories with TP1, PP1, CP4, and EP8 on eight
GB200 GPUs and a 32K-token cap. `dataset.enable_global_batch_packing=True` packs
the global batch once per step, and `model.dynamic_context_parallel=True` runs
every packed microbatch on a context-parallel group sized for its longest
sequence. The script needs the Megatron-Core dev pin and a converted pretrained
checkpoint:

```bash
./scripts/switch_mcore.sh dev && uv sync
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py \
  checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b \
  train.train_iters=20 logger.log_interval=1
```

Static-CP baseline on the same packing path:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py \
  model.dynamic_context_parallel=false model.sequence_packing_scheduler=dp_balanced
```

Two-GPU smoke test with a tiny Qwen3.5-shaped model on a generated
variable-length SFT file, with no download and no checkpoint:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 \
  examples/training_features/long_context/qwen35_35b_a3b_dynamic_cp.py --tiny-model
```

### Measured on GB200

Eight GB200 GPUs in one NVL72 rack, NeMo 26.06.01 container, Megatron-Core dev
pin with [Megatron-LM #7417](https://github.com/NVIDIA/Megatron-LM/pull/7417),
random initialization, the first 4,000 SWE-Rebench trajectories, and 12 steps
of which the first four are dropped. Throughput is Bridge's model TFLOP/s per
GPU.

| Configuration | Median step | Median TFLOP/s per GPU | Peak allocated per GPU |
|---|---|---|---|
| Recipe: dynamic CP, full recompute, 16 loader workers | 88 s | 90 | 108 GB |
| Static CP4 (`dp_balanced`), otherwise identical | 83 s | 93 | 108 GB |
| Recipe with 2 loader workers | 254 s | 31 | 108 GB |
| Recipe with selective recompute of `moe_act` and `layernorm` | 75 s over four steps, then NCCL out of memory | 106 | 163 GB |

- GPT-SFT tokenizes each trajectory lazily, which takes about 9 seconds per
  sample here, and the packing scheduler pulls a whole step's samples at once.
  With two loader workers the GPUs wait on data; the recipe uses 16.
- About nine in ten of these trajectories are longer than the 32K cap, so
  nearly every packed sequence needs the full CP4 group and dynamic CP has no
  short sequences to place on smaller groups. Its median step is 5% slower
  than static CP4 here, within this run's step-to-step noise. Dynamic CP pays
  off on streams with a meaningful share of short sequences.
- Selective recompute runs about 13% faster per step, but it reserves over
  170 GB per GPU, and the next communicator dynamic CP creates for a new group
  size fails to allocate. Full recompute leaves that headroom.

See `docs/training/dynamic-context-parallel.md` for when dynamic CP helps, the
configuration rules, and the notes for wide-head MoE models.
