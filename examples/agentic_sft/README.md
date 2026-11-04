# Qwen3.5-35B-A3B with dynamic context parallelism on 4×GB200

`qwen35_text_35b_a3b_4gpu_gb200_fp8mx_dynamic_cp_config` trains the **full 40-layer Qwen3.5-35B-A3B (plus one MTP
layer)** on one four-GPU GB200 node with **dynamic** context parallelism (CP), on **mock data** whose sample lengths
follow an agentic tool-use workload. Dynamic CP runs each packed bin on a CP1, CP2 or CP4 group sized for its longest
sequence instead of splitting every bin across the full CP4 group. This directory shows how to run the recipe in both
modes and compare them.

The model and training configuration follow the Megatron-LM agentic SFT example by Zhongbo Zhu
([`examples/agentic_sft`, commit `420686e`](https://github.com/zhongbozhu/Megatron-LM/commit/420686e1a7e9d51d422e8ac1a18a35818843bb86)):
TP1/PP1/base CP4/EP4/ETP1, 64K context, micro batch 1, global batch 32, per-token loss, MXFP8 compute, FP32 Adam state
offloaded to the host, HybridEP and full recompute. The model is built from the pinned Hugging Face config through the
Qwen3.5 bridge.

## Requirements

* Four GB200 GPUs and a container with Transformer Engine, CuTeDSL and DeepEP HybridEP.
* Megatron-Core with [NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048), which propagates
  the runtime CP group through the Gated DeltaNet layers. Without it, a bin on a group smaller than the base CP fails
  with `GDN: cu_seqlens_q[-1]=16384 does not match total_sequence_length=65536`.
* A large host-memory allocation for the CPU optimizer.

## Mock data

`MockGPTSFTDatasetConfig` generates random token IDs, so no dataset or tokenizer is read. Half of the samples need at
most CP1 capacity (up to 16K tokens), 30% need CP2 (16K–32K) and 20% need CP4 (32K–64K), with lengths uniform inside
each bucket. That puts about 19% / 35% / 46% of the tokens in the three buckets and averages 21K tokens per sample,
close to the original example's mixed data (19% / 36% / 45%). The last 20% of each sample is supervised. Lengths and
tokens are deterministic in the seed, so two runs see identical samples in identical order. The loss is not
meaningful: random tokens have nothing to learn.

## Run

From scratch, as in the original example:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py \
    --recipe qwen35_text_35b_a3b_4gpu_gb200_fp8mx_dynamic_cp_config --mode pretrain \
    train.train_iters=200 scheduler.lr_decay_iters=200 scheduler.lr_warmup_iters=20
```

Static-CP4 baseline on the same packing path and the same data:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py \
    --recipe qwen35_text_35b_a3b_4gpu_gb200_fp8mx_dynamic_cp_config --mode pretrain \
    train.train_iters=200 scheduler.lr_decay_iters=200 scheduler.lr_warmup_iters=20 \
    model.dynamic_context_parallel=false
```

Offline packing, as the original example's `offline_dynamic_cp` mode: the same samples are packed ahead of time,
first-fit decreasing, into 64K-token rows, each step takes eight rows, and dynamic CP re-bins their samples. Add
`model.dynamic_context_parallel=false` for static CP4 on the same rows:

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 scripts/training/run_recipe.py \
    --recipe qwen35_text_35b_a3b_4gpu_gb200_fp8mx_offline_dynamic_cp_config --mode pretrain \
    train.train_iters=200 scheduler.lr_decay_iters=200 scheduler.lr_warmup_iters=20
```

To start from converted weights, use `--mode sft` with `checkpoint.pretrained_checkpoint=/path/to/megatron/checkpoint`.
Nothing is saved unless `checkpoint.save` is set; a full-model checkpoint with optimizer state is about 460 GB.

The recipe sets `train.cuda_memory_fraction=0.9`. PyTorch's caching allocator otherwise grows to nearly all device
memory on these variable-length batches, and NCCL then fails to create the communicator it builds lazily at the end of
the first step (`NCCL WARN Cuda failure 2 'out of memory'`).

## Compare the two runs

Keep each run's log. The first steps include one-time compilation, so they are excluded. With the mock dataset's
settings, `compare_runs.py` also reports runtime tokens per second and compares by it:

```bash
uv run python examples/agentic_sft/compare_runs.py --baseline static.log --candidate dynamic.log --warmup 10 \
    --mock-seq-length 65536 --train-iters 200 --global-batch-size 32
```

Online and offline runs process different samples per step, so give each run its batch:

```bash
uv run python examples/agentic_sft/compare_runs.py --baseline online.log --candidate offline.log --warmup 10 \
    --mock-seq-length 65536 --train-iters 200 --baseline-global-batch-size 32 \
    --candidate-global-batch-size 8 --candidate-packed-sequence-size 65536
```

Step times depend on the batch, and a batch can favor either mode, so compare over many steps (a few dozen is a limited
sample).

## Measured performance

One four-GPU GB200 node per run, NeMo 26.10 container, Megatron-Core with
[NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048), the two commands above (200 steps, global
batch 32, 6,400 mock samples), steps 1–10 excluded as compilation warmup:

| Mode | Mean s/step | Median s/step | Runtime tokens/s | Median model TFLOP/s per GPU |
| --- | ---: | ---: | ---: | ---: |
| Static CP4 (`dp_balanced`) | 55.98 | 54.24 | 12,157 | 96 |
| Dynamic CP | 52.07 | 49.64 | 13,069 | 104 |

**Dynamic CP is 7.5% faster in throughput** (95% bootstrap interval +4.8% to +10.3%) and its median step is 8.5%
faster. It is faster on 161 of the 190 paired steps (mean paired difference 3.9 ± 0.7 s) and in each of five
consecutive 38-step blocks (+3.9% to +11.0%).

* Step times depend on the batch, so short runs can mislead in either direction.
* The two runs used different nodes.
* The original example reports +20.6% on its own mixed-length data (11.9K static tokens/s); its samples and order
  differ from this mock data, so the two gains are not directly comparable.

### Offline packing

Same model, samples, node type and container. The offline runs pack the samples into 1,600 rows of 64K tokens and take
eight rows per step (524K runtime tokens per step, against 682K on average for 32 online samples). Steps 1–10 are
excluded:

| Packing | Mode | Mean s/step | Runtime tokens/s | Median model TFLOP/s per GPU |
| --- | --- | ---: | ---: | ---: |
| Online, GBS 32 samples | Static CP4 | 55.98 | 12,157 | 96 |
| Online, GBS 32 samples | Dynamic CP | 51.16 | 13,301 | 106 |
| Offline, GBS 8 rows | Static CP4 | 36.27 | 14,450 | 113 |
| Offline, GBS 8 rows | Dynamic CP | 42.05 | 12,463 | 101 |

* **Offline static CP4 is the fastest setup: +18.9% over online static CP4, +8.6% over online dynamic CP.**
* On identical offline rows, dynamic CP is 13.8% slower than static CP4 (95% bootstrap interval −16.5% to −10.9%; it
  wins 8 of 190 paired steps; −9.7% after its longer warmup, steps 41–200).
* First-fit decreasing fills each row to its 64K capacity, and static CP4 keeps them as eight full CP4 bins per step.
  Dynamic CP re-bins the same sequences into about nine microbatches (simulated with Megatron-Core's scheduler: 9.05
  against 8.00; online it is the reverse, 11.3 against 13.7), with the same per-rank critical path.
* The online dynamic CP run on this branch matches the earlier one (13,301 against 13,069 tokens/s).
* The original example reports offline dynamic CP at 14,925 tokens/s against 14,342 online on its own mixed data; it
  did not measure offline static CP4.
