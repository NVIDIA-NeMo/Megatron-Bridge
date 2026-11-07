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

One four-GPU GB200 node per run, NeMo 26.10 container, Megatron-Core at Bridge's pin (which includes
[NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048)), the two commands above (200 steps, global
batch 32, 6,400 mock samples), steps 1–10 excluded as compilation warmup:

| Mode | Mean s/step | Median s/step | Runtime tokens/s | Median model TFLOP/s per GPU |
| --- | ---: | ---: | ---: | ---: |
| Static CP4 (`dp_balanced`) | 55.43 | 53.87 | 12,253 | 95 |
| Dynamic CP | 50.07 | 47.56 | 13,564 | 109 |

**Dynamic CP is 10.7% faster in throughput** (95% bootstrap interval +7.5% to +14.0%) and its median step is 13.3%
faster. It is faster on 158 of the 190 paired steps (mean paired difference 5.4 ± 0.8 s) and in each of five
consecutive 38-step blocks (+3.9% to +13.7%; the first block includes dynamic CP's longer warmup, and steps 41–200
give +12.5%).

* Step times depend on the batch, so short runs can mislead in either direction.
* The two runs used different nodes.
* The original example reports +20.6% on its own mixed-length data (11.9K static tokens/s); its samples and order
  differ from this mock data, so the two gains are not directly comparable.

### Offline packing

Same model, mock data, node type, container and Megatron-Core. The offline runs pack the samples first-fit decreasing,
in chunks of 4,096 samples, into rows of 64K tokens and take eight rows per step (524K runtime tokens per step, against
679K on average for 32 online samples). Steps 1–10 are excluded:

| Packing | Mode | Mean s/step | Runtime tokens/s | Median model TFLOP/s per GPU |
| --- | --- | ---: | ---: | ---: |
| Online, GBS 32 samples | Static CP4 | 55.43 | 12,253 | 95 |
| Online, GBS 32 samples | Dynamic CP | 50.07 | 13,564 | 109 |
| Offline, GBS 8 rows | Static CP4 | 39.49 | 13,271 | 103 |
| Offline, GBS 8 rows | Dynamic CP | 41.80 | 12,536 | 104 |

* **Online dynamic CP is the fastest setup here.** Offline packing speeds up static CP4 by 8.3% (95% bootstrap
  interval +5.3% to +11.3%), but offline static CP4 stays 2.2% behind online dynamic CP.
* On identical offline rows, dynamic CP brings no gain over static CP4. Over steps 11–200 it is 5.5% slower (95%
  bootstrap interval −8.5% to −2.4%), almost all of it in its longer warmup: by consecutive 38-step blocks it is
  −22.6%, +0.7%, −2.4%, +1.9% and −0.9%. Over steps 41–200 it is 1.2% slower (−3.9% to +1.6%), with a 1.9% faster
  median step, and it is faster on 92 of those 160 paired steps.
* First-fit decreasing fills each row to nearly its 64K capacity, and static CP4 runs each row as one balanced CP4
  bin. Dynamic CP re-bins a step's sequences into about one more microbatch (simulated with Megatron-Core's
  scheduler: 9.03 against 8.00; online it is the reverse, 11.33 against 13.64).
* The original example reports offline dynamic CP at 14,925 tokens/s against 14,342 online on its own mixed data; it
  did not measure offline static CP4.
