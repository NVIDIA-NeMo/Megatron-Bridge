# Qwen3.5-35B-A3B with dynamic context parallelism

`qwen35_text_35b_a3b_4gpu_gb200_fp8mx_dynamic_cp_config` trains the **full 40-layer Qwen3.5-35B-A3B (plus one MTP
layer)** on one four-GPU GB200 node with **dynamic** context parallelism (CP), on **mock data** whose sample lengths
follow an agentic tool-use workload. Dynamic CP runs each packed bin on a CP1, CP2 or CP4 group sized for its longest
sequence instead of splitting every bin across the full CP4 group. This directory shows how to run the recipe in both
modes and compare them.

The same model also trains on real agentic data:
[CoderForge at a 128K context](#coderforge-at-a-128k-context-on-gb300) fine-tunes it on CoderForge-Preview
trajectories on eight or more GB300 GPUs, online or from offline packed rows.

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

These numbers predate Qwen3.5's move to Megatron-Core's hybrid model. On the current model, one four-GPU GB300
node and the same container, the two commands give 14,903 runtime tokens/s for static CP4 and 15,733 for dynamic CP
over steps 11–200 (+5.6%), and 15,244 against 17,118 over steps 41–200 (+12.3%): dynamic CP's first 40 steps are
slower there, and it is ahead afterwards.

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

## CoderForge at a 128K context on GB300

`qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config` fine-tunes the same full model on the
[CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) SWE-Rebench trajectories at a
131,072-token context on eight GB300 GPUs (two four-GPU nodes in one NVL72 rack): TP1, PP1, base CP8, EP8, 16,384
tokens per rank, 16 trajectories per step, and otherwise the settings of the mock-data recipe (MXFP8, FP32 Adam state
on the host, HybridEP, full recompute). `qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config`
reads the same trajectories packed offline into 131,072-token rows, eight rows per step. Both read the dataset as the
other CoderForge recipes do, through `default_coderforge_config`, and truncate a longer trajectory to the context.

### Requirements

* Eight GB300 GPUs in one NVLink domain, the container and Megatron-Core of the mock-data recipe, and host memory for
  the CPU optimizer.
* A Megatron checkpoint of the text model. Extract the text weights and import them as described in
  [the Qwen3.5 example](../models/qwen/qwen35_vl/README.md#prepare-a-text-only-checkpoint-with-pretrained-mtp).
* `HF_HOME` and `NEMO_HOME` on storage every node can read: rank 0 writes the dataset under `NEMO_HOME`.

### Prepare the data

`item_tokens.py` prepares a recipe's data the way rank 0 of a training run would (download, JSONL and, for the offline
recipe, tokenization and packing), and writes the token count of every training item for `compare_runs.py`. Run it
once per recipe before training; tokenizing a trajectory takes seconds, so use many workers:

```bash
uv run python examples/agentic_sft/item_tokens.py \
    --recipe qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config \
    --output offline_items.json --num-workers 96 "dataset.hf_dataset.split='SWE_Rebench[:8000]'"
uv run python examples/agentic_sft/item_tokens.py \
    --recipe qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config \
    --output online_items.json --num-workers 96 "dataset.hf_dataset.split='SWE_Rebench[:8000]'"
```

The split override keeps the first 8,000 of the 77,169 trajectories, which the measurements below use; drop it to
train on all of them. Of these 8,000 trajectories, 12% fit a CP2 group (up to 32K tokens), 66% need CP4 (32K–64K) and
22% need CP8 (64K–128K); almost none fits one GPU. They hold 420M tokens, 52K per trajectory, and pack into 3,413 rows
of 123K tokens on average.

### Run

One process per GPU, on two nodes with Slurm:

```bash
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \
    uv run python scripts/training/run_recipe.py \
    --recipe qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config --mode sft \
    checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b-text \
    "dataset.hf_dataset.split='SWE_Rebench[:8000]'" \
    train.train_iters=100 scheduler.lr_decay_iters=100
```

* Static-CP8 baseline on the same packing path and data: add `model.dynamic_context_parallel=false`.
* Online packing: use `qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_dynamic_cp_config`.
* More GPUs: the extra ranks become data-parallel replicas. Scale the batch with them, for example
  `train.global_batch_size=16` rows on 16 GPUs. With dynamic CP they also join the pool a bin's group is drawn from,
  and Megatron-Core's Gated DeltaNet splits its 16 key heads across that group, so dynamic CP runs this model on at
  most 16 GPUs per pool. Configuration validation rejects a larger pool; static CP has no such limit.

### Compare runs

```bash
uv run python examples/agentic_sft/compare_runs.py --baseline static.log --candidate dynamic.log --warmup 40 \
    --item-tokens offline_items.json --global-batch-size 8
```

For an online run against an offline run, give each its token counts and batch, as in the mock-data comparison above.

### Measured performance

GB300 nodes (four GPUs of 284 GB each), NeMo 26.10.00 container with the `transformers` version Bridge locks,
Megatron-Core at Bridge's pin, the pretrained text checkpoint of `Qwen/Qwen3.5-35B-A3B`, and the first 8,000
SWE-Rebench trajectories. Each context-parallel and expert-parallel group of eight ranks runs on two nodes of one NVL72
rack; at 16 and 32 GPUs the data-parallel replicas are in different racks. Every run takes 100 steps at 8 rows or 16
trajectories per 8 GPUs. Throughput is runtime tokens per second from `compare_runs.py`; memory is the peak device
memory in use on any GPU.

Offline packed rows (about 1.0M tokens per step on 8 GPUs):

| GPUs | Mode | Tokens/s, steps 11–100 | Tokens/s, steps 41–100 | Median s/step | Median model TFLOP/s per GPU | Peak GiB per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | Static CP8 | 35,099 | 36,190 | 25.6 | 182 | 118 |
| 8 | Dynamic CP | 25,432 | 32,448 | 25.9 | 177 | 123 |
| 16 | Static CP8 | 66,019 | 70,858 | 24.7 | 188 | 120 |
| 32 | Static CP8 | 115,393 | 125,425 | 24.5 | 187 | 120 |

Online packing (about 0.8M tokens per step on 8 GPUs):

| GPUs | Mode | Tokens/s, steps 11–100 | Tokens/s, steps 41–100 | Median s/step | Median model TFLOP/s per GPU | Peak GiB per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | Static CP8 | 17,252 | 20,246 | 38.3 | 106 | 118 |
| 8 | Dynamic CP | 14,556 | 16,331 | 38.9 | 98 | 123 |

* **Dynamic CP trains the same model.** From the same checkpoint and on identical rows, the step-1 loss is 0.18854
  with static CP8 and 0.18860 with dynamic CP, and 0.14035 against 0.14028 at step 100. Online, the two modes give
  0.18065 and 0.18071 at step 1 and 0.14700 at step 100.
* **Offline rows with static CP8 are the fastest setup on this data, and scale with GPUs.** Per GPU, steps 41–100 run
  at 4,524 tokens/s on 8 GPUs, 4,429 on 16 and 3,920 on 32, with the median step flat at about 25 s. A 300-step run on
  another pair of nodes reached 40,641 tokens/s over steps 41–300 (median 23.3 s), so differences of about 10% between
  runs are within node-to-node variation.
* **On these trajectories dynamic CP has little to gain.** Almost none of them fits one GPU, and the scheduler puts
  47% of the offline tokens on CP4 groups and 53% on CP8 groups, in 9.1 microbatches per step against 8.0 for static
  CP8. After warm-up a dynamic-CP step takes as long as a static one (median 25.9 s against 25.6 s); over steps 41–100
  it is 10.3% behind in throughput (95% bootstrap interval −19.0% to −0.6%), and 27.5% behind over steps 11–100.
  Expect a gain only on streams with a meaningful share of short sequences, as on the mock data above.
* **Warm-up is kernel compilation per microbatch shape.** The first microbatch a rank runs with a token count it has
  not seen takes about 20 s longer: about 6 s in its first MoE layer's forward pass and 12 s in backward, while the
  other ranks wait. Offline rows under static CP8 are nearly full, so a rank sees few distinct token counts (79 in 623
  microbatches, 21 of them slow, 15 among the first 90). Dynamic CP and online packing produce bins of many more
  sizes: with dynamic CP on offline rows, 49 of the first 90 microbatches were slow, 22 of the next 90, and at most 4
  of every 90 after step 30.
* **Online packing is slower here for the same reason.** Its bins hold whatever a step's trajectories add up to, so new
  token counts keep appearing through step 100: static CP8 reaches 20,246 tokens/s over steps 41–100 and dynamic CP
  16,331 (19.3% behind, interval −30.2% to −7.2%), with equal median steps (38.3 s and 38.9 s).
* **Memory.** Device memory in use peaks at 118 GiB per GPU with static CP8 and 123 GiB with dynamic CP.
* **Dynamic CP is limited to 16 GPUs per pool for this model**, as described under Run, so only static CP8 was run on
  32 GPUs.
