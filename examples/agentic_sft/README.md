# Qwen3.5-35B-A3B with dynamic context parallelism on 4×GB200

Trains the **full 40-layer Qwen3.5-35B-A3B (plus one MTP layer)** on a single four-GPU GB200 node with **static** or
**dynamic** context parallelism (CP), on **mock data** whose sample lengths follow an agentic tool-use workload. Dynamic
CP regroups the four ranks as CP1, CP2 or CP4 for each packed bin, instead of running every sample on the full CP4 group.

The model and training configuration follow the Megatron-LM agentic SFT example by Zhongbo Zhu
([`examples/agentic_sft`, commit `420686e`](https://github.com/zhongbozhu/Megatron-LM/commit/420686e1a7e9d51d422e8ac1a18a35818843bb86)),
ported to Megatron Bridge. Its CoderForge data pipeline is not part of this example; the data here is generated.

Run the commands from the repository root. Nothing is downloaded: the architecture is embedded in the script and the
mock data and its tokenizer are generated locally in a workspace (`agentic_sft_workspace/` by default).

## What is here

| File | Purpose |
| --- | --- |
| `train_qwen35_35b_a3b.py` | Builds the Bridge `ConfigContainer` and trains (`online_static_cp`, `online_dynamic_cp`) |
| `mock_agentic_data.py` | Generates the mock dataset and the tokenizer that reads it |
| `compare_runs.py` | Compares the steady-state step time of two training logs |

## Requirements

* Four GB200 GPUs and a container with Transformer Engine, CuTeDSL and DeepEP HybridEP.
* Large host memory: the default FP32 CPU optimizer offload keeps master weights and Adam state on the host. The
  full-model runs below completed within a 900 GB allocation; peak host usage was not measured.
* **Megatron-Core with [NVIDIA/Megatron-LM#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048)** for
  `online_dynamic_cp`. This branch pins the head commit of that pull request so the example can be tested now; once it
  is merged, the pin moves to the merged commit. See [Dynamic CP and Megatron-Core](#dynamic-cp-and-megatron-core).

## Model and modes

The model has **40 decoder layers**: ten cycles of 3 Gated DeltaNet (GDN) + 1 attention layers, plus **MTP1 (loss
coefficient 0.1)**. Hidden size 2048, 16 attention heads and 2 query groups, 256 top-8 experts (FFN 512), a gated
shared expert, vocabulary 248320. Pass `--hf-config` to build from another Hugging Face text config instead of the
embedded one, and `--num-layers` for a shallower prefix of whole cycles.

Both modes use **64K context, 16384 tokens per rank, TP1/PP1/base CP4/EP4/ETP1, micro batch 1** and per-token loss.

| Mode | Scheduler | Runtime CP | Default global batch |
| --- | --- | --- | --- |
| `online_static_cp` | `dp_balanced` | 4 | 32 samples |
| `online_dynamic_cp` | `default_dynamic_cp` | 1, 2, 4 | 32 samples |

With four ranks the base data-parallel size is 1, so static mode does not exercise multi-DP balancing.

The model is initialized from scratch by default. To start from a converted, shape-compatible Megatron checkpoint pass
`--init-mode checkpoint --pretrained-checkpoint PATH`; only weights are loaded. By default nothing is saved. Pass
`--save-dir` to write checkpoints (about 460 GB with optimizer state for the full model) and TensorBoard logs, and
`--resume` to continue a run from there.

## Mock data

Each row is a prompt and a completion of random token IDs; only the completion (20% of the tokens) is supervised. Row
lengths follow the `mixed` distribution of the original example: **50% / 30% / 20% of samples** need at most CP1, CP2
or CP4 capacity (up to 16K, 16K–32K and 32K–64K tokens), with a length drawn uniformly inside each bucket. That gives
about 19% / 35% / 46% of the tokens in the three buckets (the original example reports 19% / 36% / 45% for its real
data) and a mean of about 21K tokens per sample.

The generator writes `training.jsonl`, a `manifest.json` with every row's length, and a tiny local word-level tokenizer
in which token `n` is the string `"n"`, so a row has exactly its intended number of tokens and no tokenizer download
is needed. Generation is deterministic and runs once per node on the first launch; by default one pass of
`train-iters × global-batch-size` rows is written. To generate without training:

```bash
uv run python examples/agentic_sft/train_qwen35_35b_a3b.py online_dynamic_cp --prepare-data-only \
    --train-iters 50 --global-batch-size 32
```

## Train

```bash
uv run python -m torch.distributed.run --nproc_per_node=4 \
    examples/agentic_sft/train_qwen35_35b_a3b.py online_dynamic_cp --train-iters 50
```

`--dry-run` builds and prints the configuration without launching or generating data. Arguments the script does not
recognize are applied as dotted `ConfigContainer` overrides, for example `train.manual_gc=true`. Run `--help` for every
option.

## Compare static and dynamic CP

Run both modes over the same data (same `--data-dir`, `--train-iters` and `--global-batch-size`), keep each run's log,
and compare them. The first iterations include one-time compilation, so they are excluded:

```bash
uv run python examples/agentic_sft/compare_runs.py --baseline static.log --candidate dynamic.log --warmup 10 \
    --manifest agentic_sft_workspace/data/mock/manifest.json --global-batch-size 32
```

Rows are consumed in file order, so the manifest gives each iteration's exact token count and the tool reports runtime
tokens per second next to seconds per step.

## Performance settings

| Option | Default | Effect |
| --- | --- | --- |
| `--precision` | `mxfp8` | MXFP8 compute with the TE op fuser, grouped tensors and CuTeDSL; `bf16` disables this bundle |
| `--optimizer-cpu-offload` | on | CPU Adam with FP32 master weights, gradients and moments; BF16 parameter gather. Off keeps the optimizer on the GPU and enables FP8 parameter gather with MXFP8 |
| `--hybrid-ep` | on | HybridEP with uneven-input padding; off selects all-to-all |
| `--full-recompute` | on | Uniform full recompute; off recomputes only GDN norm and MoE |
| `--ddp-overlap` | on | Overlap gradient reduction and parameter gather |
| `--cuda-memory-fraction` | 0.9 | Caps PyTorch's allocator so NCCL can still allocate; 0 disables the cap |

PyTorch's caching allocator tends to claim nearly all device memory with variable-length batches. NCCL creates some
communicators lazily (for example the model-parallel group's, at the end of the first step), so without headroom the
full-model dynamic run fails at its first step with `NCCL WARN Cuda failure 2 'out of memory'`. The default cap avoids
that; static mode is unaffected by it (step times agree within 1–2% with and without the cap).

Both precisions use BF16 as the base precision. Per-token loss, FP32 router and gradient accumulation, native fused
cross entropy, zero dropout and an MoE auxiliary-loss coefficient of 0.001 stay on. With CPU offload the script sets
12 OpenMP threads per rank, otherwise one.

## Measured performance

One GB200 node (four GPUs) per run, NeMo 26.10 container, Megatron-Core at the head of #7048, the full 40-layer model with
MXFP8, CPU optimizer offload, HybridEP and full recompute, global batch 32. Both modes process identical samples in
identical order, so step *i* is directly comparable. Steps 1–10 are excluded because they include one-time compilation.

**200 steps on 6,400 mock rows** (136 M runtime tokens, mean 21.3K per sample; steps 11–200 measured):

| Mode | Mean s/step | Median s/step | Runtime tokens/s |
| --- | ---: | ---: | ---: |
| `online_static_cp` | 44.47 | 43.05 | 15,302 |
| `online_dynamic_cp` | 40.60 | 38.17 | 16,763 |

Dynamic CP is **+9.6%** faster in total (95% bootstrap interval +6.0% to +13.2%), with a 12.8% faster median step. It is
faster in 163 of the 190 paired steps (mean paired difference 3.9 s ± 0.7 s), and in each of five consecutive 38-step
blocks (+4.4% to +15.7%).

What to expect from this number:

* **The gain is real but smaller than the +20.6% the original example reports** for its mixed data. The original
  measured a static baseline of about 11.9K tokens/s; Bridge's static baseline here is 15.3K tokens/s, so there is
  less to recover. Dynamic CP reaches 16.8K tokens/s.
* **It depends on the batch, so measure over many steps.** Step times are deterministic per batch (rerunning a mode
  on the same data reproduces them), and a batch can favor either mode. Two earlier 50-step runs per mode on a different
  1,600-row dataset gave only +3.6% to +4.0%: dynamic was 10–11% faster over their steps 11–32 and 5% slower over steps
  33–50. A short window is a limited sample of the workload.
* Dynamic CP wins by forming fewer microbatches: offline, the two schedulers produce on average 11.3 microbatches per
  step for dynamic against 13.5 for static on that dataset, with an equal per-rank token critical path.

Reproduce with the same `--data-dir`, `--train-iters 200` and `--global-batch-size 32` for both modes, then use
`compare_runs.py` as shown above. The two modes above ran on different nodes; static step times agree across nodes
within about 1%. The loss is not meaningful here: the rows are random tokens, so there is nothing to learn.

## Dynamic CP and Megatron-Core

`online_dynamic_cp` regroups the four ranks as CP1, CP2 or CP4 per packed bin. Megatron-Core before
[#7048](https://github.com/NVIDIA/Megatron-LM/pull/7048) assumes the static CP size in its Gated DeltaNet layers, so a
bin that runs on a smaller group fails:

```text
GDN: cu_seqlens_q[-1]=16384 does not match total_sequence_length=65536
```

#7048 propagates the runtime CP group through hybrid layers and normalizes the MTP loss across it.

## Differences from the original example

* Data is generated, not CoderForge trajectories, so there is no chat template, assistant masking or length filtering.
  The original's offline packing step and `offline_dynamic_cp` mode are not included.
* The per-sequence alignment padding is derived by Bridge (`2 × data-parallel × CP`, 8 here); the original used a fixed
  32. It is folded into each sequence (`fold_alignment_padding`), which Qwen3.5's 256-wide attention heads need so
  that FlashAttention sees no padding between packed sequences.
* The GDN kernel backend is the Megatron-Core default.

## Tests

CPU-only, no GPU needed:

```bash
uv run python -m pytest tests/unit_tests/examples/test_agentic_sft_*.py
```
