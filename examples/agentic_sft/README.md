# Qwen3.5-35B-A3B with dynamic context parallelism

`qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config` fine-tunes the **full 40-layer
Qwen3.5-35B-A3B (plus one MTP layer)** on [CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
SWE-Rebench trajectories at a 131,072-token context on 16 GPUs with **dynamic** context parallelism (CP): each
step, the global-batch scheduler re-bins the sequences of the step's packed rows and runs every bin on a group of
ranks sized for it (one GPU, CP2, CP4, CP8 or, rarely, the whole 16-rank pool), instead of splitting every row across
the full CP8 group.

The recipe is the static long-context recipe `qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config` with
global-batch packing and dynamic CP switched on. Everything else is taken from it: TP1, base CP8, EP16, 16,384 tokens
per rank, the offline packed CoderForge rows (32 rows per step, 5% held out for validation), MXFP8 compute, selective
recompute of the Gated DeltaNet output norm and the MoE layers, NCCL EP, the optimizer and the schedule. Three settings
differ, each because dynamic CP needs it:

* **Every bin is padded to its group's capacity** (`model.pad_packed_seq_alignment="max"`), so a rank always runs
  16,384 tokens, as it does on the static recipe's fixed-width rows. See [Fixed-width bins](#fixed-width-bins).
* **Evaluation takes two global batches instead of eight** (`validation.eval_iters=2`). Global-batch packing draws the
  rows of every evaluation of a run from the validation split without cycling, and 11 evaluations of 8 x 32 rows do
  not fit the 5% split.
* **The Gated DeltaNet layers run without the fused pre-delta-rule kernels**
  (`model.gdn_pre_gated_delta_rule_fusion=false`). A bin can run on the whole 16-rank pool, where those kernels fail
  (the channels a rank holds must be a multiple of 8; with one key head and two value heads per rank they are not).
  The unfused path is a few percent slower.

This directory has the commands to run it and compare it with static CP, and a launcher that runs the same recipe on
mock mixed-length data, where dynamic CP has short sequences to place on small groups.

## Requirements

* 16 GPUs in one NVLink domain (four nodes of one NVL72 rack): the expert-parallel group spans all 16 ranks. The
  measurements below are from GB300 GPUs. The recipe has not been run on GB200 hardware; device memory in use peaked
  at 153 GiB per GPU on CoderForge and 158 GiB on the mock data.
* The NeMo container, with Transformer Engine built with NCCL EP, and Megatron-Core at Bridge's pin.
* A Megatron checkpoint of the text model. Extract the text weights and import them as described in
  [the Qwen3.5 example](../models/qwen/qwen35_vl/README.md#prepare-a-text-only-checkpoint-with-pretrained-mtp).
* `HF_HOME` and `NEMO_HOME` on storage every node can read: rank 0 writes the dataset under `NEMO_HOME`.

Megatron-Core splits Gated DeltaNet's heads across a bin's CP group, and the largest group is the whole `dp x cp`
pool. Qwen3.5-35B-A3B has 16 key heads, so dynamic CP runs it on at most 16 GPUs per pool; configuration validation
rejects a larger pool. Static CP has no such limit.

## Prepare the data

`item_tokens.py` prepares a recipe's data the way rank 0 of a training run would (download, JSONL, tokenization and
packing), and writes the token count of every training row for `compare_runs.py`. Run it once before training;
tokenizing a trajectory takes seconds, so use many workers:

```bash
uv run python examples/agentic_sft/item_tokens.py \
    --recipe qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config \
    --output items.json --num-workers 96 "dataset.hf_dataset.split='SWE_Rebench[:8000]'"
```

The split override keeps the first 8,000 of the 77,169 trajectories, which the measurements below use; drop it to
train on all of them. Of these 8,000, 7,600 are training trajectories: 12% fit a CP2 group (up to 32K tokens), 66%
need CP4 (32K–64K) and 22% need CP8 (64K–128K); almost none fits one GPU. They hold 399M tokens, 52K per trajectory,
and pack into 3,245 rows of 123K tokens on average.

## Run

One process per GPU, on four nodes with Slurm. NCCL EP maps its buffers into every GPU a process can see, so expose
one GPU to each rank:

```bash
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
    CUDA_VISIBLE_DEVICES=$SLURM_LOCALID uv run python scripts/training/run_recipe.py \
    --recipe qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config --mode sft \
    checkpoint.pretrained_checkpoint=/path/to/megatron/qwen3.5-35b-a3b-text'
```

* Static CP8 on the same packing path and data: add `model.dynamic_context_parallel=false`.
* The runs measured below add `"dataset.hf_dataset.split='SWE_Rebench[:8000]'" train.train_iters=100
  validation.eval_iters=1 logger.log_interval=1 checkpoint.save=null`: the first 8,000 trajectories, whose
  validation split holds 172 rows (enough for evaluations of one global batch), every step logged, nothing saved.
* Without `checkpoint.save=null` the recipe saves a full checkpoint with optimizer state under `nemo_experiments/`
  when training ends; point `checkpoint.save` at storage with room for it.

## Compare runs

Keep each run's log, with `logger.log_interval=1` so that every step is logged. The first steps include one-time
compilation, so they are excluded:

```bash
uv run python examples/agentic_sft/compare_runs.py --baseline static.log --candidate dynamic.log --warmup 10 \
    --item-tokens items.json --global-batch-size 32
```

With the token counts, the script reports runtime tokens per second and compares by it, so runs that process
different work per step can be compared too; give each run its own `--baseline-item-tokens` /
`--candidate-item-tokens` and batch size.

## Measured performance on CoderForge

Four GB300 nodes of one NVL72 rack (16 GPUs of 284 GB each), NeMo 26.10.01 container with the `transformers`
version Bridge locks, Megatron-Core at Bridge's pin, one GPU visible per rank, the pretrained text checkpoint of
`Qwen/Qwen3.5-35B-A3B`, and the first 8,000 SWE-Rebench trajectories: 100 steps of 32 rows (3.9M tokens per step),
with the overrides listed under [Run](#run) and no checkpoint saved. Throughput is runtime tokens per second from
`compare_runs.py` over steps 11–100; the median step and model TFLOP/s are over steps 41–100; memory is the peak
device memory in use on any GPU.

| Recipe | Mode | Tokens/s | Median s/step | Median model TFLOP/s per GPU | Peak GiB per GPU |
| --- | --- | ---: | ---: | ---: | ---: |
| Static long-context recipe, unchanged | Static CP8 on fixed-width rows | 146,651 | 26.6 | 358 | 135 |
| This recipe with `model.dynamic_context_parallel=false` | Static CP8, global-batch packing | 152,732 | 25.9 | 355 | 140 |
| This recipe | Dynamic CP | 145,985 | 26.7 | 345 | 153 |

| Recipe | Train loss at step 1 / 50 / 100 | Validation loss at step 50 / 100 |
| --- | --- | --- |
| Static long-context recipe, unchanged | 0.19829 / 0.14810 / 0.13699 | 0.14302 / 0.13259 |
| This recipe with `model.dynamic_context_parallel=false` | 0.19822 / 0.15051 / 0.13327 | 0.14388 / 0.13190 |
| This recipe | 0.19820 / 0.15049 / 0.13341 | 0.14396 / 0.13189 |

* **The recipe trains.** From the pretrained checkpoint the training loss falls from 0.198 to 0.133 in 100 steps and
  the validation loss from 0.144 at step 50 to 0.132 at step 100, as with the static recipe. The learning rate is
  still warming up over these steps (200 warm-up steps). No run skipped an iteration or produced a NaN.
* **Dynamic CP trains the same model as static CP8.** On identical rows the training losses of the two modes differ
  by 0.00007 on average over the 100 steps and by 0.0002 at most, and the validation losses by 0.0001. The unchanged
  static recipe reads the rows in another order (its `batch` loader), so its losses are comparable only as a curve.
* **On these trajectories dynamic CP is 4.4% behind static CP8** on the same packing path (95% paired bootstrap
  interval −8.4% to −1.2%), and level with the unchanged static recipe (−0.5%, interval −4.8% to +3.1%). Almost
  none of them fits one GPU, so the scheduler has nothing to move to small groups: on these rows Megatron-Core's
  scheduler forms 17.7 microbatches per step with dynamic CP against 16 with static CP8. Expect a gain on data with a
  meaningful share of short sequences; see [Mock mixed-length data](#mock-mixed-length-data).
* **Static CP8 through global-batch packing runs like the unchanged static recipe**, slightly ahead: by 4.1% in
  these runs and by 2.4% in a second pair of 30-step runs on other nodes (median step 25.5 s against 26.2 s). Stack
  samples of that pair show the same split of time between forward, backward and communication.
* **Dynamic CP's first microbatch on a new group size takes about a minute** (communicators and kernels): steps 8, 13
  and 56 took 55–69 s here. Its peak device memory is 153 GiB against 140 GiB.
* With the fused pre-delta-rule kernels switched back on, static CP8 on the packing path reached 159,377 tokens/s
  in another 100-step run (4.2% more; other nodes). Dynamic CP cannot use them on 16 GPUs, see above.

### Fixed-width bins

A packed bin holds whatever its sequences add up to, so without padding nearly every microbatch gives a rank a token
count it has not run before. Kernels on this model's training path are specialized per sequence length (Triton kernels
are autotuned for each new length, CuTe DSL kernels compiled for it), which costs seconds on one rank while the ranks
it communicates with wait. On 16 GPUs most steps then contain such a microbatch: on the mock data below (32 samples
per step), steps 9–14 of a dynamic-CP run took 40–66 s without padding, with a third of the run's first minutes
spent inside the autotuner. With every bin padded to 16,384 tokens per rank, a step takes 4–13 s once each group
size has been used.

The static recipe never meets this, because its rows are padded to the full 131,072 tokens.
`model.pad_packed_seq_alignment="max"` gives the bins of global-batch packing the same property: Megatron-Core appends
a masked dummy sequence to every bin so that each rank of its group runs exactly `max_seqlen_per_dp_cp_rank` tokens.
The padding is computed but not trained on. On the CoderForge rows it is 6% of the tokens run with static CP8 (16
microbatches per step) and 15% with dynamic CP (17.7).

## Mock mixed-length data

`mock_data.py` trains the same recipe on `MockGPTSFTDatasetConfig` samples instead of CoderForge: random token IDs
whose lengths mix short and long sequences, as an agentic tool-use workload does. Only the data changes; the model,
parallelism, precision and execution settings are the recipe's. 40% of the samples need at most a CP1 group (up to
16K tokens), 30% CP2, 20% CP4 and 10% CP8, with lengths uniform inside each bucket: 11% / 24% / 33% / 32% of the
tokens, 30K tokens per sample on average. The last 20% of each sample is supervised. Lengths and tokens are
deterministic in the seed, so two runs see identical samples in identical order. The model trains from random weights
and the loss is not meaningful: random tokens have nothing to learn. Nothing is saved.

```bash
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
    CUDA_VISIBLE_DEVICES=$SLURM_LOCALID uv run python examples/agentic_sft/mock_data.py --packing online \
    train.train_iters=120 train.global_batch_size=64 logger.log_interval=1'
```

* Static CP8 on the same samples: add `model.dynamic_context_parallel=false`.
* `--packing offline` first packs the samples, first-fit decreasing, into 131,072-token rows; a step then takes a
  global batch of rows and the scheduler re-bins their samples.
* Token counts for `compare_runs.py`: `item_tokens.py --mock-packing online --num-items 7680 --output mock_items.json`
  with the same recipe (`--num-items` is `train.train_iters x train.global_batch_size`).

### Measured performance

Four GB300 nodes of one NVL72 rack (16 GPUs of 284 GB each), NeMo 26.10.01 container, Megatron-Core at Bridge's pin,
one GPU visible per rank. Throughput is runtime tokens per second from `compare_runs.py` over the steps from 11 on;
the median step and model TFLOP/s are over the steps from 41 on; memory is the peak device memory in use on any GPU.

| Packing | Mode | Steps | Tokens/s | Median s/step | Median model TFLOP/s per GPU | Peak GiB per GPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Online, 64 samples per step (1.9M tokens) | Static CP8 | 120 | 119,375 | 16.1 | 275 | 142 |
| Online, 64 samples per step (1.9M tokens) | Dynamic CP | 120 | 147,069 | 12.2 | 368 | 156 |
| Offline, 32 rows per step (4.2M tokens) | Static CP8 | 60 | 164,872 | 24.9 | 392 | 149 |
| Offline, 32 rows per step (4.2M tokens) | Dynamic CP | 60 | 140,111 | 30.3 | 322 | 158 |

* **Online, dynamic CP is 23.2% faster** on identical samples (95% paired bootstrap interval +17.1% to +29.1%) and
  27.2% faster over steps 41–120 (+21.9% to +32.5%); it is faster on 94 of the 110 steps. With every bin padded to
  16,384 tokens per rank, the cost of a step follows its number of microbatches, and Megatron-Core's scheduler needs
  8.1 per step with dynamic CP against 9.5 with static CP8.
* **On full offline rows, dynamic CP is 15.0% behind static CP8** (−21.0% to −9.4%). First-fit decreasing fills
  each row to nearly its 131,072-token capacity, so static CP8 runs a step as 16 full microbatches, while dynamic CP
  re-bins the same sequences into 17.2. Offline packing with static CP8 is also the fastest of the four setups.
* Dynamic CP's first steps are slower: the first microbatch on each group size (CP1, CP2, CP4, CP8 and the whole
  16-rank pool) sets up communicators and kernels, about a minute each time. Steps 1–10 are excluded above.
