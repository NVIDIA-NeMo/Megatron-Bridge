# DPO Training

Direct Preference Optimization on Megatron Bridge, with reference logprobs
scored **offline** — the trainer never loads reference weights.

## Input format

Expected input: rows with two chat-format message-list fields (default field
names `chosen` / `rejected`, override with `--chosen-key` / `--rejected-key`).
Both sides must share the same context (all messages except the last) and end
with an assistant turn.

Rows that keep the shared prompt in its own field, with `chosen` / `rejected`
holding only the completion turns (TRL's *explicit-prompt conversational*
format), are read by naming that field with `--prompt-key`:

```jsonc
{"messages":  [{"role": "system", ...}, {"role": "user", ...}],
 "chosen":    [{"role": "assistant", "content": "..."}],
 "rejected":  [{"role": "assistant", "content": "..."}]}
```

Each side is assembled into a full conversation before tokenization, so the
context match holds by construction and everything downstream is unchanged.
Extra row fields (`preference_margin`, `pairs_metadata`, …) are ignored.
The scorer's `--prompt-key` and the trainer's `dataset.prompt_key` must agree;
the trainer refuses to start when it disagrees with the value recorded in the
artifact.

Any of the three fields may be a plain string instead of a message list (TRL's
*standard* format, the shape most public preference sets ship in) — the prompt
becomes one user turn and a completion one assistant turn, matching how NeMo-RL's
`BinaryPreferenceDataset` reads the same row. String and message-list fields mix
freely within a row:

```jsonc
{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
```

A pair is unusable when either side fails: `context_mismatch`,
`no_assistant_completion`, `empty_completion`, `over_length`. Pairs are never
dropped (the sampler needs a fixed dataset length) — they become zero-loss stubs.
The scorer logs one warning per stub, and training reports the live share as the
`live fraction` metric; check both before letting a run continue.

> Reading explicit-prompt rows **without** `--prompt-key` does not crash — every
> pair becomes a `no_assistant_completion` stub and training gets zero signal.
> A `live fraction` of 0 at step 0 is the tell.

## Workflow

Two steps, in order. Step 1 runs once per (dataset, tokenizer, max-seq-length,
reference model); step 2 can then sweep beta, LR, etc. without re-scoring — the
ref-logprob artifact is beta-free.

### 1. Score reference logprobs (offline, one-time)

```bash
python scripts/dpo/score_reference_logprobs.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset argilla/distilabel-capybara-dpo-7k-binarized \
    --output msc://profile/bucket/capybara_ref_logprobs --num-pairs 1024
```

Forwards the reference model (the checkpoint the policy will initialize from)
over the same dataset and writes the artifact to `--output` — a local
directory or an `msc://` URL, written in place:

- `ref_logprobs.jsonl` — one record per pair, keyed by `pair_id` (= source row
  index): completion logprob sums + token counts for both sides.
- `scoring_metadata.json` — the inputs this artifact is only valid for
  (see invariants below). Training refuses artifacts that don't match.

The scorer applies the trainer's default `bf16_mixed` precision recipe and disables
any MTP head, and records the recipe name; `--tp` and
`--sequence-parallel` are recorded and must match training (MoE models at
TP > 1 need `--sequence-parallel` — MCore refuses MoE + TP without it in
train mode).

For models that don't fit on one GPU, score with tensor parallelism — launch
exactly `--tp` processes; every rank computes identical sums (vocab-parallel
CE all-reduces across the TP group) and rank 0 writes the artifact:

```bash
python -m torch.distributed.run --nproc_per_node=2 \
    scripts/dpo/score_reference_logprobs.py --tp 2 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
    --output /data/uf_ref_logprobs
```

Training must then run with the same `--tp` (recorded in the artifact's
metadata and checked at startup): TP changes the vocab-parallel CE reduction
order, so a mismatch would shift step 0 off ln 2 and hide a real mismatch
behind it. Pipeline parallelism is free to differ: the scorer always runs at
PP=1 and its artifact is valid for any training PP, because PP only decides
which rank runs a layer. Training with PP > 1 turns on
`model.variable_seq_lengths` automatically, since pair batches are padded to
their own longest row and the stages must exchange shapes.

MoE models can additionally shard experts with `--ep` (experts split *between*
ranks) and `--etp` (each expert's GEMMs split *within* a rank group). `--etp`
defaults to `--tp`, so dense models need neither flag. The expert mesh must
also cover the world exactly, so `--ep > 1` requires `--etp 1`:

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    scripts/dpo/score_reference_logprobs.py --tp 4 --ep 4 --etp 1 \
    --model <moe-model> \
    --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
    --output /data/uf_ref_logprobs
```

`(EP, ETP)` is recorded only when it differs from the implied `EP=1 / ETP=TP`.
Unlike TP, an expert-layout mismatch against the trainer warns rather than
fails: it perturbs expert-GEMM numerics without mis-anchoring margins.

By default batches are a fixed pair count (`--micro-batch-size`), padded to
the longest pair in the batch — at long `--max-seq-length` a batch of long
pairs can OOM. `--token-budget N` replaces it with length-sorted batches
capped at `N` *padded* tokens (`2 rows × pairs × batch-max length`): peak
memory is bounded by construction, short pairs pack densely instead of being
padded to a long neighbor (typically several times faster). Batching perturbs
the scored values on MoE models by a few nats per pair (the router amplifies
batch-shape numerics; dense models are insensitive) — measured to have no
training effect, so pick whatever fits the hardware. `--micro-batch-size 1`
only matters if you want step 0 to read ln 2 to the third decimal:

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    scripts/dpo/score_reference_logprobs.py --tp 4 --token-budget 16384 \
    --model <moe-model> --max-seq-length 4096 \
    --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
    --output /data/uf_ref_logprobs
```

### 2. Train

Training runs through the shared recipe launcher with `--mode dpo`
(see `scripts/training/README.md` for the full launcher docs). `--mode dpo`
selects the model's `<model>_dpo_config` library recipe; point the run at the
scored source and artifact with trailing overrides, and at the reference
weights with a local HF model directory:

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --model qwen3_30b_a3b --mode dpo \
    --pretrained_checkpoint /checkpoints/qwen3-30b-a3b \
    dataset.source.path_or_dataset=argilla/distilabel-capybara-dpo-7k-binarized \
    dataset.source.split=train \
    dataset.ref_artifact=msc://profile/bucket/capybara_ref_logprobs \
    dataset.num_pairs=1024
```

The launcher routes `--mode dpo` through `dpo_train()`
(`megatron.bridge.training.dpo_train`), the API entry for programmatic
callers. `dpo_train` fail-fast validates the run config and the artifact
metadata, then delegates to the stock `finetune` loop with the `dpo_step`
forward step from the shared registry.

Multi-GPU: `--nproc_per_node = tp × pp × dp`. Pass `-tp` matching the scoring
run; PP is free, and the remaining ranks form the data-parallel group. Recipes carry their
model's recompute settings; override via `model.recompute_granularity` if the
measured headroom allows.

### Multinode

`world_size = nnodes × nproc_per_node = tp × pp × dp`. DP > 1 is what a second
node buys: the distributed optimizer (already enabled) shards the fp32 Adam
state — the largest per-GPU resident — across the DP replicas. DP is
parity-neutral: it only deals different *pairs* to different replicas, so
existing ref artifacts stay valid.

Launch with any standard torchrun rendezvous (same command on every node,
varying `--node_rank`). Every node needs the same code tree, the run-critical
env (`CUDA_DEVICE_MAX_CONNECTIONS=1`, the allocator conf, any model-specific
vars), the HF cache/token, the pretrained-checkpoint
directory, and the ref artifacts readable at the same path. Batch math:
`train.global_batch_size % (train.micro_batch_size × dp) == 0` — checked at
startup.

**World size and step 0:** step 0 reads exactly `ln 2` only when the
trainer's world size equals the scoring run's — NCCL reduction order shifts
with topology, and MoE routers amplify it into a fixed, reproducible offset
(measured on gemma-4 26B: world 8 → 0.6931; world 16, same artifact →
0.8507). Same noise class as batching, same verdict: harmless. For multinode
runs against a single-node-scored artifact, the step-0 health signal is a
*stable* opening value, not ln 2 itself.

### Optional: validation

Validation runs the same DPO metrics (`dpo loss`, `margin`, `accuracy`,
`rewards chosen/rejected`, `live fraction`) over a held-out split at every
`eval_interval` steps. The validation split needs its **own** scored artifact —
repeat step 1 over the validation rows (same model, tokenizer, and
`--max-seq-length` as the train scoring run).

Enable it by setting `dataset.validation_source` (an `HFDatasetSourceConfig`
or a `JSONLSourceConfig`) plus `dataset.validation_ref_artifact` — in the
model's DPO recipe or from a programmatic caller — and turn it on with
`validation.eval_interval` / `validation.eval_iters`. `eval_global_batch_size` /
`eval_micro_batch_size` default to the train sizes and are row-denominated
(even) like everything else. All invariants below apply to the validation
artifact too, checked against the validation source. Semantics match the SFT
dataset configs: eval settings without a validation split are rejected at
startup, while a configured validation split with `eval_iters=0` is simply not
built — validation toggles via `validation.eval_iters` alone.

**Step-0 sanity check (read the log):** at step 0 the policy equals the
reference, so every pair's margin should be 0 and the first logged iteration
shows `preference loss ≈ ln 2 ≈ 0.6931`, `margin ≈ 0`, `rewards ≈ 0`. Kernel
and layout noise moves these by hundredths; a wrong artifact, wrong pretrained
checkpoint, or tokenization mismatch moves them by whole units. There is no
automatic gate on this — the same posture as TRL's precomputed reference
logprobs — so look at iteration 1 before letting a long run continue.

## Invariants: scorer run == training run

The margins are only meaningful if both jobs compute logprobs over the exact
same token streams. These must match between the scoring run and the training run:

| Invariant | Enforced by |
|---|---|
| Dataset source + split | `scoring_metadata.json` check at startup |
| Tokenizer | `scoring_metadata.json` check at startup |
| Sequence length (`dataset.seq_length`, stored as `max_seq_length` in the artifact) | `scoring_metadata.json` check at startup |
| Row layout (`dataset.prompt_key` / `--prompt-key`) | `scoring_metadata.json` check at startup |
| TP size and sequence parallelism | `scoring_metadata.json` check at startup (PP may differ) |
| `num_pairs` (same rows, same order) | artifact coverage check: ref logprobs must cover `pair_id 0..N-1` exactly |
| Reference model == `pretrained_checkpoint` | not checked — visible as step-0 `preference loss` ≠ ln 2 in the log |
| No MTP head | `validate_dpo_run_config` |
| bf16 forward | not checked; the scorer pins it and the recipes default to `bf16_mixed` |

A mismatch in any of these silently mis-anchors every margin, which is why
each one is either rejected at startup or worth a glance at the step-0 log.

## Batch sizes are rows, not pairs

`micro_batch_size` / `global_batch_size` / consumed-sample counters are
denominated in **rows** everywhere (config, train state, logs), and one pair
is two rows — chosen at even row indices, rejected at odd. Both batch sizes
must therefore be even. The row→pair halving happens in exactly one place
(`build_preference_data_loader`) so the shared training loop, LR scheduler
increments, and checkpoint resume accounting all stay row-denominated and
unmodified.


## Verification tools

- Unit tests: `tests/unit_tests/training/test_dpo*.py`,
  `tests/unit_tests/data/datasets/test_preference*.py` (includes a NeMo-RL
  reference-implementation parity test for the loss).
