# DPO Training

DPO on Megatron Bridge. The reference model is scored once, offline, and the
trainer reads the scores back from disk, so it only ever holds the policy.

## Input format

Each row has a `chosen` and a `rejected` conversation (`dataset.chosen_key` /
`dataset.rejected_key`): chat-format message lists that share everything except
the final assistant turn. Rows come from a Hub dataset, or from local or
`msc://` files through the `json` loader:

```bash
dataset.source.path_or_dataset=json dataset.source.split=train \
+dataset.source.load_kwargs.data_files=/data/pairs.jsonl
```

If the prompt lives in its own field and `chosen` / `rejected` hold only the
completion (TRL's explicit-prompt format), name it with `dataset.prompt_key`.
Completions may be message lists or plain strings:

```jsonc
{"messages": [{"role": "user", "content": "What is 2+2?"}], "chosen": "4", "rejected": "5"}
```

Pairs may also split at an earlier assistant turn (agentic trajectories): every
assistant turn after the split is scored, tool turns never are, and a pair whose
sides first differ at a user or tool turn is unusable. Name the tool-schema field
with `dataset.tools_key` so the chat template renders it. Thinking templates
(Nemotron 3, Qwen3) drop earlier reasoning by default; set
`dataset.chat_template_kwargs='{truncate_history_thinking: false}'` to keep it.

Unusable pairs (`context_mismatch`, `no_assistant_completion`,
`empty_completion`, `over_length`) become zero-loss stubs so the dataset length
stays fixed. The scorer warns per stub and the trainer logs the usable share as
`live fraction`; 0 at step 0 usually means a forgotten `dataset.prompt_key`.

## Workflow

### 1. Score the reference

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/dpo/score_reference_logprobs.py \
    --recipe qwen3_30b_a3b_dpo_config \
    --output /data/uf_ref_logprobs \
    dataset.num_pairs=8192
```

The scorer takes the training recipe plus the same trailing `KEY=VALUE`
overrides, so dataset, tokenizer, sequence length, precision and parallel layout
cannot drift between the two jobs. It writes `ref_logprobs.jsonl` (one record
per `pair_id`, the row index) and `scoring_metadata.json` to `--output`, a local
directory or an `msc://` URL. The sums are raw, so one artifact serves every
beta and learning-rate sweep.

Reference weights come in HF format from `--model`, else from
`checkpoint.pretrained_checkpoint` when that is an HF directory, else from the
recipe's HF model.

TP and sequence parallelism change the logprobs and are checked at training;
EP and ETP only warn on mismatch. The scorer always runs at PP=1 and the
artifact is valid for any training PP and DP, so override the layout freely
when scoring on fewer GPUs, e.g. `model.expert_model_parallel_size=8`.

`--micro-batch-size` pairs per batch (default 8), padded to the longest pair.
`--token-budget N` switches to length-sorted batches capped at `N` padded tokens
when long pairs OOM.

### 2. Train

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --model qwen3_30b_a3b --mode dpo \
    --pretrained_checkpoint /checkpoints/qwen3-30b-a3b \
    dataset.num_pairs=8192 \
    dataset.ref_artifact=/data/uf_ref_logprobs
```

Same dataset overrides as the scorer, plus `ref_artifact`. `--mode dpo` picks
the model's `<model>_dpo_config` recipe; the launcher is documented in
`scripts/training/README.md`.

At step 0 the policy is the reference, so `preference loss` should read about
ln 2 (0.693) with `margin` and `rewards` near 0. Whole-unit deviations mean a
wrong artifact, checkpoint or tokenizer. Nothing checks this automatically.

### Validation

Set `dataset.validation_source` (and `dataset.validation_num_pairs`), score it
with `--split validation` and a second `--output`, point
`dataset.validation_ref_artifact` at that, and turn evaluation on with
`validation.eval_interval` and `validation.eval_iters`.

## What has to match between scoring and training

Recorded in `scoring_metadata.json` and checked at startup: dataset source and
split, tokenizer, `dataset.seq_length`, `dataset.prompt_key`,
`dataset.tools_key`, `dataset.chat_template_kwargs`, TP and sequence
parallelism, and the tokenization version. The artifact must cover exactly
`pair_id 0..N-1` for the rows the trainer loads. Not checked: that the reference
model is the policy's `pretrained_checkpoint`; that shows up as a step-0 loss
far from ln 2.

## Batch sizes are rows, not pairs

`micro_batch_size`, `global_batch_size` and the consumed-sample counters count
rows. A pair is two rows, chosen at the even index and rejected right after it,
so both batch sizes must be even.
