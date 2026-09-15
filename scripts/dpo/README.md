# DPO Training

DPO on Megatron Bridge. The reference model is scored once, offline, and the
trainer reads the scores back from disk, so it only ever holds the policy.

## Input format

Each row has a `chosen` and a `rejected` conversation (rename the fields with
`--chosen-key` / `--rejected-key`). Both are chat-format message lists that
share everything except the final assistant turn.

If your data keeps the prompt in a separate field and `chosen` / `rejected`
only hold the completion (TRL calls this the explicit-prompt format), pass the
prompt field with `--prompt-key`:

```jsonc
{"messages":  [{"role": "system", ...}, {"role": "user", ...}],
 "chosen":    [{"role": "assistant", "content": "..."}],
 "rejected":  [{"role": "assistant", "content": "..."}]}
```

Plain strings work too (TRL's standard format). The prompt becomes a user turn
and each completion an assistant turn:

```jsonc
{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
```

Use the same `--prompt-key` when scoring and `dataset.prompt_key` when
training. The value is recorded in the artifact and the trainer refuses to start
on a mismatch. Other fields in the row are ignored.

Pairs that can't be used (`context_mismatch`, `no_assistant_completion`,
`empty_completion`, `over_length`) are not dropped, because the sampler needs a
fixed dataset length. They become stubs with zero loss. The scorer warns once
per stub and the trainer logs the share of usable pairs as `live fraction`. If
that reads 0 at step 0, you most likely have explicit-prompt rows and forgot
`--prompt-key`.

## Workflow

### 1. Score the reference

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/dpo/score_reference_logprobs.py --tp 2 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
    --output /data/uf_ref_logprobs
```

This runs the reference model (the checkpoint you will start training from)
over the dataset and writes two files to `--output`, which can be a local
directory or an `msc://` URL:

- `ref_logprobs.jsonl`, one record per pair keyed by `pair_id` (the row index
  in the source), with the completion logprob sum and token count for both sides.
- `scoring_metadata.json`, the settings this artifact was scored with. The
  trainer compares them against its own config at startup, see the table at the
  end.

The sums are raw, nothing is scaled by beta, so you score once and sweep beta
and learning rate against the same artifact.

Parallelism: `--tp` and `--sequence-parallel` change the logprobs, so both are
recorded and checked against the training config. `--ep` and `--etp` are
recorded too, but a mismatch there only warns. PP and DP change nothing: the
scorer runs at PP=1 and its artifact is valid at any training PP and DP.

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/dpo/score_reference_logprobs.py --tp 4 --ep 4 --etp 1 --sequence-parallel \
    --model Qwen/Qwen3-30B-A3B \
    --dataset HuggingFaceH4/ultrafeedback_binarized --split train_prefs \
    --output /data/uf_ref_logprobs
```

Batching: by default a batch is `--micro-batch-size` pairs (8), padded to the
longest pair in it. With long sequences that can OOM on a batch of long pairs.
`--token-budget N` switches to length-sorted batches capped at `N` padded
tokens, so memory is bounded and short pairs don't get padded up to a long
neighbour.

### 2. Train

```bash
uv run python -m torch.distributed.run --nproc_per_node=8 \
    scripts/training/run_recipe.py \
    --model qwen3_30b_a3b --mode dpo \
    --pretrained_checkpoint /checkpoints/qwen3-30b-a3b \
    dataset.source.path_or_dataset=HuggingFaceH4/ultrafeedback_binarized \
    dataset.source.split=train_prefs \
    dataset.ref_artifact=/data/uf_ref_logprobs \
    dataset.num_pairs=8192
```

`--mode dpo` picks the model's `<model>_dpo_config` recipe. The launcher itself
is documented in `scripts/training/README.md`.

PP > 1 turns on `model.variable_seq_lengths` for you, since pair batches are
padded to their own longest row.

Look at the first logged iteration. At step 0 the policy is the reference, so
`preference loss` should be about ln 2 (0.693) with `margin` and `rewards` near
0. Kernel and layout noise moves these by hundredths. A wrong artifact,
checkpoint or tokenizer moves them by whole units. Nothing checks this
automatically, so read the log before you walk away from a long run.

### Validation

Validation runs the same metrics on a held-out split every
`validation.eval_interval` steps. It needs its own artifact: run step 1 again on
the validation rows with the same model, tokenizer and `--max-seq-length`. Then
set `dataset.validation_source` and `dataset.validation_ref_artifact` in the
recipe and turn it on with `validation.eval_interval` and
`validation.eval_iters`. Eval settings without a validation split are rejected
at startup.

## What has to match between scoring and training

The margins only mean something if both runs saw the same tokens.

| Setting | Checked how |
|---|---|
| Dataset source and split | metadata check at startup |
| Tokenizer | metadata check at startup |
| Sequence length (`dataset.seq_length`, stored as `max_seq_length`) | metadata check at startup |
| Row layout (`dataset.prompt_key` / `--prompt-key`) | metadata check at startup |
| TP and sequence parallelism | metadata check at startup (PP and DP may differ) |
| `num_pairs`, same rows in the same order | artifact must cover `pair_id 0..N-1` exactly |
| Reference model is `pretrained_checkpoint` | not checked, shows up as step-0 loss far from ln 2 |
| No MTP head | `validate_dpo_run_config` |
| bf16 forward | not checked, the scorer pins it and the recipes default to `bf16_mixed` |

## Batch sizes are rows, not pairs

`micro_batch_size`, `global_batch_size` and the consumed-sample counters all
count rows, in the config, the train state and the logs. A pair is two rows,
chosen at the even index and rejected right after it, so both batch sizes must
be even.
