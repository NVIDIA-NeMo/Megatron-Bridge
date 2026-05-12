# Troubleshooting Guide

A compact, symptom-first triage reference for the most common training and
conversion failure modes in Megatron-Bridge. The goal is to help you narrow
down a failure to the right subsystem in a few minutes — so you can either
fix it locally or open an issue with the right terminology.

This page **does not replace** the detailed pages elsewhere in the docs.
Each row points at the canonical doc you should read next.

## How to use this page

1. Find the row whose **Symptom** matches what you're seeing.
2. Run the **First checks** in order — they are ordered cheapest-first.
3. Follow the **Where to look next** link to the authoritative page if the
   first checks don't resolve it.
4. If you still need help, open an issue and quote the row from this page —
   it gives oncall a fast starting point.

---

## Conversion and checkpoint loading

### Symptom: HF → Megatron conversion produces a mismatched parameter count

| | |
|---|---|
| **Likely cause** | Wrong bridge selected for the model family, or HF config has fields the Bridge config mapping does not handle yet |
| **First checks** | • Confirm `AutoBridge.from_hf_pretrained(...)` resolves to the bridge you expect — print `bridge.__class__`.<br>• Compare HF `config.json` against the matching `param_mapping.py` in `src/megatron/bridge/models/<family>/`.<br>• Verify the HF model is on the supported-models list in `README.md`. |
| **Where to look next** | [`bridge-tech-details.md`](bridge-tech-details.md), [`bridge-guide.md`](bridge-guide.md) |

### Symptom: Checkpoint resume produces a different loss than a continuous run

| | |
|---|---|
| **Likely cause** | RNG state not restored, optimizer state not loaded, or `nemo_experiments/` from a previous run was auto-resumed |
| **First checks** | • Verify `--load <path>` points at the checkpoint you expect (not a stale auto-resume).<br>• Confirm `rng.seed` is identical between the runs.<br>• Inspect the saved checkpoint contents — `torch.load(...)` and check `optimizer_state_dict` is present and non-empty.<br>• Delete `nemo_experiments/` if you are starting fresh, to avoid silent auto-resume. |
| **Where to look next** | [`training/checkpointing.md`](training/checkpointing.md), [`training/resiliency.md`](training/resiliency.md) |

### Symptom: Loading a checkpoint fails with "missing keys" or "unexpected keys"

| | |
|---|---|
| **Likely cause** | Architecture mismatch between the saved checkpoint and the current model config (e.g. different number of layers, hidden size, head count, or vocab size) |
| **First checks** | • Compare the model section of `config.yaml` saved alongside the checkpoint against your current `model_provider` config.<br>• If you renamed parameters (e.g. via a refactor), re-run conversion from HF rather than mutating the checkpoint by hand. |
| **Where to look next** | [`training/checkpointing.md`](training/checkpointing.md) |

---

## Tokenizer and vocabulary

### Symptom: Tokenizer or vocab size mismatch after migration

| | |
|---|---|
| **Likely cause** | Recipe is using `NullTokenizer` with one vocab size but the checkpoint embedding layer was built for a different size; or `make_vocab_size_divisible_by` was changed between training and inference |
| **First checks** | • Print `cfg.tokenizer.tokenizer_type` and `cfg.tokenizer.vocab_size`.<br>• Compare against `model.embedding.word_embeddings.weight.shape[0]` of the loaded checkpoint.<br>• If `should_pad_vocab=True`, confirm `make_vocab_size_divisible_by` and TP size match the original run. |
| **Where to look next** | [`training/config-container-overview.md`](training/config-container-overview.md) |

### Symptom: SFT or PEFT runs but loss is computed over the entire conversation

| | |
|---|---|
| **Likely cause** | Loss mask is not being produced correctly — usually a chat-template / `return_assistant_tokens_mask` interaction |
| **First checks** | • Inspect a single batch's `loss_mask` tensor: it should be 1 on assistant tokens and 0 elsewhere.<br>• Verify the HF tokenizer has a chat template that supports `{% generation %}` / `return_assistant_tokens_mask`. |
| **Where to look next** | [`training/peft.md`](training/peft.md) |

---

## Sequence length and packing

### Symptom: "RuntimeError: ... shape mismatch ... seq_len" during training

| | |
|---|---|
| **Likely cause** | `model.seq_length` does not match `dataset.sequence_length`, or packed sequences are enabled on a dataset that wasn't pre-padded to a multiple of `pad_seq_to_mult` |
| **First checks** | • Confirm `cfg.model.seq_length == cfg.dataset.sequence_length`.<br>• If using packed chat, verify `pad_seq_to_mult` and inspect a packed batch's shape end-to-end. |
| **Where to look next** | [`training/packed-sequences.md`](training/packed-sequences.md) |

### Symptom: Packed dataset produces unexpectedly long sequences

| | |
|---|---|
| **Likely cause** | The histogram-key (length before padding) and the stored tensor length (after padding to a multiple) are different; packing concatenates the stored length, not the bucket length |
| **First checks** | • Print histogram keys vs. stored `len(input_ids)` for a couple of samples.<br>• Confirm `pad_seq_to_mult` is the value you expect. |
| **Where to look next** | [`training/packed-sequences.md`](training/packed-sequences.md) |

---

## Parallelism setup

### Symptom: Training hangs at the start of step 0

| | |
|---|---|
| **Likely cause** | `world_size` does not equal `TP × PP × CP × DP`, or the launcher is not creating the process groups Bridge expects |
| **First checks** | • Print `(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size)` and the launcher's `world_size`.<br>• Confirm exactly one `torch.distributed.run` (or `srun --ntasks-per-node` + `uv run python`) is launching the job — not both.<br>• On Slurm, do **not** mix `torchrun` with `ntasks-per-node=8`; pick one. |
| **Where to look next** | [`parallelisms.md`](parallelisms.md), [`training/communication-overlap.md`](training/communication-overlap.md) |

### Symptom: NaN loss across all ranks shortly after eval

| | |
|---|---|
| **Likely cause** | TP > 1 + CUDA graph capture interacting with an evaluation pass that doesn't replay correctly |
| **First checks** | • Disable CUDA graphs (`cuda_graph_impl=none`) and re-run.<br>• If that fixes it, file a bug — this is a known interaction surface. |
| **Where to look next** | [`training/cuda-graphs.md`](training/cuda-graphs.md) |

### Symptom: OOM at the same step that previously fit on the same GPU count

| | |
|---|---|
| **Likely cause** | A new run is using a different optimizer (distributed-optimizer toggled), virtual-PP penalty changed, or activation recomputation was disabled |
| **First checks** | • Use the theoretical-memory estimator to compare expected memory across the two configs.<br>• Compare `optimizer.use_distributed_optimizer`, `model.virtual_pipeline_model_parallel_size`, and `model.recompute_granularity` between runs. |
| **Where to look next** | [`training/activation-recomputation.md`](training/activation-recomputation.md), [`training/megatron-fsdp.md`](training/megatron-fsdp.md) |

---

## Config drift and runtime workflow

### Symptom: Two "identical" runs produce different loss curves

| | |
|---|---|
| **Likely cause** | Config drift via CLI overrides, mixed-precision recipe, or a different MCore submodule pin |
| **First checks** | • Diff the `config.yaml` files saved by each run (Bridge writes the full effective config to the run dir).<br>• Confirm the MCore submodule SHA is identical (`git -C 3rdparty/Megatron-LM rev-parse HEAD`).<br>• Confirm the container tag is identical. |
| **Where to look next** | [`training/config-container-overview.md`](training/config-container-overview.md), [`releases/software-versions.md`](releases/software-versions.md) |

### Symptom: CI never started on my external-contributor PR

| | |
|---|---|
| **Likely cause** | `copy-pr-bot` requires either GPG-signed commits or an `/ok to test <sha>` comment from an NVIDIAN |
| **First checks** | • Confirm a maintainer has posted `/ok to test <full-sha>` (not the short SHA) on the PR.<br>• Push a new commit only if you also need a new `/ok to test` for the new SHA. |
| **Where to look next** | [`CONTRIBUTING.md`](../CONTRIBUTING.md#running-github-ci) |

---

## When to open an issue

Open an issue when the **First checks** rule out the obvious causes and the
failure is reproducible on a small config. Include:

- The exact symptom (error message, NaN at step N, loss diff value, etc.)
- The row of this guide you followed, and which First checks you ran
- The model family, parallelism shape (`TP × PP × CP × DP`), and container tag
- A minimal recipe call or CLI command that reproduces it
- The relevant section of the run's saved `config.yaml`

This template lets oncall route the issue to the right area label (`area:ckpt`,
`area:data`, `area:perf`, etc.) and start triage immediately.
