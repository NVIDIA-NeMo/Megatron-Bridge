# DSpark Speculative Decoding (Draft Training) Design

## Overview

This page is a design proposal (RFC) for adding **DSpark** speculative-decoding
draft training to Megatron-Bridge. It tracks
[issue #5230](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5230).

DSpark ([arXiv:2607.05147](https://arxiv.org/abs/2607.05147)) is a
speculative-decoding method that trains a lightweight **draft** against a frozen
target. The draft proposes a block of tokens; the target verifies them. DSpark's
draft is *semi-autoregressive*: a parallel backbone predicts a whole block in one
pass, and a small sequential head restores the intra-block token dependency that a
purely parallel drafter loses (the "suffix decay" problem). A confidence head
predicts per-position acceptance so the serving engine can schedule its verify
budget.

A working draft-training reference already exists in NeMo AutoModel at
`nemo_automodel/components/speculative/dspark/`. This document specifies how the
same training recipe maps onto Megatron-Bridge and Megatron-Core, and proposes a
phased plan. It intentionally does **not** ship the draft module itself: that is a
Megatron-Core (or ModelOpt) neural module whose parallel/loss integration must be
validated on multi-GPU runs, so it belongs in later, separately reviewed PRs.

## Background: the DSpark draft

### Semi-autoregressive draft

For a block of `block_size` positions (`gamma`, default 7 offline / 5 in
production), the draft factorizes the block distribution as
`P(x | x0) = prod_k p_k(x_k | x0, x_1..x_{k-1})`:

- **Parallel backbone.** A small transformer predicts every position of the block
  in one forward pass. Crucially, the backbone does not re-encode the prompt: it
  attends to the frozen target's hidden states as the attention **context K/V**,
  with the block's own "noise" slots as the queries. Each block query attends to
  the context strictly before its anchor and bidirectionally within its own block.
  This is the block-parallel (DFlash-style) half of the draft.
- **Sequential head.** A lightweight module adds intra-block dependency on top of
  the parallel logits. DSpark provides a **Markov head** (a low-rank,
  token-conditioned additive logit bias, rank 256) and an **RNN head** (a
  GRU-like recurrence carrying prefix state across the block). At training time
  the head is teacher-forced with the previous block tokens; only at inference is
  it run serially.
- **Confidence head.** A small projection `c_k = sigmoid(w^T [h_k ; embed(x_{k-1})])`
  predicts each position's acceptance probability.

The draft **shares and freezes the target's token embedding and LM head**; there
is no separate (compressed) draft vocabulary or `d2t`/`t2d` remapping. The
target's selected decoder-layer hidden states are fused by a learned linear
projection (`fc`) into the draft hidden size and used as the attention context.

### Training objective

The draft is trained with three position-weighted terms (weights `exp(-k/gamma)`,
emphasizing earlier positions):

- `L_ce` (weight 0.1): cross-entropy of the draft logits against the target's
  teacher-forced next tokens.
- `L_tv` (weight 0.9): total-variation distance `0.5 * ||p_draft - p_target||_1`
  to the target's next-token distribution. This is the dominant term and a direct
  acceptance proxy.
- `L_conf` (weight 1.0): binary cross-entropy training the confidence head against
  the analytical acceptance label `c_k* = 1 - TV(p_draft, p_target)`.

The target's teacher distribution is produced by passing the target's last hidden
state through the (shared, frozen) LM head and is always detached, so no gradient
flows into the target.

### Acceptance and confidence

Training measures acceptance analytically rather than by running rejection
sampling. Per position, `accept_rate = clamp(1 - TV, 0, 1)`. The expected accepted
prefix length of a block is `tau = sum_k cumprod(accept_rate)_k + 1` (a token
survives only if every earlier token in its block is accepted; the `+1` counts the
verified anchor token). `tau` is the training-time speedup proxy. The confidence
head regresses to `accept_rate`; the paper applies a post-hoc Sequential
Temperature Scaling calibration for serving.

### Reference implementation (NeMo AutoModel)

The AutoModel implementation separates cleanly into an architecture-agnostic core
and per-target glue:

- Architecture-agnostic core: the Markov/RNN heads, the three-term loss and
  acceptance metrics, anchor sampling and block "noise" embedding, the eval mask,
  the confidence-head module, and the block-attention mask builder.
- Per-target glue: the draft backbone (attention that reads target hidden states
  as context K/V, plus the decoder MLP), a draft config builder, a one-line
  registry entry, and the frozen-target hidden-state capture.

This split is what makes a Megatron-Bridge port tractable: the core is nearly
verbatim, and only the backbone must be re-expressed in Megatron-Core layer specs.

## Where DSpark fits in Megatron-Bridge

### Current state

Megatron-Bridge has no standalone speculative or draft-training subsystem today.
The only draft-adjacent training path is **Multi-Token Prediction (MTP)**, which
is a native Megatron-Core module that Bridge integrates thinly (provider flags,
batch plumbing in `training/gpt_step.py`, and per-model HF weight mappings such as
`mtp.*` in `models/qwen/qwen3_bridge.py`). EAGLE and Medusa appear only in the
ModelOpt **export** path, not in training.

DSpark's draft (semi-autoregressive block plus Markov and confidence heads) is a
new neural module. It therefore has to live in one of two places:

1. **A Megatron-Core native module** (as MTP does in
   `megatron/core/transformer/multi_token_prediction.py`), with tensor- and
   pipeline-parallel-aware layers and loss folding. This is a Megatron-LM change,
   not a Bridge change.
2. **A ModelOpt-injected module** (as EAGLE/Medusa are, via
   `modelopt.torch.speculative`), in which case Bridge only needs a provider hook,
   a custom forward step, and export mappings.

### Blueprint: the ModelOpt distillation subsystem

Bridge already contains a self-contained, bolt-on training subsystem that DSpark
should mirror: **ModelOpt knowledge distillation**. It slots a new training mode
in without touching the core training loop, via four pieces:

- `models/distillation_provider.py`: a provider that dynamically subclasses the
  base provider and registers a pre-wrap hook to mutate the already-built,
  weight-loaded model (there, it attaches the teacher).
- `training/post_training/distillation.py`: the custom loss (`loss_func_kd`).
- `training/gpt_step.py`: `forward_step_modelopt`, which selects that loss when the
  model is a distillation model.
- `training/distill.py`: a thin entry point, `distill(config) = pretrain(config,
  forward_step_modelopt)`.

Plus an `examples/distillation/` recipe and a `docs/training/` page. This
provider-hook plus custom-forward-step plus custom-loss plus thin-entry pattern is
the established, low-friction way to add a training subsystem here, and it is the
recommended shape for DSpark.

## Proposed design

Following the distillation blueprint, the Bridge-side surface is:

| Piece | Location (proposed) | Mirrors |
|---|---|---|
| Draft provider hook | `models/dspark_provider.py` | `distillation_provider.py` (dynamic subclass + pre-wrap hook that attaches the frozen target's feature capture and the draft heads) |
| Custom forward + loss dispatch | `training/gpt_step.py` (`forward_step_dspark`) | `forward_step_modelopt` |
| DSpark loss | `training/post_training/dspark.py` | `post_training/distillation.py` (the three-term CE/TV/confidence loss and the `tau` acceptance metric) |
| Thin entry point | `training/dspark.py` | `training/distill.py` |
| HF to/from Megatron weight mapping for `dspark.*` params | `models/<family>/<name>_bridge.py` mapping registry | the `mtp.*` mappings in `qwen3_bridge.py` |
| Export flag | ModelOpt `export_extra_modules` path | the existing EAGLE/Medusa/MTP export |
| Recipe + example | `recipes/<family>/…` + `examples/dspark/` | the distillation recipe/example |
| Docs | this page | `multi-token-prediction.md` |

### Frozen-target supervision

The draft trains against features captured from the frozen target: the selected
decoder-layer hidden states (fused by `fc` into the draft's context K/V) and the
final hidden state (which drives the teacher distribution through the shared LM
head). In Megatron-Core, this capture must respect tensor, pipeline, and context
parallelism. Two options, to be decided in Phase 1:

- Colocated capture during the same forward (analogous to how MTP consumes extra
  token IDs), keeping the target and draft in one model.
- Offline precomputation of target features, streamed to a draft-only training
  job (as AutoModel's `precompute_dspark.py` does), which sidesteps holding the
  target in memory but adds a cache format.

### HF to/from Megatron conversion

Because the draft reuses the target's embedding and LM head and adds `fc`,
`markov_head`, `confidence_head`, and the draft decoder layers, the bridge mapping
registry must gain `dspark.*` entries (QKV and gated-MLP splits for the draft
backbone, plain linears for `fc` and the heads), so trained drafts round-trip
through `AutoBridge` for serving. The `mtp.*` mappings are the closest existing
template.

## Phased implementation plan

- **Phase 0 (this PR): design.** Specify the algorithm, the Bridge mapping, and
  the plan.
- **Phase 1: the DSpark draft module.** Implement the semi-autoregressive backbone
  plus Markov and confidence heads as a parallel-aware module (Megatron-Core
  native, or a ModelOpt speculative module), with the three-term loss. Validate
  numerically with L0 unit tests and L1/L2 functional tests within the 2-GPU cap.
  This is the load-bearing, separately reviewed piece.
- **Phase 2: Bridge integration.** Add `dspark_provider.py`, `forward_step_dspark`,
  `post_training/dspark.py`, the `training/dspark.py` entry, and the `dspark.*`
  weight mappings, following the distillation files.
- **Phase 3: recipe, example, and convergence.** Add a recipe/config and an
  example, and validate acceptance-length (`tau`) convergence against the AutoModel
  reference on a small target (for example Qwen3 dense) on multi-GPU.

## Open questions and risks

- **Module placement.** MCore-native (a Megatron-LM PR, full TP/PP/EP/CP support)
  versus ModelOpt-injected (Bridge-only, but tied to ModelOpt). Phase 1 must
  decide this first; it drives everything downstream.
- **Parallelism.** The draft heads and the block-attention mask need correct
  tensor/pipeline/expert/context-parallel behavior; this is the part that cannot
  be validated single-GPU and must go through functional tests.
- **Optional dependency.** If the draft leans on `modelopt.torch.speculative`, it
  must be an optional extra and land as a separate dependency PR first, per
  `CONTRIBUTING.md`.
- **Target-feature capture cost.** Colocated capture holds the target in memory;
  offline precompute adds a cache format. The choice affects the recipe surface.

## References

- DSpark paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147),
  *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*.
- Draft-training reference implementation: NeMo AutoModel,
  [`nemo_automodel/components/speculative/dspark/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components/speculative/dspark).
- Tracking issue: [#5230](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5230).
- Related in-repo training: [Multi-Token Prediction](multi-token-prediction.md).
