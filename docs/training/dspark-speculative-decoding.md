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
same training recipe maps onto Megatron-Bridge and Megatron-Core. It
intentionally does **not** ship the draft module itself: that is a Megatron-Core
neural module (see [Decision: Megatron-Core native](#decision-megatron-core-native))
whose parallel/loss integration must be validated on multi-GPU runs, so it belongs
in later, separately reviewed PRs.

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
- `L_l1` (weight 0.9): the **raw** probability L1 distance `||p_draft - p_target||_1`
  to the target's next-token distribution. This is the dominant term and a direct
  acceptance proxy.
- `L_conf` (weight 1.0): binary cross-entropy training the confidence head against
  the analytical acceptance label `c_k* = 1 - 0.5 * ||p_draft - p_target||_1`.

The factor `1/2` belongs to the acceptance label only. Paper
[Eq. 10](https://arxiv.org/html/2607.05147#S3.E10) minimizes the weighted **raw**
L1 distance at coefficient 0.9; the `1/2` that converts L1 into a total-variation
distance appears in [Eq. 8](https://arxiv.org/html/2607.05147#S3.E8), where the
analytical acceptance/confidence target is formed. Both the
[DeepSpec](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/dspark/loss.py)
and [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/speculative/dspark/loss.py)
implementations make the same distinction. Training `0.5 * L1` instead would halve
the dominant term's gradient while leaving the published coefficient in place, so
the two must not be conflated. A regression test asserts the pair together
(disjoint distributions give `l1_loss == 2` and `accept_rate == 0`) so the
implementation and this page cannot drift apart.

The target's teacher distribution is produced by passing the target's last hidden
state through the (shared, frozen) LM head and is always detached, so no gradient
flows into the target.

### Acceptance and confidence

Training measures acceptance analytically rather than by running rejection
sampling. Per position, `accept_rate = clamp(1 - 0.5 * ||p_draft - p_target||_1, 0, 1)`
(this is where the `1/2` of Eq. 8 lives). The expected accepted
prefix length of a block is `tau = sum_k cumprod(accept_rate)_k + 1` (a token
survives only if every earlier token in its block is accepted; the `+1` counts the
verified anchor token). `tau` is the training-time speedup proxy. The confidence
head regresses to `accept_rate`; the paper applies a post-hoc Sequential
Temperature Scaling calibration for serving.

## Where DSpark fits in Megatron-Bridge

### Current state

Megatron-Bridge has no standalone speculative or draft-training subsystem today.
The only draft-adjacent training path is **Multi-Token Prediction (MTP)**, which
is a native Megatron-Core module that Bridge integrates thinly (provider flags,
batch plumbing in `training/gpt_step.py`, and per-model HF weight mappings such as
`mtp.*` in `models/qwen/qwen3_bridge.py`). EAGLE and Medusa appear only in the
ModelOpt **export** path, not in training.

DSpark's draft (semi-autoregressive block plus Markov and confidence heads) is a
new neural module, so the first thing this RFC has to settle is which repository
owns it end to end. The two candidates lead to genuinely different contracts, and
picking one is a prerequisite for every section below.

1. **Megatron-Core native** (as MTP does in
   `megatron/core/transformer/multi_token_prediction.py`): Megatron-Core owns the
   parallel module, the loss folding, and the sharded checkpoint; Bridge owns
   configuration and the HF mapping. Requires a Megatron-LM change and a
   Bridge-defined HF export schema.
2. **ModelOpt-injected** (as EAGLE/Medusa are, via `modelopt.torch.speculative`):
   ModelOpt owns the transformation, module, loss, and export; Bridge only
   configures it. Requires nothing new in Megatron-LM, but ModelOpt has to grow a
   Megatron-side DSpark plugin first.

### Decision: Megatron-Core native

**This RFC selects option 1.** ModelOpt's DSpark support is currently
HF-path-only: `modelopt/torch/speculative/plugins/` ships `hf_dspark.py` and
`modeling_dspark.py`, and `modelopt/torch/export/plugins/hf_spec_export.py` ships
`DSparkExporter`, but the Megatron-side plugins are `megatron_eagle.py` and
`megatron_medusa.py` only. There is no `megatron_dspark.py`. Choosing option 2
would therefore mean landing a Megatron-native DSpark in Model-Optimizer first and
then waiting on Bridge's ModelOpt pin (currently `nvidia-modelopt==0.46.0rc1`) to
pick it up, which puts a cross-repo dependency in front of every Bridge-side
milestone. Option 1 keeps the critical path inside Megatron-LM and Bridge, where
MTP already demonstrates the pattern end to end.

Consequences of the decision, which the rest of this document then specifies:

- The parallel draft module, its loss folding, and its sharded checkpoint are a
  **Megatron-Core** contribution, not a Bridge one.
- Bridge owns the **HF export schema** for trained drafts, because ModelOpt's
  `DSparkExporter` covers its own HF-path artifacts and not an MCore-native
  checkpoint.
- The architecture-agnostic pieces that depend only on `torch` (the sequential and
  confidence heads, and the CE / L1 / confidence objective) can be developed and
  unit-tested in Bridge, and move to Megatron-Core when the parallel module lands
  there. They are not a substitute for that module.

## Proposed design

The construction path is `ModelConfig` + `ModelBuilder`, **not** a
`ModelProviderMixin` subclass: `ModelProviderMixin`-based configuration is
deprecated in current Bridge main (`models/model_provider.py` raises a
`DeprecationWarning` directing callers to `ModelConfig` + `ModelBuilder`), so a new
provider subclass could not be the stable API this RFC promises. The DSpark config
subclasses `GPTModelConfig` and names a `DSparkModelBuilder`, mirroring how
`MambaModelConfig` names `MambaModelBuilder`.

| Piece | Location (proposed) | Mirrors |
|---|---|---|
| Draft config + builder | `models/dspark/dspark_builder.py` (`DSparkModelConfig`, `DSparkModelBuilder`) | `models/mamba/mamba_builder.py` (`MambaModelConfig` naming `MambaModelBuilder`) |
| Parallel draft module | Megatron-Core `megatron/core/transformer/` | `multi_token_prediction.py` |
| Custom forward + loss dispatch | `training/gpt_step.py` (`forward_step_dspark`) | `forward_step_modelopt` |
| DSpark loss | `training/post_training/dspark/` | `post_training/distillation.py` (the three-term CE/L1/confidence loss and the `tau` acceptance metric) |
| Thin entry point | `training/dspark.py` | `training/distill.py` |
| HF to/from Megatron weight mapping for `dspark.*` params | `models/<family>/<name>_bridge.py` mapping registry | the `mtp.*` mappings in `qwen3_bridge.py` |
| Recipe + example | `recipes/<family>/…` + `examples/dspark/` | the distillation recipe/example |
| Docs | this page | `multi-token-prediction.md` |

### Loss return contract

`dspark_loss` follows the Megatron-Core 3-tuple contract used by
`training/losses.py`: it returns the **unnormalized** micro-batch numerator, this
micro-batch's `num_tokens`, and a report mapping each metric to a reducible
`[numerator, denominator]` pair. Normalization happens once, against the globally
reduced token count.

This requires `calculate_per_token_loss=True` on the model config, and DSpark
training must fail fast when it is unset. With the default `False`, Megatron-Core
divides each micro-batch by its own `num_tokens` before averaging over
micro-batches, which trains a mean of micro-batch means. That is not a rounding
detail: with two ranks running two micro-batches each at uneven supervised-token
counts, the per-micro-batch normalization produces a gradient **37% away** from
the full-batch reference, while the global-token normalization matches it to
`1.8e-07`.

Reporting pairs exist for the same reason. A logged ratio must be formed as
`sum(numerator) / sum(denominator)` over the log window and the data-parallel
group; averaging per-micro-batch ratios reintroduces the same bias in the metrics.

### Memory contract for the full-vocabulary terms

Both the cross-entropy and the L1 term need the draft's normalized distribution
over the whole vocabulary. At reference scale (`num_blocks=512`, `block_size=7`,
`vocab≈150k`) one FP32 distribution tensor is 2.03 GiB, and computing the terms
separately keeps several live at once: the L1 needs the draft probabilities, the
target probabilities, and their difference, while the cross-entropy saves its own
`log_softmax` for the backward. Measured on an 80 GB card, an unchunked pass costs
**10.15 GiB** of forward-plus-backward overhead above its inputs.

The implementation therefore derives both terms from a single FP32 `log_softmax`,
chunks the flattened rows, and recomputes each chunk in the backward. That brings
the overhead to **2.06 GiB** (4.9x) at identical values for both terms. Any owner
of this module inherits the same constraint: either chunk, or state and enforce a
much smaller anchor-count contract with measurements. Chunking only the L1 is not
enough, because the cross-entropy's saved `log_softmax` is a full-size tensor of
the same order.

### Frozen weights: replicas, not aliases

The draft does **not** physically alias the target's parameters. The AutoModel
reference creates draft-owned `embed_tokens` and `lm_head`, copies the target
weights into them, freezes the replicas, and controls their trainability
separately. This RFC adopts the same storage model, because it is what makes the
remaining contracts well-defined:

- **Pipeline placement.** Replicas let the draft's embedding and LM head sit on
  whatever pipeline stage hosts the draft, instead of forcing the draft to share a
  stage with the target's first and last stages.
- **Checkpointing.** The frozen replicas are ordinary sharded-checkpoint entries.
  They are saved (not reconstructed) so a resumed run is bit-identical without
  needing the target checkpoint present.
- **Trainability.** "Frozen" is a `requires_grad=False` property of draft-owned
  parameters, not an aliasing side effect, so unfreezing the LM head later is a
  config change rather than a structural one.

### Frozen-target supervision

The draft trains against features captured from the frozen target: the selected
decoder-layer hidden states (fused by `fc` into the draft's context K/V) and the
final hidden state (which drives the teacher distribution through the shared LM
head). Two capture modes are supported, with different schemas:

- **Colocated capture** during the same forward (analogous to how MTP consumes
  extra token IDs), keeping the target and draft in one model. No cache format;
  costs target memory.
- **Offline precomputation** of target features, streamed to a draft-only training
  job (as AutoModel's `precompute_dspark.py` does). Sidesteps holding the target
  in memory, and requires a versioned cache schema recording the target checkpoint
  id, the selected layer indices, the hidden dtype, the anchor sampling seed, and
  the block size, so a cache can never be silently paired with a different target.

Both modes must respect tensor, pipeline, and context parallelism:

- **Tensor parallel.** The teacher and draft logits are vocab-sharded, so the L1
  distance must be computed shard-locally and summed across the TP group rather
  than gathered: `||p_d - p_t||_1` is the sum over TP shards of the local absolute
  difference, with the softmax denominators reduced across the group first. The
  exactness of this term is what the acceptance label depends on, so an
  approximate (top-k, or gathered in lower precision) path is not acceptable here.
- **Pipeline parallel.** The selected decoder-layer hidden states and the final
  hidden state must be transported to the draft's stage. When the draft shares the
  last stage, this is a local read; otherwise the selected hiddens are additional
  pipeline payload, and their dtype and layout are part of the module contract.
- **Context parallel.** Anchor blocks must not straddle CP shard boundaries.
  Anchors are sampled per CP shard from positions whose whole block is local, and
  the loss mask zeroes any block that would cross a boundary, so no cross-shard
  attention or gather is needed for the draft.

### Checkpoint and export ownership

- **Training checkpoint.** MCore-native sharded checkpoint, owned by the
  Megatron-Core module, covering the draft backbone, `fc`, `markov_head`,
  `confidence_head`, and the frozen embedding and LM-head replicas.
- **HF export.** Owned by Bridge. ModelOpt's `DSparkExporter` targets ModelOpt's
  own HF-path artifacts, so an MCore-native checkpoint needs its own HF schema
  before `AutoBridge` can round-trip it. The bridge mapping registry gains
  `dspark.*` entries (QKV and gated-MLP splits for the draft backbone, plain
  linears for `fc` and the heads); the frozen replicas are exported as ordinary
  weights so a served draft is self-contained. The `mtp.*` mappings in
  `qwen3_bridge.py` are the closest existing template.

These mappings are specified only once the storage model above is fixed; listing
them earlier would have been guesswork about whether the shared tensors are stored
or reconstructed.

## References

- DSpark paper: [arXiv:2607.05147](https://arxiv.org/abs/2607.05147),
  *Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation*.
- Draft-training reference implementation: NeMo AutoModel,
  [`nemo_automodel/components/speculative/dspark/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/nemo_automodel/components/speculative/dspark).
- Tracking issue: [#5230](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5230).
- Related in-repo training: [Multi-Token Prediction](multi-token-prediction.md).
