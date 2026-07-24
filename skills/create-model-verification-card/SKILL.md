---
name: create-model-verification-card
description: Create or update concise, agent-readable Megatron Bridge model verification cards. Use when adding a model support card, auditing cross-model convergence comparability or verification coverage, recording conversion, deterministic inference, training, checkpoint resume, post-SFT export, or performance results, or preparing a model-support PR. Enforce the required core inventory, convergence-versus-performance contracts, optional canonical performance item, public Slurm launcher commands, training metrics, important-feature allowlist, and a strict privacy boundary that excludes private runtime wiring, internal paths, credentials, and job metadata.
---

# Create Model Verification Card

Create `examples/model_verification_cards/<model-slug>/card.yaml` from public
model facts and verified commands. Keep the card small enough for an agent to
scan without interpreting logs or reconstructing the execution environment.

## Use the repository resources

- Validate the result with [scripts/validate_card.py](scripts/validate_card.py).
- Verify exact-length deterministic HF output with
  [scripts/verify_hf_inference.py](scripts/verify_hf_inference.py).
- Use the inventory and field rules below as the format contract. Do not infer
  model-specific settings from another family or variant.

Do not add a README, evidence blobs, log excerpts, runtime setup, or scheduler
metadata to the skill or card.

## Workflow

### 1. Pull facts before drafting

Read the model implementation, public HF config, conversion bridge, recipes,
tests, examples, and the exact revision being verified. Determine which modes
exist; do not infer support from a family name or another model size.

Record the public HF model name in commands, its immutable revision in
`model.hf_revision`, and the minimum supported Transformers version in
`model.min_transformers_version`. Do not introduce an HF snapshot path.

Record only two execution-environment facts: the public base container
identifier and the exact Bridge commit used for verification. Put them in
`verification_environment.base_container` and
`verification_environment.bridge_commit`. If the run mounted a checkout over
the container, record the mounted checkout commit. Never substitute a private
image path for the public base container identifier.

The top-level Bridge commit is the default for every item leaf. If one verified
item was run from a different clean checkout, put that exact 40-hex commit in
the leaf's optional `bridge_commit` field. Omit the field when it would repeat
the top-level value, and never use a commit field to disguise uncommitted
runtime changes. Items that are not verified must not carry a commit override.

### 2. Create the core inventory and add performance when available

Include these twelve required items, even when their status is `unsupported` or
`not_applicable`:

1. CPU HF-to-Megatron conversion
2. GPU HF-to-Megatron conversion
3. CPU Megatron-to-HF conversion
4. GPU Megatron-to-HF conversion
5. Manual HF/Megatron forward-pass correlation
6. Deterministic Megatron inference
7. Pretraining
8. SFT
9. SFT checkpoint export and deterministic HF inference
10. Long-context SFT
11. PEFT
12. Checkpoint resume

Add `pretrain_performance` as a thirteenth item only when the exact model
variant has a canonical public performance recipe. If no such recipe is
exported, omit the item instead of adding an unverified placeholder. Once a
canonical recipe exists, keep the item in the card even if its run is still
unverified.

A concrete `pretrain_performance.<hardware>` leaf means a tuned canonical
performance recipe exists for that hardware. Its item status states whether
the card's benchmark run has been verified; an `unverified` leaf still records
the existence of the recipe. If the card has no concrete performance leaf,
start `summary` with this exact disclaimer before the functional-support
summary:

```text
Performance disclaimer: this model has not been performance-tuned; reported timing and throughput metrics are sanity checks, not optimized performance results.
```

Omit `pretrain_performance` entirely when no canonical recipe exists; do not
add an `all` terminal placeholder. Routine timing and throughput metrics in
functional pretrain, SFT, PEFT, long-context, and resume items remain sanity
observations; they do not become optimized results merely because a separate
performance recipe exists. When a tuned recipe exists, scope the summary to
the exact `pretrain_performance.<hardware>` leaf.

Keep conversion, manual-forward, and base-inference items as direct item
records. Key every training item by its public hardware target, and key the
dependent SFT export/inference item the same way:

```yaml
items:
  pretrain:
    H100:
      status: verified
      precision: bf16
      enabled_features: {}
      command: ...
      last_verified: 2026-07-21
      metrics: ...
      expected_result: ...
```

The hardware-scoped names are `pretrain`, `sft`, `sft_export_inference`,
`sft_long_context`, `peft`, `checkpoint_resume`, and optional
`pretrain_performance`. Use canonical public accelerator identifiers such as
`H100`, `B200`, or `GB200`, never a private cluster name. The validator's
public-hardware allowlist is authoritative and must be updated when a new
accelerator target is introduced. The hardware key replaces the old `gpu_type`
field. Each hardware leaf is independent and must carry its own status plus the
command or commands, date, metrics, features, and optional commit override that
apply to that item. Dependencies resolve within the same hardware key:
`checkpoint_resume.H100` consumes `pretrain.H100`, and
`sft_export_inference.H100` consumes `sft.H100`. Never fall back across
hardware targets. Use the reserved key `all` only as the sole leaf for a
model-wide `unsupported` or `not_applicable` limitation. A terminal dependency
leaf still names its logical dependency but does not require a matching `all`
or concrete-hardware dependency leaf.

Use only `unverified`, `verified`, `unsupported`, or `not_applicable`. Do not
add `smoke` or an evidence field.

#### Add a compact verification index

Every card must put a top-level `verification_index` immediately after
`summary`. This is a concise directory of the detailed `items`, not a second
source of evidence. The validator must reject an index that drifts from the
item records.

Use `model_level` for the six direct items: the four conversion directions,
`manual_forward_pass`, and `inference`. Use `training` for the six functional
hardware-scoped items: `pretrain`, `sft`, `sft_export_inference`,
`sft_long_context`, `peft`, and `checkpoint_resume`. Keep the optional
`pretrain_performance` item separate under `performance`; omit `performance`
when the card has no canonical performance recipe.

Group item names under the same four status names used by the detailed items.
For an explicitly indexed hardware target with no corresponding item leaf,
summarize the missing verification as `unverified`; do not add empty item
leaves or placeholder commands merely to populate the index. Omit empty status
buckets and always list every item name explicitly, even when every item in the
scope has the same status. Never use the scalar `all` in `verification_index`:
an explicit inventory makes it clear which existing items were assessed when
new items are added later.

For example, a card with partial H100 functional-training coverage and no
GB200 functional-training verification uses:

```yaml
verification_index:
  model_level:
    verified:
      - hf_to_megatron_cpu
      - hf_to_megatron_gpu
      - megatron_to_hf_cpu
      - megatron_to_hf_gpu
      - manual_forward_pass
      - inference
  training:
    H100:
      verified: [sft, sft_export_inference, sft_long_context, peft]
      unverified: [pretrain, checkpoint_resume]
    GB200:
      unverified: [pretrain, sft, sft_export_inference, sft_long_context, peft, checkpoint_resume]
```

When a canonical performance recipe exists, mirror only its concrete leaves:

```yaml
  performance:
    H100: verified
```

The index may declare an allowlisted public hardware target such as `GB200`
before detailed evidence exists. It must also include every concrete hardware
target present in the detailed items. Do not use a private cluster name or
invent a separate `not_tested` status; `unverified` covers both pending and
not-yet-run verification.

For `unsupported` and `not_applicable`, leave `command` or `commands`, date,
precision, and metrics null, then state the public product limitation in
`expected_result`.

Put a scalar `precision` on every direct item or hardware leaf. It describes
the workload that was actually verified, not every precision the model might
support:

- for conversion, record the imported or exported weight precision;
- for forward pass and inference, record the compute precision;
- for training, record the recipe's mixed-precision mode.

Use `bf16` for BF16. Training items may instead use `fp8_mx` for MXFP8 or
`nvfp4` for NVFP4. Keep MXFP8 and NVFP4 training-only, and do not list either
until that exact item has completed in that mode.

### 3. Use the public Slurm launchers

Assume the caller supplies the account, partition, concrete runtime image,
credentials, and storage mappings outside the card. The top-level
`verification_environment.base_container` is provenance only; it is not
launcher configuration. Record the portable public launcher and the
model-verification workload:

- use `scripts/conversion/convert.sh --executor slurm` for CPU and GPU
  conversion, with portable node and GPU counts;
- use `scripts/training/train.sh --nodes ... --gpus-per-node ...` for every
  training item;
- use `uv run python ...` only for inference helpers that do not yet have a
  public Slurm executor;
- use short, ignored repository-relative logical paths under `work/...`;
  prefer aliases such as `work/data/<dataset>` and `work/cache/<model>` over
  reproducing a physical storage hierarchy.

The public launchers may read their required generic Slurm configuration from
the caller's environment. Do not include `srun`, `sbatch`, concrete account or
partition values, container image arguments, `--mount`, `--env`, shell exports,
environment-variable references, cluster-specific `--srun-arg` values, or
launcher overlays. That wiring is personal to the verifier and does not belong
in a model verification card.

Never record or reproduce:

- execution-environment names, hostnames, IPs, usernames, emails, or accounts;
- concrete partitions, reservations, node lists, private image locations,
  mount sources, or environment forwarding;
- host/shared-storage paths, home-directory paths, log locations, or job IDs;
- tokens, token-loading commands, private URLs, or private registry references;
- environment-specific launcher overlays.

Keep private run notes outside the tracked repository and public PR text. If a
private codename cannot be recognized generically, pass it to the validator via
`--deny-term` or an untracked file through `--denylist "$PRIVATE_DENYLIST"`.
The validator reports the match without printing the private term.

### 4. Freeze convergence, execution, and benchmark contracts

Before launching any training item, resolve the selected recipe and classify
its effective configuration into the three groups below. Do this from the
built `ConfigContainer`, not only from command-line overrides. Record the
resolved convergence and execution field/value fingerprints in the internal
per-model record; keep the public card concise.

The **convergence contract** contains settings that change the training
objective, examples seen at an optimizer boundary, or numerical update rule:

- starting checkpoint and trainable parameter set;
- dataset identity/revision, split or bounded selection, sample order, seeds,
  tokenizer/chat template, truncation, masking, and packing semantics;
- sequence length, global batch size, global tokens per optimizer step, total
  optimizer steps, and total token budget;
- objective and loss settings, including label masking, MoE auxiliary/router
  losses, token dropping/capacity, natural versus forced routing, and loss
  normalization;
- optimizer family, peak/minimum learning rate, schedule shape, warmup and
  decay horizon, betas, epsilon, weight decay, gradient clipping, and dropout;
- model/gradient/optimizer-state precision and loss-scaling behavior;
- for PEFT, adapter type, targets, rank, alpha, dropout, and which base weights
  are frozen.

The **execution/performance contract** maps the frozen training semantics to
hardware. It may vary across models or be tuned for throughput:

- node/GPU count and TP, PP, VP, CP, EP, ETP, DP, and sequence parallelism;
- activation recompute, activation/optimizer offload, distributed optimizer or
  FSDP sharding, checkpoint I/O strategy, and garbage-collection policy;
- communication overlap, fused kernels, Transformer Engine implementation,
  attention backend, CUDA graphs, and compilation;
- MoE transport/dispatcher backend such as all-to-all, DeepEP, or HybridEP,
  provided routing, capacity, token dropping, and auxiliary losses are
  unchanged.

Treat micro batch size and gradient-accumulation layout as execution
fingerprints. They may be tuned while global batch size, global batch
membership/order, loss normalization, optimizer-step boundaries, and total
token budget remain intact. Because they can change accumulation order,
dropout RNG, MoE token grouping, and auxiliary-loss reduction, require fresh
loss sentinels for every layout and do not claim step-by-step numerical parity.

Performance settings are **intended** to preserve training semantics, not
guaranteed to be bitwise neutral. Parallel reductions, fusions, recompute, and
dispatcher implementations can change floating-point order. After changing
them, require finite loss, no skipped iterations, and compatible loss sentinels
before calling the mapping verified. Anything that changes arithmetic
precision, forced router balancing, token dropping, packing, or effective batch
construction is a convergence change, even when introduced to improve speed.

The **benchmark-only configuration** may deliberately change semantics to find
an upper throughput bound. It includes mock data, forced MoE load balancing,
changed batch/LR or timing-only schedules, and disabled NaN/large-gradient,
evaluation, or checkpoint checks. These settings may be valid for a canonical
`pretrain_performance` item, but their losses and checkpoints are never
convergence evidence.

Use `qwen3_30b_a3b_convergence_v2` as the named default cross-model bounded
convergence cohort. It retains the optimizer, data, objective, and token-budget
contract derived from the resolved Qwen3-30B-A3B H100 recipes while changing
offline-packed SFT and PEFT from 2K/GBS32 to 8K/GBS8. The name identifies
target settings, not evidence status.
Do not call a workload cohort-verified until its recipe owns this contract and
a clean-commit run passes the applicable gates below. Its 100 optimizer steps
test finite loss, short-horizon loss trend, checkpoint reload, and direct
resume; they do not establish full training convergence.

Historical `qwen3_30b_a3b_convergence_v1` evidence used 2K/GBS32 for SFT and
PEFT. Keep an existing verified card on that exact command, recipe commit, and
metrics until the model is rerun. Never relabel 2K evidence as v2 or rewrite its
command to 8K without fresh training. When refreshing a model, update the
recipe first, write packed data to a fresh output root, rerun the training
gates, and rerun post-SFT export/inference.

Freeze this optimizer fingerprint for all three workloads:

| Field | Value |
| --- | --- |
| Optimizer | Megatron distributed fused Adam |
| Betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Effective weight decay | `0.033`, constant |
| Gradient clipping | `1.0` |
| LR schedule | Cosine, starting from zero |
| Model / compute precision | BF16 |
| Optimizer master parameters, main gradients, moments | FP32 |

Record the effective weight decay applied to optimizer parameter groups, not
only the nominal optimizer-config value. The Qwen anchor has a nominal
`optimizer.weight_decay=0.1`, but its constant scheduler applies `0.033`; use
`0.033` when reproducing this contract. Treat any other effective value
as a convergence-contract change.

Freeze the following workload profiles:

| Field | Pretrain | Full SFT | PEFT |
| --- | --- | --- | --- |
| Start / trainable set | Random initialization, no checkpoint load, full model | Exact immutable HF checkpoint revision, full model | Same immutable HF revision, frozen base model; LoRA on model-native attention Q/K/V and output projections, rank 8, alpha 16, dropout 0 |
| Data | Same bounded raw RP2 selection, revision, sample order, and seeds | Tulu 3 `train[:10000]`; same revision, order, chat template, label mask, truncation, and offline packing | Same as full SFT |
| Sequence / GBS | `4096 / 1024` | `8192 / 8` | `8192 / 8` |
| Reference MBS | `1` | `1` | `1` |
| Offline packing alignment | Not applicable | Derive from resolved CP/TP/SP topology and pin explicitly | Same rule |
| Token slots | `4,194,304` per step; `419,430,400` total | `65,536` per step; `6,553,600` total | `65,536` per step; `6,553,600` total |
| Peak / minimum LR | `3e-4 / 3e-5` | `5e-6 / 0` | `1e-4 / 0` |
| Horizon | 100 steps, 40 warmup steps, cosine decay through step 100, saves at steps 50 and 100 | 100 steps, 10 warmup steps, cosine decay through step 100, final checkpoint at step 100 | 100 steps, 10 warmup steps, cosine decay through step 100, final adapter checkpoint at step 100 |
| RNG | Model and dataset seed `1234` | Model RNG seed `5678`; data-order and packing seed `1234` | Model RNG seed `5678`; data-order and packing seed `1234` |
| Gradient path | BF16 gradient reduction; precision-aware optimizer enabled | FP32 gradient reduction; precision-aware optimizer disabled | FP32 gradient reduction; precision-aware optimizer disabled |

Derive `pad_seq_to_mult` for both SFT and PEFT from the resolved execution
topology:

```text
cp_multiple = 2 * CP if CP > 1 else 1
sp_multiple = CP * TP if sequence parallelism is enabled and TP > 1 else 1
pad_seq_to_mult = lcm(cp_multiple, sp_multiple)
```

The two workloads use the same rule; there is no intrinsic SFT-versus-PEFT
alignment difference. Offline packing does not finalize this value
automatically, so set the derived integer explicitly in the recipe and card
command. Because changing it changes padding and pack membership, use a fresh
packing output, record the resolved value, packing-manifest hash, and actual
supervised-token count, and require fresh loss sentinels after a topology
change.

Use 8K as the default offline-pack target only when the exact model supports at
least 8192 tokens, the recipe supports offline packing, and the resolved
topology fits one MBS1 pack per data-parallel rank. Set `model.seq_length`,
`dataset.seq_length`, and `packed_sequence_size` to 8192 together. A
model-family packing opt-out, MTP incompatibility, context limit, or
demonstrated memory limit is a cohort exception, not a reason to silently fall
back.

Treat fixed-width pack padding as an execution requirement, not part of the
cross-model convergence profile. Set `pad_to_max_length=true` when the selected
dispatcher or kernel requires a fixed token width; for example, the verified
Moonlight HybridEP path requires the pack width to be divisible by its
128-token combine chunk. CUDA graphs also require fixed token width and
`pad_cu_seqlens=true` plus packing metadata. Otherwise do not require
fixed-width padding universally. Any padded tail must retain a zero loss mask.

The equal-token rule is:

```text
token_slots_per_step = packed_sequence_size * global_batch_size
```

Moving from 2K/GBS32 to 8K/GBS8 preserves 65,536 token slots per optimizer step
and can reduce gradient accumulation while presenting more source sequences in
each physical MBS1 pack. It still changes truncation and packing membership, so
it is a convergence-contract migration that requires fresh loss sentinels.
Because offline packing requires MBS1, require `GBS % DP == 0` and `GBS >= DP`.
For the v2 GBS8 target, select a model-appropriate topology with DP in
`{1, 2, 4, 8}`; do not reuse a DP16 layout.

The public training runner applies `--dataset` after constructing the model
recipe and replaces the recipe's dataset object with the selected preset.
Consequently, a card command that uses `--dataset tulu3` must explicitly pin
the dataset revision, split, data-order/packing seed, offline-packing enablement,
and the derived `+dataset.offline_packing_specs.pad_seq_to_mult`; recipe-level
dataset defaults alone do not freeze the resolved CLI workload. When the
execution requires fixed-width packs, also pin
`dataset.dataset_kwargs={pad_to_max_length:true}` and, for CUDA graphs,
`+dataset.offline_packing_specs.pad_cu_seqlens=true`. Use a fresh
`dataset.hf_output_root` (or force a deliberate rewrite) whenever sequence
length, packing width, alignment, or any of these fields changes, and audit the
final `ConfigContainer` after all overrides and runtime synchronization.

Before launching training, create the parent directory named by
`logger.save_config_filepath` and require the post-setup file to persist.
Treat only that saved, post-synchronization `ConfigContainer` as runtime-config
evidence. YAML printed by the recipe runner before setup is launch-time
configuration and may differ after model finalization; never relabel it as
resolved runtime evidence or combine it with another run's logs. If the file
does not persist, fix the output path and rerun the workload from a fresh root.

Treat the token counts above as token slots. For SFT and PEFT, also record the
actual supervised-token count after label masking; do not present padded or
masked token slots as supervised tokens.

Use these accumulation constraints for the reference executions:

| Workload | Topology constraint | DP | Gradient accumulation |
| --- | --- | ---: | ---: |
| Pretrain | 16 GPUs, TP1/PP1/CP1/EP16 | 16 | 64 |
| Full SFT | Model-appropriate topology with DP dividing GBS8 | 1, 2, 4, or 8 | `8 / DP` |
| PEFT | Model-appropriate topology with DP dividing GBS8 | 1, 2, 4, or 8 | `8 / DP` |

Topology, DP, and gradient accumulation are execution fingerprints rather than
convergence constraints. They may change while GBS8 remains fixed. Every new
execution layout must pass fresh loss sentinels, and its step-by-step values
are not strictly numerically comparable with another layout. The current
offline-packed SFT implementation requires MBS1; treat that as an
implementation limit, not a convergence rule.

For the Qwen anchor, record 128 experts, top-8 post-softmax-normalized routing,
auxiliary load-balancing loss coefficient `1e-3`, natural routing, and no
forced balancing or token dropping. These are model-native identity fields,
not universal cross-model overrides. Keep another model's native expert count,
top-k, router objective, auxiliary loss, capacity, and dropout values, record
them in its convergence fingerprint, and never alter them merely to imitate
Qwen. Require natural routing and prohibit benchmark-only forced balancing in
all convergence cohorts.

The Qwen PEFT anchor names its fused attention projections `linear_qkv` and
`linear_proj`. Preserve the same semantic adapter scope on architectures that
split Q, K, and V: list every model-native attention projection explicitly and
record the names as a model-specific fingerprint. For Moonlight MLA this is
`linear_q_proj`, `linear_kv_down_proj`, `linear_kv_up_proj`, and
`linear_proj`. Never retain a nonexistent fused-module name merely to make the
textual configuration look identical; verify that every declared target
actually matches modules before accepting PEFT evidence.

Use the same numerical value for global batch size within a comparison cohort.
If a model cannot use that value, change and validate its recipe separately,
record the exception, and treat the result as support verification rather than
an apples-to-apples convergence comparison. Compare progress at equal processed
token counts as well as equal optimizer steps. Different model architectures
and tokenizers make absolute cross-model loss values non-comparable; the shared
contract supports comparisons of stability and loss trend, not a ranking by
final loss.

Pin the same raw-document selection across models, then tokenize it with each
model's verified tokenizer unless a deliberately shared tokenizer is part of
the cohort. Do not assume that one indexed token-ID prefix represents the same
text under different tokenizers. The Qwen anchor's pinned tokenizer revision
may reproduce the Qwen run, but do not silently reuse its token IDs for another
model family. Compare cross-model stability and trend, never absolute loss.

Make every bounded convergence override explicit in the card command and apply
the same cohort values across models; never tune these values merely to improve
throughput. Library recipes own global and micro batch size. If a recipe batch
disagrees with the cohort contract, update and validate the recipe separately
instead of overriding it in the card. A canonical `perf_recipes` benchmark may
intentionally use mock data, forced balancing, or a different batch and is not
convergence evidence.

Keep long-context SFT outside `qwen3_30b_a3b_convergence_v2`. Sequence length,
CP, packing, batch construction, LR, and horizon define a separate convergence
cohort even when the starting checkpoint and dataset are shared.

### 5. Apply the verification gates

Mark an item `verified` only after the workload represented by its command has
completed with the recorded model, recipe, data, and checkpoint arguments and
the concrete expected result has been checked. A successful detached Slurm
submission is not completion; wait for the submitted workload and inspect its
result. Private executor configuration stays outside the card.

- **Conversion:** Test CPU and GPU import/export separately. Reload every
  output. Require exact keys, shapes, dtypes, and values when the conversion is
  expected to be lossless; otherwise state the numerical tolerance. Do not use
  `--detach` or a dry-run flag in a verified conversion command.
- **Manual forward pass:** Compare Hugging Face and Megatron logits on the same
  prompt with `examples/conversion/compare_hf_and_megatron/compare.py`. Record
  whether the next token matches, the cosine similarity, and the maximum and
  mean absolute logit differences. For new evidence, pass `--hf-revision` with
  the exact `model.hf_revision` so the command itself is reproducibly pinned.
  Historical evidence verified before 2026-07-20, when the helper gained
  explicit revision pinning, may remain verified without a rerun when its
  clean-run provenance is tied to the card's immutable `model.hf_revision`;
  state this grandfathering explicitly in `expected_result`. Do not use the
  exception for unverified items or evidence dated 2026-07-20 and later.
  Mark the item verified when the next token matches and cosine similarity is
  at least 0.99 (cosine distance at most 1%). Numeric maximum and mean absolute
  differences are required diagnostic observations, but are report-only and
  must not guard the item status. This gate establishes functional logit
  correlation, not strict numerical equality. Keep this result separate from
  generation. Choose a prompt whose tokenized length is divisible by TP so the
  helper does not append padding before selecting the compared next-token
  position.
- **Megatron inference:** Disable sampling, run one deterministic greedy
  generation with an exact token count, and record the literal completion
  including whitespace. A second replay may help diagnose nondeterminism, but
  it is not required verification evidence.
- **Pretrain:** Use a bounded public dataset description and a stable schedule.
  Save a middle and final checkpoint when resume is in scope. For expensive
  workloads, a 100-step reference with checkpoints at steps 50 and 100 is a
  suitable support-verification run when it crosses the peak learning rate and
  completes the configured decay; resume directly from step 50 through step
  100. This verifies bounded training and resume behavior, not full convergence.
- **SFT and PEFT:** Prefer about 100 optimizer steps with warmup and full-horizon
  decay. Use a public dataset name or preset, not its storage location. Save the
  final full-SFT checkpoint when export verification is in scope.
- **SFT export and inference:** Depend on `sft`, export its final full-model
  checkpoint to HF, reload the exported model with Transformers, and run one
  deterministic greedy generation. Store this item as an ordered `commands`
  list containing exactly two strings: the synchronous Slurm export first and
  the `uv run` HF inference second. Specify an exact new-token count and record
  the literal completion, including whitespace, in `expected_result`.
- **Long-context SFT:** Verify sequence packing and CP together. Record CP only
  when its size is greater than one.
- **Checkpoint resume:** Depend on `pretrain`; load its middle checkpoint
  directly, resume into a distinct new output root, load optimizer and RNG
  state, and compare the first resumed and final steps with the uninterrupted
  reference. For each declared loss sentinel, require this bound:
  `abs(resumed - reference) <= 1e-6 + 0.01 * abs(reference)`. Tighter
  model-specific tolerances are allowed. Do not repeat the pre-checkpoint
  training segment.
- **Performance (when present):** Use the exact canonical public performance
  recipe. Keep its bounded mock-data run separate from the real-data functional
  run and state public hardware plus thresholds.

Before adding checkpoint overrides, inspect the selected recipe and its
inherited checkpoint defaults. Keep only values that change the effective
behavior, such as an explicit load/save root, save interval, resume step, or
intentional strictness. For resume, the effective configuration must load and
save optimizer and RNG state and must not use finetune mode, but do not restate
those values when the recipe already inherits the required defaults.

Submission is not success. Require the workload process to finish successfully,
the exact optimizer-step set to be present, finite losses and performance
values, zero skipped/NaN iterations, and complete reloadable artifacts.
For training items, also require the saved post-setup `ConfigContainer`; a
launch-time config dump or dry run cannot replace it.

For ordered command lists, wait for the preceding workload to finish and verify
its artifact before starting the next command. Reference and resumed checkpoint
roots must be distinct; the resumed root must be new or empty. Use the same
Bridge commit, public base container, accelerator topology, and convergence
fingerprint as the reference run. Keep the execution fingerprint identical
when possible. Record any necessary execution-only deviation, explain why it
preserves semantics, and require the resume loss sentinels to pass.

Use the batch sizes defined by the selected recipe. A model verification card
must not override global or micro batch size. If a recipe batch size is wrong,
change and validate the recipe separately instead of hiding the change in the
card command.

### 6. Record training metrics consistently

For every verified training item, record:

- `initial_loss`: loss at the first optimizer step executed by that item;
- `final_loss`: loss at its final optimizer step;
- `last_10_steps_step_time_ms_avg`: arithmetic mean over the final 10 executed
  optimizer steps;
- `last_10_steps_model_tflops_per_gpu_avg`: arithmetic mean over the same rows.

Parse all fields from each complete keyed optimizer-step line. Do not collect
loss, time, and throughput independently and zip them. Reject missing,
duplicate, skipped, NaN, or non-finite rows rather than excluding them.
Every verified training item must execute at least 10 optimizer steps so this
window is complete.

For a resume, use the first optimizer step after the selected checkpoint as
`initial_loss`, the final executed optimizer step as `final_loss`, and average
the final 10 resumed steps. For example, a step-50-to-100 continuation uses
step 51, step 100, and the average over steps 91-100.

For Megatron-indexed data, command values are prefixes and omit `.bin` and
`.idx`. A bounded RedPajama2 prefix such as `head_01` is suitable for a
reproducible functional run; keep the physical dataset root private.

### 7. Record only important enabled features

Use `enabled_features` only on pretrain, SFT, long-context SFT, and PEFT. Keep
it empty when none of these are central to the verification.

| Key | Allowed value |
| --- | --- |
| `sequence_packing` | `offline` or `in_batch` |
| `cuda_graph.implementation` | `local` or `transformer_engine` |
| `cuda_graph.scopes` | `full_iteration`, `attn`, `mlp`, `moe`, `moe_router`, `moe_preprocess`, or `mamba` |
| `context_parallel_size` | integer greater than one |
| `moe_dispatcher` | `deepep` or `hybridep` |

Do not list routine TP/PP/DP sizes, Transformer Engine, fused loss,
distributed optimizer, ordinary communication overlap, LoRA, or DoRA.

### 8. Validate before review

Run:

```bash
uv run --no-project --with pyyaml python \
  skills/create-model-verification-card/scripts/validate_card.py \
  examples/model_verification_cards/<model-slug>/card.yaml
```

When private terminology is known only at runtime, add:

```bash
--denylist "$PRIVATE_DENYLIST"
```

Then parse the YAML, run relevant targeted tests, run
`uv run pre-commit run --all-files`, and inspect the complete diff. Do not mark
an item verified merely to make validation pass.

## Completion checklist

- Keep all twelve core inventory items and use only the four statuses. Include
  `pretrain_performance` only when the exact variant has a canonical public
  performance recipe.
- Start the summary with the exact untuned performance disclaimer unless at
  least one concrete `pretrain_performance` hardware leaf exists; never use an
  `all` placeholder, and scope any tuned claim to the exact concrete leaf.
- Put the verified workload precision on every direct item or hardware leaf;
  use `fp8_mx` and `nvfp4` only for training leaves that ran in those modes.
- Pin a public immutable HF revision, minimum Transformers version, public base
  container, and exact Bridge verification commit; use an item override only
  for a verified workload run from a different clean commit.
- Use the public model name in commands.
- Include commands and concrete expected results for verified items.
- For manual forward pass, require a next-token match and cosine similarity of
  at least 0.99, and record numeric max/mean absolute logit differences without
  guarding on them. New evidence must pass the exact `model.hf_revision`
  through `--hf-revision`; retain older unpinned evidence only under the
  explicitly documented grandfathering rule above.
- Use `convert.sh --executor slurm` for conversion and `train.sh` for training.
- Keep private executor wiring out of commands: no mounts, environment
  forwarding, concrete accounts/partitions/images, or remote-launch setup.
- Put every training result under its canonical public hardware key and include
  all four metrics for each verified training leaf; never record a private
  cluster name or retain the old `gpu_type` field.
- Keep `verification_index` synchronized with `items`, omit empty status
  buckets, list every item name explicitly, and never use the scalar `all` in
  the index.
- Audit the resolved convergence contract before each training run; align it
  with `qwen3_30b_a3b_convergence_v2` or record the exception and classify the
  result as support verification rather than cross-model convergence evidence.
- Change only the execution/performance contract while tuning throughput, and
  recheck loss sentinels after numerically non-bitwise changes.
- Leave recipe global and micro batch sizes unchanged in card commands.
- Save full SFT, export it to HF, and record an exact deterministic N-token HF
  completion in a two-command ordered list.
- Keep resume as one direct continuation from the pretrain checkpoint.
- Keep enabled features within the four-family allowlist.
- Pass the bundled validator, including any caller-supplied denylist.
- Confirm the card, commit, and PR contain no private runtime information.
