---
name: create-model-verification-card
description: Create or update concise, agent-readable Megatron Bridge model verification cards. Use when adding a model support card, auditing verification coverage, recording conversion, deterministic inference, training, checkpoint resume, post-SFT export, or performance results, or preparing a model-support PR. Enforce the required core inventory, optional canonical performance item, public Slurm launcher commands, training metrics, important-feature allowlist, and a strict privacy boundary that excludes private runtime wiring, internal paths, credentials, and job metadata.
---

# Create Model Verification Card

Create `model_cards/<model-slug>/card.yaml` from public model facts and verified
commands. Keep the card small enough for an agent to scan without interpreting
logs or reconstructing the execution environment.

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

The top-level Bridge commit is the default for every item. If one verified item
was run from a different clean checkout, put that exact 40-hex commit in the
item's optional `bridge_commit` field. Omit the field when it would repeat the
top-level value, and never use a commit field to disguise uncommitted runtime
changes. Items that are not verified must not carry a commit override.

### 2. Create the core inventory and add performance when available

Include these twelve required items, even when their status is `unsupported` or
`not_applicable`:

1. CPU HF-to-Megatron conversion
2. GPU HF-to-Megatron conversion
3. CPU Megatron-to-HF conversion
4. GPU Megatron-to-HF conversion
5. Manual HF/Megatron forward-pass parity
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

Use only `unverified`, `verified`, `unsupported`, or `not_applicable`. Do not
add `smoke` or an evidence field.

For `unsupported` and `not_applicable`, leave `command` or `commands`, date,
precision, GPU type, and metrics null, then state the public product limitation
in `expected_result`.

Put a scalar `precision` on every item. It describes the workload that was
actually verified, not every precision the model might support:

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
in a model support card.

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

### 4. Apply the verification gates

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
  mean absolute logit differences. Keep this result separate from generation.
  Choose a prompt whose tokenized length is divisible by TP so the helper does
  not append padding before selecting the compared next-token position.
- **Megatron inference:** Use deterministic greedy generation, specify an exact
  token count, run twice, and record the literal completion including
  whitespace.
- **Pretrain:** Use a bounded public dataset description and a stable schedule.
  Save a middle and final checkpoint when resume is in scope.
- **SFT and PEFT:** Prefer about 100 optimizer steps with warmup and full-horizon
  decay. Use a public dataset name or preset, not its storage location. Save the
  final full-SFT checkpoint when export verification is in scope.
- **SFT export and inference:** Depend on `sft`, export its final full-model
  checkpoint to HF, reload the exported model with Transformers, and run greedy
  generation twice. Store this item as an ordered `commands` list containing
  exactly two strings: the synchronous Slurm export first and the `uv run` HF
  inference second. Specify an exact new-token count and record the literal
  byte-identical completion, including whitespace, in `expected_result`.
- **Long-context SFT:** Verify sequence packing and CP together. Record CP only
  when its size is greater than one.
- **Checkpoint resume:** Depend on `pretrain`; load its middle checkpoint
  directly, resume into a distinct new output root, load optimizer and RNG
  state, and compare the first resumed and final steps with the uninterrupted
  reference. Use at most a 1% relative loss tolerance for sentinel comparison;
  tighter model-specific tolerances are allowed. Keep a small independent
  absolute tolerance, such as 1e-6, for values near zero. Do not repeat the
  pre-checkpoint training segment.
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

For ordered command lists, wait for the preceding workload to finish and verify
its artifact before starting the next command. Reference and resumed checkpoint
roots must be distinct; the resumed root must be new or empty. Use the same
Bridge commit, public base container, accelerator topology, dataset, and
configuration as the reference run.

Use the batch sizes defined by the selected recipe. A model verification card
must not override global or micro batch size. If a recipe batch size is wrong,
change and validate the recipe separately instead of hiding the change in the
card command.

### 5. Record training metrics consistently

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

For a resume that executes steps 201-400, use step 201 as `initial_loss`, step
400 as `final_loss`, and average steps 391-400.

For Megatron-indexed data, command values are prefixes and omit `.bin` and
`.idx`. A bounded RedPajama2 prefix such as `head_01` is suitable for a
reproducible functional run; keep the physical dataset root private.

### 6. Record only important enabled features

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

### 7. Validate before review

Run:

```bash
uv run --no-project --with pyyaml python \
  skills/create-model-verification-card/scripts/validate_card.py \
  model_cards/<model-slug>/card.yaml
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
- Put the verified workload precision on every item; use `fp8_mx` and `nvfp4`
  only for training items that ran in those modes.
- Pin a public immutable HF revision, minimum Transformers version, public base
  container, and exact Bridge verification commit; use an item override only
  for a verified workload run from a different clean commit.
- Use the public model name in commands.
- Include commands and concrete expected results for verified items.
- Use `convert.sh --executor slurm` for conversion and `train.sh` for training.
- Keep private executor wiring out of commands: no mounts, environment
  forwarding, concrete accounts/partitions/images, or remote-launch setup.
- Include GPU type and all four metrics for verified training items.
- Leave recipe global and micro batch sizes unchanged in card commands.
- Save full SFT, export it to HF, and record an exact deterministic N-token HF
  completion in a two-command ordered list.
- Keep resume as one direct continuation from the pretrain checkpoint.
- Keep enabled features within the four-family allowlist.
- Pass the bundled validator, including any caller-supplied denylist.
- Confirm the card, commit, and PR contain no private runtime information.
