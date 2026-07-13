---
name: verl-e2e-testing
description: External verl end-to-end validation workflow for Megatron-Bridge changes. Covers running a small verl Megatron backend job from a Bridge checkout, choosing LoRA/DDP plus optional save/resume and parallelism variants, setting PYTHONPATH so verl imports the local Bridge tree, and reporting pass/fail evidence.
when_to_use: Changing Megatron-Bridge code that needs downstream verl compatibility validation; checking non-vanilla Bridge paths; testing PEFT/LoRA, DDP, checkpoint behavior, or explicitly requested advanced variants through verl; 'does this work in verl', 'run verl e2e', 'external RL loop validation'.
---

# verl E2E Testing

Validate a Megatron-Bridge change through verl's Megatron backend. This catches integration issues that Bridge-only tests miss: provider/configuration finalization, HF import through Bridge, PEFT wrapping, DDP wrapping, optimizer setup, rollout/ref wiring, and checkpoint ownership by an external RL loop.

Use this as an external compatibility smoke test after the relevant Bridge unit and functional tests are green.

This is not a replacement for Bridge correctness tests. The default verl run proves that the changed Bridge path can survive an external RL training loop; numerical parity, import/export fidelity, and feature-specific behavior still come from focused Bridge tests.

## Scope

Think in coverage levels. Start with Level 0 and add only the levels justified by the change.

| Level | Required when | What it proves |
|---|---|---|
| 0: LoRA + DDP smoke | Any Bridge change that claims verl compatibility | verl can import the local Bridge checkout, apply PEFT, wrap with Megatron DDP, build optimizer state, run rollout/ref/critic wiring, and finish one training step |
| 1: Save/resume | PEFT, checkpointing, HF export, adapter export, optimizer state, or resume behavior changed | verl-owned checkpoint scheduling can save and reload Bridge-built model state |
| 2: Parallelism stress | Provider finalization, mpu-derived settings, TP/PP/CP/EP, sequence parallel, or dispatcher behavior changed | provider settings remain correct under non-trivial Megatron parallel state |
| 3: Optional Megatron-FSDP | Only when downstream explicitly asks for verl Megatron-FSDP coverage or the change directly touches that integration path | the same provider works when verl selects Megatron-FSDP instead of DDP |
| 4: Feature-specific e2e | The change depends on runtime behavior not exercised by the default text-only smoke | the requested feature is activated by a targeted verl command or script |
| 5: Convergence / learning signal | Optimizer, scheduler, loss, reward, PEFT trainability, gradient flow, or training stability changed | metrics move in the expected direction over a short run and do not silently produce zero/NaN/unstable updates |

The default Level 0 target is a short, non-vanilla Bridge run in verl with LoRA enabled and Megatron DDP selected:

```bash
USE_MBRIDGE=True
VANILLA_MBRIDGE=False
VALUE_VANILLA_MBRIDGE=False
LORA_RANK=4
USE_MEGATRON_FSDP=False
TOTAL_TRAIN_STEPS=1
```

This is intentionally small. It exercises the Bridge-facing path in verl without making Megatron-Bridge own rollout scheduling, reward handling, optimizer scheduling, or checkpoint orchestration.

Level 0 is not a convergence test. It only proves the training loop can complete one update. Use Level 5 when the question is whether the run shows a learning signal under verl.

Megatron-FSDP is not part of the default validation expected for current provider compatibility work. Run it only for Level 3 coverage when FSDP is explicitly in scope:

```bash
USE_MEGATRON_FSDP=True
ALL_OFFLOAD=False
COMMON_PP=1
COMMON_VPP=null
COMMON_CP=1
COMMON_TP=1
INFER_TP=1
```

## Repos

Use explicit repo variables. Do not rely on an installed `megatron-bridge` wheel; the purpose is to test the current Bridge checkout.

Use the upstream verl repository as the default source:

```text
https://github.com/verl-project/verl
```

If a checkout is not already available, clone it next to the Bridge checkout or into the site's standard workspace:

```bash
git clone https://github.com/verl-project/verl.git /path/to/verl
```

```bash
export BRIDGE_REPO=${BRIDGE_REPO:-/path/to/Megatron-Bridge}
export VERL_REPO=${VERL_REPO:-/path/to/verl}
export PYTHONPATH="${BRIDGE_REPO}/src:${BRIDGE_REPO}/3rdparty/Megatron-LM:${VERL_REPO}:${PYTHONPATH:-}"
```

Before running, record both states:

```bash
git -C "$BRIDGE_REPO" status --short
git -C "$VERL_REPO" status --short
git -C "$BRIDGE_REPO" rev-parse --short HEAD
git -C "$VERL_REPO" rev-parse --short HEAD
```

If testing on a remote GPU machine, sync the exact local changes first. Do not reset or overwrite unrelated changes in either tree.

Verify that Python imports the checkout under test:

```bash
python - <<'PY'
import megatron.bridge
print(megatron.bridge.__file__)
PY
```

The printed path must live under `$BRIDGE_REPO/src`. If it points at site-packages, fix `PYTHONPATH` before trusting any result.

When running against an existing Ray cluster, the driver import is not enough. Ray workers may not inherit the shell `PYTHONPATH`, and the run can fail later with `ModuleNotFoundError: No module named 'megatron.bridge'` even though the driver import passed. Verify a worker import:

```bash
python - <<'PY'
import os
import ray

ray.init(address="auto", runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}})

@ray.remote
def bridge_path():
    import megatron.bridge
    return megatron.bridge.__file__

print(ray.get(bridge_path.remote()))
PY
```

If the worker import fails, pass the checkout path through Ray's runtime environment when launching the wrapper:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}"
```

If this import fails before Bridge-managed construction, fix the runtime environment first. The official verl image may not contain every Bridge dependency; for example, Bridge imports `modelopt` through `AutoBridge`, so a missing `nvidia-modelopt` can fail the smoke before verl exercises the changed path:

```bash
python -m pip show nvidia-modelopt || \
  python -m pip install --extra-index-url https://pypi.nvidia.com nvidia-modelopt
```

Treat ad-hoc installs as container setup evidence, not repository changes. If the image lacks `uv`, run focused Bridge unit tests in a Bridge development environment instead of forcing them through the verl container.

## Checkpoint Or Config Choice

Prefer the smallest representative HF checkpoint or local converted checkpoint that exercises the changed Bridge path. Start small before testing larger variants.

If there is no practical checkpoint for the changed path, use verl's dummy-config path with a minimal HF config:

```bash
USE_DUMMY_MODEL=True
DUMMY_MODEL_CONFIG_PATH=/path/to/minimal_config.json
MODEL_ID=<org>/<representative-model-name>
```

Report dummy-config results carefully: they validate construction and training mechanics, not pretrained weight compatibility.

The default text-only smoke does not activate every feature. If the change depends on a specialized runtime path, use the closest maintained verl example or project script that actually enables that path, and record the exact environment variables and Hydra overrides.

For distributed behavior, a Level 0 run with all parallel sizes set to 1 is not enough evidence. Add Level 2 coverage when the change touches tensor, pipeline, context, expert, sequence, dispatcher, or mpu-derived settings.

When a script already exists for the requested workflow, prefer it over reconstructing a long command by hand. Verify the log shows the requested RL algorithm and backend rather than inferring it from the script name.

## Bridge Checks First

Run focused Bridge tests before the external verl e2e. Include any change-specific tests added by the PR.

```bash
cd "$BRIDGE_REPO"
uv run python -m pytest -q \
  tests/unit_tests/models/test_model_provider_mixin.py \
  tests/unit_tests/models/test_param_mapping.py \
  tests/unit_tests/training/test_integration.py \
  <change-specific-test-paths>
```

For changes that affect import/export, also run the relevant conversion, roundtrip, or parity test from the PR.

Minimum Bridge-side evidence for a new verl-facing Bridge path:

- provider/config unit tests
- parameter mapping tests
- HF to Megatron import or roundtrip on a small checkpoint or config
- change-specific runtime or parity check when available
- this verl external-loop smoke after the above pass

## verl Data Setup

verl's default Megatron smoke wrapper expects task parquet files. Prepare the default dataset once from the verl checkout if it is missing:

```bash
cd "$VERL_REPO"
export PYTHONPATH="$VERL_REPO:${PYTHONPATH:-}"
python3 examples/data_preprocess/gsm8k.py \
  --local_save_dir "${GSM8K_DIR:-$HOME/data/gsm8k}"
```

Use `--local_dataset_path "$GSM8K_SOURCE_DIR"` only when that raw local dataset path actually exists. Otherwise let `datasets` fetch `openai/gsm8k`.

Set explicit paths when running in a container or shared filesystem:

```bash
export TRAIN_FILES=/path/to/gsm8k/train.parquet
export VAL_FILES=/path/to/gsm8k/test.parquet
```

The wrapper also enables a reward model by default. Ensure the default reward model path exists, or set:

```bash
export RM_MODEL_PATH=/path/to/local/reward/model
```

For a Level 0 rule-reward smoke, it is acceptable to disable the reward-model rollout when no local reward model is available:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  reward.reward_model.enable=False
```

Report this as a limitation; it still tests Bridge actor/ref/critic construction, LoRA, DDP wrapping, rollout, and one training update, but not reward-model serving.

## Minimal verl Run

Use verl's maintained wrapper, or the closest maintained example script for the requested workflow, rather than constructing a long Hydra command manually:

```bash
cd "$VERL_REPO"
ray stop --force || true

export MODEL_ID=<small-compatible-hf-model>
export TRAIN_FILES=${TRAIN_FILES:-/path/to/gsm8k/train.parquet}
export VAL_FILES=${VAL_FILES:-/path/to/gsm8k/test.parquet}
export RM_MODEL_PATH=${RM_MODEL_PATH:-/path/to/local/reward/model}
export ENGINE=vllm
export USE_MBRIDGE=True
export VANILLA_MBRIDGE=False
export VALUE_VANILLA_MBRIDGE=False
export LORA_RANK=4
export USE_MEGATRON_FSDP=False
export COMMON_PP=1
export COMMON_VPP=null
export COMMON_CP=1
export COMMON_TP=1
export INFER_TP=1
export ALL_OFFLOAD=False
export TOTAL_TRAIN_STEPS=1
export SAVE_FREQ=-1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1

bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Use `MODEL_ID` when the checkpoint is available through the wrapper's default cache layout. Add `MODEL_PATH=/path/to/local/hf/model` only when testing a local or converted checkpoint.

When `$HOME` is small or shared slowly, put HF caches and downloaded checkpoints on a larger shared or node-local scratch path and pass `MODEL_PATH` explicitly. Pre-download large models once in the same container environment to avoid Ray workers racing the cache:

```bash
export HF_HOME=${HF_HOME:-/scratch/$USER/verl_hf}
export HF_HUB_CACHE=$HF_HOME/hub
MODEL_PATH=${MODEL_PATH:-/scratch/$USER/models/<org>/<model>}
hf download <org>/<model> --local-dir "$MODEL_PATH"
```

Capture logs to a file for review:

```bash
mkdir -p "${LOG_DIR:-$PWD/verl_e2e_logs}"
LOG_FILE="${LOG_DIR:-$PWD/verl_e2e_logs}/verl_lora_ddp_$(date +%Y%m%d_%H%M%S).log"
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  "++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${PYTHONPATH}" \
  2>&1 | tee "$LOG_FILE"
grep -E "Training Progress|VANILLA_MBRIDGE|Traceback|RuntimeError|KeyError|ValueError" "$LOG_FILE"
```

Prefer a saved log over a pasted terminal excerpt in PR descriptions.

Do not trust the command shape alone. Inspect the log or resolved config to confirm the requested algorithm, rollout backend, non-vanilla Bridge path, response length, checkpoint policy, and parallelism settings are actually active.

For time-limited smoke runs where vLLM CUDA graph capture dominates setup and the goal is Bridge provider validation, it is acceptable to add:

```bash
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  actor_rollout_ref.rollout.enforce_eager=True
```

Report this override as a limitation. It still validates Bridge import, HF import, LoRA, Megatron DDP, rollout wiring, and one training update, but not vLLM CUDA graph capture.

## Save/Resume Coverage

After the minimal run passes, add checkpoint coverage if the change touches PEFT, checkpointing, export, or optimizer state:

```bash
# Save once.
SAVE_FREQ=1 TOTAL_TRAIN_STEPS=1 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh

# Resume and train one more step.
RESUME_MODE=auto SAVE_FREQ=1 TOTAL_TRAIN_STEPS=2 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Remove stale verl `checkpoints/` output between unrelated experiments. Keep it for resume validation.

If save/load fails, first confirm whether verl expected keys from an original checkpoint that the Bridge export intentionally did not regenerate. Missing sibling tensors, stale source-key maps, or intentionally dropped auxiliary keys should be handled explicitly rather than ignored by a broad non-strict save.

## Parallelism Stress

Use Level 2 when the provider reads or mutates parallelism-related fields, or when the change touches `provider.configure(...)`, Megatron `mpu`, sequence parallel, context parallel, dispatcher behavior, or tensor/expert tensor parallel settings.

The variants below assume the Level 0 exports above are still in the shell; each command overrides only the values being tested.

Example dense stress variant:

```bash
COMMON_TP=2 \
COMMON_PP=2 \
COMMON_VPP=null \
COMMON_CP=1 \
INFER_TP=2 \
USE_MEGATRON_FSDP=False \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Example routed/expert stress variant, only when the checkpoint and change support it:

```bash
COMMON_EP=2 \
COMMON_ETP=1 \
ROUTING_REPLAY_MODE=disabled \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

Keep these as follow-up runs. Do not make them the first debugging surface for a new provider.

## Optional Megatron-FSDP Variant

Use Level 3 after Level 0 passes only when downstream explicitly requests Megatron-FSDP coverage or the Bridge change directly touches FSDP wrapping, sharding, checkpoint format, or distributed optimizer behavior:

```bash
USE_MEGATRON_FSDP=True \
ALL_OFFLOAD=False \
COMMON_PP=1 \
COMMON_VPP=null \
COMMON_CP=1 \
COMMON_TP=1 \
INFER_TP=1 \
bash tests/special_e2e/run_ppo_trainer_megatron.sh \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
  ++critic.megatron.override_transformer_config.gradient_accumulation_fusion=False
```

For Bridge-native FSDP behavior and constraints, also read @skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md.

## Convergence / Learning Signal

Use Level 5 only when the change affects trainability or when downstream validation explicitly asks for convergence. Do not require it for every provider-only PR; RL convergence is slower, noisier, and more environment-dependent than the compatibility smoke.

The goal is a short learning-signal run, not a full benchmark. Prefer a small representative checkpoint, fixed data, fixed seed when available, and enough steps to observe non-random metric movement:

```bash
TOTAL_TRAIN_STEPS=20 \
SAVE_FREQ=-1 \
VAL_BEFORE_TRAIN=True \
TEST_FREQ=10 \
LORA_RANK=4 \
USE_MBRIDGE=True \
VANILLA_MBRIDGE=False \
VALUE_VANILLA_MBRIDGE=False \
USE_MEGATRON_FSDP=False \
ENGINE=vllm \
bash tests/special_e2e/run_ppo_trainer_megatron.sh
```

For a stronger signal, run 50-100 steps if GPU time allows. Keep rollout, reward model, dataset, batch sizes, and model checkpoint fixed between baseline and candidate runs.

Acceptable convergence evidence depends on the task, but the report should include at least:

- no NaNs or infs in loss, reward, KL, entropy, grad norm, or logprob metrics
- nonzero trainable parameter count when PEFT is enabled
- actor/critic losses and reward-related metrics logged for multiple steps
- validation or reward trend compared against the starting point or a known-good baseline
- no repeated zero gradients, frozen LoRA adapters, or constant logprobs unless expected

Treat flat or saturated metrics as a real signal to triage. Constant rewards, constant critic values, repeated cap-hit response lengths, or near-identical rollout logprobs can indicate data, reward, rollout, masking, or Bridge-side integration problems.

For response-generation changes, verify the configured max response length is long enough for the dataset and reward. If many samples hit the cap, the learning-signal run is not strong evidence; increase context/response limits or adjust batch sizing before interpreting metrics.

Do not call a 20-step run "converged" in the benchmark sense. Call it "learning-signal passed" unless it reaches a pre-agreed metric threshold.

## Container Image

Use the official verl Docker images as the default source:

```text
https://hub.docker.com/r/verlai/verl
```

For this skill's default smoke path, pick a vLLM-flavored `verlai/verl` image tag unless the test intentionally changes the rollout engine. The maintained wrapper defaults to vLLM, and the command should make that explicit with:

```bash
ENGINE=vllm
```

Avoid using sglang, TRT-LLM, or generic images for the default Level 0 run unless the point of the test is to validate that rollout backend. A backend-specific image can fail before Bridge model construction, which makes the result a poor signal for a Megatron-Bridge provider change.

Pin the exact image tag in the test log or PR description:

```bash
export VERL_IMAGE=${VERL_IMAGE:-verlai/verl:<vllm-compatible-tag>}
```

If the cluster requires Enroot/SquashFS images, convert or mirror the selected `verlai/verl` tag through the site's normal process, but keep the source tag visible in the report.

## Slurm or Container Runs

Use the cluster's standard container and mount both checkouts into the container. Keep setup and the actual training run in the same container step when using node-local paths such as `/tmp`; node-local model caches and ad-hoc pip installs disappear when a fresh container step starts. Keep paths generic in scripts committed to Megatron-Bridge:

```bash
export VERL_IMAGE=${VERL_IMAGE:-verlai/verl:<vllm-compatible-tag>}

srun <site-specific-slurm-options> \
  --container-image="${VERL_IMAGE}" \
  --container-mounts="${BRIDGE_REPO}:/workspace/Megatron-Bridge,${VERL_REPO}:/workspace/verl,<data-root>:<data-root>" \
  --container-workdir=/workspace/verl \
  bash -lc '
    export BRIDGE_REPO=/workspace/Megatron-Bridge
    export VERL_REPO=/workspace/verl
    export PYTHONPATH=$BRIDGE_REPO/src:$BRIDGE_REPO/3rdparty/Megatron-LM:$VERL_REPO
    ray stop --force || true
    MODEL_ID=<small-compatible-hf-model> \
    ENGINE=vllm \
    USE_MBRIDGE=True VANILLA_MBRIDGE=False VALUE_VANILLA_MBRIDGE=False \
    LORA_RANK=4 USE_MEGATRON_FSDP=False TOTAL_TRAIN_STEPS=1 SAVE_FREQ=-1 \
    bash tests/special_e2e/run_ppo_trainer_megatron.sh
  '
```

Use a persistent log directory on a shared filesystem or `$HOME` for long Slurm jobs. Logs written only to node-local `/tmp` can disappear when the allocation expires or is canceled. If a cluster attach helper runs the command through `srun --pty`, do not background the workload inside that attached shell; the step cleanup can terminate it immediately. To detach safely, background the attach helper itself from the login node:

```bash
mkdir -p "$HOME/verl_e2e_logs"
nohup env COMMAND='bash /path/to/run_verl_e2e.sh' \
  bash /path/to/<jobid>-attach.sh \
  > "$HOME/verl_e2e_logs/attach_driver_$(date +%Y%m%d_%H%M%S).out" 2>&1 &
```

If an attach helper enters a container that no longer sees the expected checkouts or log directory, treat that helper as stale. Start a fresh `srun` step against the existing allocation with explicit `--container-image`, `--container-mounts`, and `--container-workdir`.

On CUDA/H100 clusters, some launchers set both `CUDA_VISIBLE_DEVICES` and ROCm variables such as `ROCR_VISIBLE_DEVICES`. If verl workers fail before model construction with `Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set`, fix the launcher/container environment or apply a temporary local verl workaround that drops `ROCR_VISIBLE_DEVICES` when CUDA is already set. Do not report this as a Bridge provider failure.

Keep GPU usage proportional to the evidence needed. For OOMs, first reduce per-GPU load or adjust TP/PP/CP, micro-batches, rollout batch size, and response length tradeoffs. Increase node count only as far as needed for the requested validation, and record why the chosen scale was necessary.

For general Slurm debugging and multi-node patterns, read @skills/nemo-mbridge-multi-node-slurm/SKILL.md.

## Pass Criteria

A useful pass has all of the following:

- Focused Bridge tests pass for provider/config/mapping behavior.
- verl uses the local Bridge checkout through `PYTHONPATH`.
- The verl log shows `VANILLA_MBRIDGE=False`.
- One training step reaches completion, for example `Training Progress: 100%|1/1|`.
- No exception occurs during Bridge provider setup, HF import, LoRA wrapping, DDP wrapping, optional FSDP wrapping when enabled, optimizer setup, checkpoint manager setup, or the training step.

Ray shutdown, Python resource-tracker warnings, or post-completion DataLoader worker termination can be acceptable if the requested training step completed, metrics for the expected global step were logged, and the process exits successfully. Mention them as residual log noise.

Do not overclaim coverage if the run used a dummy config, a generic dataset that does not activate the changed feature, trivial parallelism for a distributed change, or disabled save/resume for a checkpointing change. Call it the exact level that passed.

Do not claim convergence from Level 0. Claim convergence only from Level 5, and distinguish "learning signal" from "benchmark convergence" in the report.

## Failure Triage

If construction fails, check whether the Bridge provider is finalized with the same parallel sizes that verl initialized through Megatron `mpu`.

If LoRA fails, check target module names and whether the provider path uses the non-vanilla Bridge PEFT helpers expected by verl.

If checkpoint save/load fails, first rerun without save/resume (`SAVE_FREQ=-1`) to separate model construction from checkpoint behavior.

If rollout fails before actor construction, this may be a verl rollout engine issue rather than a Bridge provider issue. Report the boundary clearly.

If the log shows the wrong Bridge path, stop. Any later failure or pass is not evidence for the local Bridge change.

If the baseline fails before Bridge-managed construction because of data, reward model, Ray, vLLM, or package setup, fix the environment first and do not report it as a Bridge failure.

If model download fails with `No space left on device`, move `HF_HOME`, `HF_HUB_CACHE`, and `MODEL_PATH` to a larger shared or node-local path, then rerun with the explicit local `MODEL_PATH`.

## Summary Format

End every run with a short user-facing summary that answers "Did the requested deliverables pass?" before adding details. Use `Pass`, `Fail`, `Skipped`, or `Blocked` for each deliverable, and do not report an overall `Pass` unless the pass criteria for the requested coverage level were met.

```text
Result: <Pass/Fail/Blocked> - <one sentence stating what was validated>
Requested coverage: <Level 0/1/2/3/4/5 and requested variants>
Checkpoint/config: <MODEL_ID, MODEL_PATH, or dummy config>

Deliverables:
- Bridge-side checks: <Pass/Fail/Skipped> - <test command or skipped reason>
- Local Bridge import in verl: <Pass/Fail> - <PYTHONPATH or imported Bridge path>
- verl Megatron backend run: <Pass/Fail/Skipped> - <LoRA + DDP or requested variant>
- Requested variants: <Pass/Fail/Skipped/Not requested> - <save/resume, parallelism stress, Megatron-FSDP, feature-specific run, or learning-signal/convergence>
- Log capture: <Pass/Fail> - <log path>

Evidence:
- Bridge repo: <commit> plus dirty files
- verl repo: <commit> plus dirty files
- Command: <exact command or script path>
- Key lines: <requested algorithm/backend, VANILLA_MBRIDGE=False, Training Progress completion, expected global-step metrics, or the first relevant error>

Limitations:
- <dummy config, skipped save/resume, trivial parallelism, generic data that did not activate a feature, no convergence claim, known shutdown warnings, etc.>

Follow-ups:
- <needed rerun, environment fix, provider fix, or "none">
```

If the job is blocked before Bridge-managed construction by data, reward model, Ray, vLLM, dependency, disk, or cluster setup, mark the overall result as `Blocked`, not `Fail`, and state that it is not evidence against the Bridge change.

If any requested deliverable was not run, mark it `Skipped` or `Not requested` with the reason. Do not leave it implicit in the limitations.
