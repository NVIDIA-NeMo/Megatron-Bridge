# Megatron-Bridge vLLM PPO CI suites

This directory provides standalone entry points for three jobs currently defined
across
[`e2e_ppo_trainer_megatron_vllm.yml`](https://github.com/verl-project/verl/blob/main/.github/workflows/e2e_ppo_trainer_megatron_vllm.yml)
and
[`e2e_ppo_trainer_megatron_vllm_2.yml`](https://github.com/verl-project/verl/blob/main/.github/workflows/e2e_ppo_trainer_megatron_vllm_2.yml).
The existing verl source files and workflows do not need to be changed to use them.
Only variants using the official NVIDIA NeMo Megatron-Bridge backend are included.

| Existing CI job | Included Megatron-Bridge variants | Standalone entry point |
| --- | --- | --- |
| `e2e_ppo_trainer_megatron-deepseek` | LoRA, save and resume | `bash run_megatron_deepseek.sh` |
| `e2e_ppo_trainer_megatron-qwen3` | Dense | `bash run_megatron_qwen3.sh` |
| `e2e_ppo_trainer_megatron-moe-expert-parallel` | MoE EP=4, FP8 rollout, and LoRA EP=2 | `bash run_megatron_moe_expert_parallel.sh` |

Each script reproduces the relevant Megatron-Bridge command order from its original
job: dependency installation, GSM8K preprocessing, selected training variants,
checkpoint boundaries, and final checkpoint cleanup. Every training invocation
uses `USE_MBRIDGE=True` and `VANILLA_MBRIDGE=False`, directly or through an
explicit shared environment array. Run a script from the verl repository root in
an isolated CI job. Do not run multiple scripts concurrently in the same checkout,
Python environment, home directory, or set of GPUs.

## Runner requirements

- One Linux node with 8 NVIDIA L20 GPUs for each active script. The shared trainer
  config uses one node and 8 GPUs, and the expert-parallel topology consumes all 8.
- A 60-minute timeout per script, matching the existing jobs.
- The tested container image:
  `verl-ci-cn-beijing.cr.volces.com/verlai/verl:vllm020.dev1`. Its preinstalled
  CUDA, Python, PyTorch, vLLM, Ray, Megatron build dependencies, and system
  libraries are part of the test environment. Several installs use `--no-deps`,
  so a generic CUDA image is not an equivalent replacement.
- An NVIDIA host driver compatible with the image, NCCL peer communication across
  all 8 GPUs, enough host RAM and shared memory for CPU offload, and usable local
  loopback ports for Ray.
- A pip-writable Python environment plus writable repository root, `${HOME}`,
  `/tmp`, and enough storage for package/model caches, datasets, generated dummy
  models, and checkpoints. The workflows do not define numeric CPU, RAM, disk, or
  shared-memory minima; use the existing `L20x8` runner profile as the baseline.
- A checkout with full history (`fetch-depth: 0`). All commands must run from the
  repository root because requirements, configs, launchers, and `checkpoints/` use
  repository-relative paths.

## Preloaded models and data

The shared trainer script does not download models. Preload the public snapshots
at these exact paths:

```text
${HOME}/models/deepseek-ai/deepseek-coder-1.3b-instruct
${HOME}/models/Qwen/Qwen3-0.6B
${HOME}/models/Qwen/Qwen3-30B-A3B-Instruct-2507
${HOME}/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B
```

The MoE suite creates a dummy model, but still needs the Qwen3-30B snapshot for
tokenizer and base artifacts. Start with no stale
`${HOME}/dummy_models/Qwen/Qwen3-30B-A3B-Instruct-2507` directory, or validate that
it was generated from the current dummy-model config. All three suites enable the
Skywork reward model.

Make the raw GSM8K dataset loadable from `${HOME}/models/hf_data/gsm8k`. Each
script preprocesses it to:

```text
${HOME}/data/gsm8k/train.parquet
${HOME}/data/gsm8k/test.parquet
```

## Network and environment

The runner must be able to pull the tested image and reach its Python package index
and GitHub. The scripts install Megatron-Bridge and Megatron-LM at run time.
Because the requirements files and the ModelOpt lower bound are not all immutable
pins, retain the exact image and capture `pip3 list` when investigating a
reproducibility failure.

No external service container is needed. Ray runs inside the job, and every
training invocation first stops any previous local Ray runtime.

The DeepSeek script preserves checkpoints between its Megatron-Bridge LoRA
save/resume phases. DeepSeek and Qwen3 remove stale checkpoints before their
retained variants; MoE retains the workflow's cleanup boundary before its LoRA
variant. Every script has an exit trap that removes the repository-local
`checkpoints/` directory after success or failure. No artifacts are uploaded, so
a zero script exit status is the CI pass condition.

## Preflight checklist

Before starting a CI job, verify:

- exactly 8 CUDA devices are visible;
- every required model and raw-data path is readable;
- the workspace, Python environment, `${HOME}/data`, `${HOME}/dummy_models`, and
  `/tmp` are writable;
- `python3`, `pip3`, `git`, and `ray` are available; and
- localhost traffic and NCCL communication are not routed through the proxy.

## NVIDIA CI integration

These scripts are discovered and run weekly by
[`.github/workflows/verl-e2e-weekly.yml`](../verl-e2e-weekly.yml). That workflow
globs every `run_*.sh` in this directory into a parallel matrix (one 8-GPU job per
script), so adding a new `run_*.sh` here is picked up automatically with no
workflow edit. The workflow reads two tuning headers from each script:

- `# CI_TIMEOUT=<minutes>` — per-leg `timeout-minutes` (default 60).
- `# GPU_COUNT=<n>` — selects the N-GPU runner label (default 8).

In NVIDIA CI the L20x8 runner profile above is served by an 8×H100 node, and the
verl image is mirrored into the internal ECR registry. Required models and the raw
GSM8K dataset are pre-staged into the shared test-data volume, so each leg runs
offline (`HF_HUB_OFFLINE=1`); the workflow reproduces the `${HOME}/models` and
`${HOME}/models/hf_data/gsm8k` layout above via symlinks before invoking a script.
