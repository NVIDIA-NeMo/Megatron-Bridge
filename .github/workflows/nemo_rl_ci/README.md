# NeMo-RL Megatron weekly CI

This directory runs an existing NeMo-RL suite against the Megatron-Bridge and
Megatron-Core revisions checked out by this repository's weekly workflow. The
initial smoke is
`tests/test_suites/llm/grpo-qwen3-8b-base-1n8g-megatron-lora.sh` from NeMo-RL.
It runs 20 GRPO steps and retains the upstream convergence, numerical-error,
reward, and step-time assertions.

## Candidate inventory

| NeMo-RL suite | Workload | Resources | Steps / suite limit | Main caches and dependencies |
| --- | --- | --- | --- | --- |
| `grpo-qwen3-8b-base-1n8g-megatron-lora.sh` | Qwen3-8B-Base GRPO, Megatron LoRA, vLLM rollout | 1 node, 8 GPUs | 20 / 40 min | Qwen3-8B-Base, OpenMathInstruct-2, Ray, vLLM, Bridge/MCore worker environment |
| `sft-llama3.1-8b-1n8g-megatron-lora.sh` | Llama-3.1-8B SFT, Megatron LoRA | 1 node, 8 GPUs | 50 / 30 min | Gated Llama model/tokenizer, Tulu 3 SFT mixture, Ray, Bridge/MCore worker environment |
| `grpo-llama3.2-1b-instruct-1n8g-megatron.sh` | Llama-3.2-1B GRPO, full Megatron policy, vLLM rollout | 1 node, 8 GPUs | 500 / 180 min | Gated Llama model/tokenizer, OpenMathInstruct-2, Ray, vLLM, Bridge/MCore worker environment |

The Qwen3 test is selected because it is the shortest public-model candidate
that exercises the actual RL path: generation, reward calculation, policy
optimization, and weight synchronization with a Megatron policy.

## Pinned environment and version injection

The workflow pins both the NeMo-RL source SHA and the immutable CI image tag
built for that SHA. It reuses that image instead of building a new container on
every weekly run. Models and datasets use the runner's shared Hugging Face cache;
missing public assets may populate that cache on the first run.

NeMo-RL normally freezes separate actor environments. Setting only the driver
checkout is therefore insufficient and can silently test the Bridge package
baked into the image. The runner takes the narrower source-override path:

1. Mount the current Megatron-Bridge checkout and its checked-out MCore
   submodule into the pinned NeMo-RL container.
2. Put their source roots first in `PYTHONPATH`, which NeMo-RL propagates into
   Ray actor runtime environments.
3. Before training, invoke NeMo-RL's Megatron policy-worker interpreter and
   require `megatron.bridge` and `megatron.core` to resolve under those mounted
   roots.
4. Print and verify the full NeMo-RL, Bridge, and MCore Git SHAs in the run log.

This avoids the much slower `NRL_FORCE_REBUILD_VENVS=true` path while proving
that the frozen Megatron worker environment imports the checkout under test.
The job also requires step 20 unconditionally before repeating the upstream
metric assertions, closing the suite script's conditional-assertion gap.

## Adding a test

Prefer an existing, bounded NeMo-RL suite with explicit end-of-run metrics. Add
a small wrapper beside `run_grpo_qwen3_8b_lora.sh`, pin a compatible NeMo-RL
revision and image, and give the test its own single-node 8-GPU workflow job.
The wrapper must:

- verify the mounted Bridge and MCore import paths with the actual policy-worker
  interpreter;
- record the three full repository SHAs;
- disable external experiment logging;
- assert the expected final step independently of conditional upstream checks;
- preserve the upstream metric assertions; and
- remove only its own output and checkpoints on exit.

Keep the workflow schedule-only plus manual dispatch, use the shared model/data
cache, and retain the failure notification job.
