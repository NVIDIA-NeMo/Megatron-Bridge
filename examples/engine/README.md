# TrainingEngine examples

Examples for `megatron.bridge.engine.TrainingEngine`, an in-process, every-rank training object
built from a `TrainingRecipe`. All of them run under `torchrun` or `srun` with the recipe's world
size (16 ranks for the Qwen3.5-35B-A3B recipe used below) and share three arguments:

| Argument | Meaning |
|---|---|
| `--recipe` | Name of a recipe in `megatron.bridge.engine.recipes`, e.g. `qwen35_text_35b_a3b_rl_16gpu_gb200_bf16` |
| `--recipe-kwargs` | JSON for the recipe function, e.g. `'{"hf_path": "/models/qwen35-text-35b-a3b"}'` (a local HF directory with config, tokenizer and safetensors) |
| `--checkpoint` | Megatron checkpoint to load the model weights from |

The SFT, scoring and gate examples read an offline-packed parquet (`input_ids`, `seq_start_id`,
`loss_mask` per row); the RL examples read the DAPO-Math-17k parquet (`prompt`, `reward_model`).

Environment that every run needs:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton-cache-${SLURM_JOB_ID}-${SLURM_PROCID}   # node-local, per rank
export VLLM_CACHE_ROOT=/tmp/vllm-cache-${USER}-gpu${SLURM_LOCALID}            # RL examples only
export HF_HUB_OFFLINE=1
```

A launch looks like this (one task per GPU):

```bash
srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \
    python examples/engine/<example>.py \
        --recipe qwen35_text_35b_a3b_rl_16gpu_gb200_bf16 \
        --recipe-kwargs '{"hf_path": "/models/qwen35-text-35b-a3b"}' \
        --checkpoint /checkpoints/qwen35-text-35b-a3b-megatron \
        <example arguments>
```

## Examples

### `construct_from_recipe.py` (construction gate)

Builds a trainable actor and a frozen reference from the same recipe, prints parallel info,
parameter counts and memory, closes the actor and checks the reference still works and the actor's
memory was released. Extra: `--skip-reference`; trailing `key=value` pairs are recipe overrides
(`model.context_parallel_size=4`).

### `score_packed_parquet.py` (`forward_only` contract)

Scores the first `--num-packs` rows with `forward_only`, in gathered and local mode and in reversed
order, checks the three agree, and prints the mean NLL for comparison with the baseline validation
loss. `--with-trainable` repeats the scoring on a trainable engine and checks its gradient buffers
stay zero.

```bash
... score_packed_parquet.py --parquet /data/packed/training.parquet --num-packs 8 --with-trainable
```

### `sft_from_recipe.py` (SFT, checkpoints, export)

Engine-driven SFT on packed data, logged to W&B. Each step consumes `--packs-per-step` rows in
`--calls-per-step` accumulation calls; the learning-rate schedule lives in the script, the engine
takes the value per step.

```bash
... sft_from_recipe.py --parquet /data/packed/training.parquet --iters 50 --packs-per-step 32 \
    --calls-per-step 1 --wandb-project <project> --wandb-entity <entity> --wandb-name <run>
```

- `--save-at N --save-dir ROOT [--async-save]` saves an engine checkpoint after step N;
  `--resume-from ROOT` continues a run (the data cursor travels in `user_state`).
- `--export-dir DIR` writes an HF checkpoint at the end and verifies it.
- `--check-accumulation` runs the strict accumulation gate instead: one update as 1 call versus 4
  calls from identical state, comparing gradient buffers, gradient norms and master parameters.

### `rl_loss_smoke.py` (RL losses on the real model)

Synthetic RL fields on SFT packs. `--output-head lm` runs `ppo_clip` and `importance_sampling`
(with truncated importance sampling and the k3 KL term); `--output-head value` builds a value-head
critic from the same recipe and runs `value_loss` plain and clipped.

```bash
... rl_loss_smoke.py --parquet /data/packed/training.parquet --packs-per-step 8 --output-head lm
... rl_loss_smoke.py --parquet /data/packed/training.parquet --packs-per-step 8 --output-head value
```

### `movement_gate.py` (`cpu()` / `cuda()`)

Two engines accumulate the same four calls; the second is moved to host memory mid-window while a
frozen reference engine is built, scores the same packs and is closed, then comes back. The update
must match the uninterrupted engine. It also checks that the GPU allocation drops by exactly what the
engine reports it moved and prints transfer times.

```bash
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 ... movement_gate.py --parquet /data/packed/training.parquet --packs-per-step 8
```

### `grpo_vllm.py` (GRPO + truncated importance sampling, vllm colocated)

Per round: refresh vllm's weights from the policy's streaming export, offload the policy, sample
`--group-size` responses per prompt with vllm (one replica per rank, sharing the GPUs through sleep
mode), score with the DAPO rule, normalize rewards within each group, bring the policy back, record
`old_logprobs`, and run `--steps-per-round` PPO-clip updates of `--calls-per-step` calls each.

```bash
... grpo_vllm.py --parquet /data/dapo-math/data/dapo-math-17k.parquet \
    --rounds 60 --prompts-per-round 32 --group-size 8 --max-tokens 126976 \
    --lr 1e-6 --clip-low 0.2 --clip-high 0.28 --tis-coef 2.0 --vllm-cuda-graphs \
    --wandb-project <project> --wandb-entity <entity> --wandb-name grpo_dapo128k \
    --eval-every 10 --save-dir /ckpt/grpo_dapo128k --wall-seconds 13500
```

Knobs:

- `--max-tokens` is the response cap; `--max-prompt-tokens` (1024) plus the cap must stay below the
  recipe's sequence length (131072), so 126976 is the largest cap for this recipe and the one used
  above (full 128K sequences). The vllm model length is set from the two. Expect about 35 minutes of
  generation per round of 64 samples at this cap on 16 GB300 GPUs; 32768 cuts that to about 12
  minutes for 256 samples but truncates this model's longest reasoning traces.
- `--vllm-cuda-graphs` lets vllm compile and capture CUDA graphs (about 4x faster decoding; off by
  default). `--gpu-memory-utilization` (0.5) is vllm's share of each GPU while the engines are
  offloaded.
- `--eval-every N` scores a fixed held-out set (`--eval-prompts` prompts from parquet row
  `--eval-offset`, de-duplicated against the training prompts, `--eval-samples` each) with the same
  weights right after the training batch; reported as `eval_*`.
- `--save-every N --save-dir ROOT` checkpoints the trainable engines under `ROOT/<engine>`;
  `--resume-from ROOT` continues from the latest checkpoint. `--wall-seconds S` stops cleanly before
  a job time limit (save, exit) so a chained job can resume; the W&B run id is the run name, so chained
  jobs append to one run.

Each round prints one JSON line (`[grpo] {...}`) with reward, accuracy, response length, truncation,
train-versus-vllm logprob agreement, clip and TIS fractions, grad norms, timings and the held-out
numbers, mirrored to W&B when a project is given.

### `ppo_vllm.py` (PPO + reference KL + critic, vllm colocated)

Same loop with three engines from one recipe: the policy, a frozen reference (`trainable=False`)
that provides `reference_logprobs` for the k3 KL penalty, and a value-head critic
(`output_head="value"`) scored and updated every round (GAE from the terminal reward, clipped value
loss). The engines and vllm take turns on the same GPUs through `cpu()` / `cuda()` and sleep mode.
Takes every `grpo_vllm.py` argument plus:

```bash
... ppo_vllm.py <grpo arguments> --kl-beta 0.01 --critic-lr 1e-5 --value-clip 0.5
```

Round lines are `[ppo] {...}` and add `value_abs_error`, `value_clipped_fraction`,
`critic_grad_norms` and `kl_per_token`.

## Shared modules

- `dapo_math.py`: DAPO-Math prompt loading with the chat template, the +1/-1 reward, GRPO group
  normalization, GAE, and the sample-to-`Datum` assembly (prompt plus response, response-only loss mask,
  per-token `advantages` and `rollout_logprobs`).
- `dapo_math_reward.py`: verl's DAPO scorer, copied unchanged (Minerva-style `Answer:` extraction
  first, then the last `\boxed{}`).
- `vllm_rollout.py`: vllm 0.29 in the training process (`distributed_executor_backend="external_launcher"`,
  one data-parallel replica per rank, `async_scheduling=False`), weight refresh through the runner's
  layerwise reload from `engine.export(streaming=True)`, sleep level 2 between rounds, and sharded
  generation with an all-gather so every rank holds the same sample list. Ranks without prompts still
  drive their engine so the data-parallel coordination stays in lockstep.

## Residency protocol used by the RL examples

1. Policy on the GPU, vllm asleep: refresh vllm's weights from the policy (about 6 s for 35B).
2. `policy.cpu()` (and the other engines): vllm wakes its KV cache and generates.
3. vllm sleeps; engines come back one at a time for scoring and updates.

With 276 GiB GPUs the whole bf16 model fits in one vllm replica per rank, so no vllm tensor
parallelism or inter-process weight transfer is needed.
