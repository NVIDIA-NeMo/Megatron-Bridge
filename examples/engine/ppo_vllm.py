# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PPO with a reference-KL penalty and a critic on DAPO-Math, vllm colocated with three engines.

Policy, reference (frozen) and critic (value head) are built from the same recipe and checkpoint and
take turns on the GPUs with vllm. Per round: refresh vllm from the policy, generate, score; the
reference scores the samples (``reference_logprobs`` for the k3 penalty); the critic scores them
(GAE advantages and returns from the terminal reward) and is updated with the clipped value loss;
the policy is updated with ``ppo_clip`` (clip, KL penalty, truncated importance sampling).
"""

import json
import logging
import time

import torch
from dapo_math import Sample, gae, response_slice, to_datum, without_routing
from grpo_vllm import (
    RunLogger,
    chunks,
    eval_prompts_for,
    load_data,
    out_of_wall_time,
    resume_round,
    rollout_round,
    save_round,
    summarize_samples,
    train_round,
)
from grpo_vllm import parse_args as parse_grpo_args
from vllm_rollout import VllmRollout

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, Datum, TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


def _ppo_arguments(parser) -> None:
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--critic-lr", type=float, default=1e-5)
    parser.add_argument("--value-clip", type=float, default=0.5)


def critic_round(critic: TrainingEngine, samples: list[Sample], datums: list[Datum], args) -> dict[str, float]:
    """Score with the critic, write GAE advantages/returns into the Datums, then update the critic."""
    critic.cuda()
    start = time.perf_counter()
    values = critic.forward_only(without_routing(datums)).values
    all_adv = []
    for sample, datum, value in zip(samples, datums, values):
        v = response_slice(datum, value.cpu())
        rewards = torch.zeros_like(v)
        rewards[-1] = sample.reward
        adv, ret = gae(rewards, v)
        mask = datum.loss_mask.bool()
        datum.advantages[mask] = adv
        datum.loss_inputs["returns"] = torch.zeros_like(datum.loss_mask)
        datum.loss_inputs["returns"][mask] = ret
        datum.loss_inputs["old_values"] = torch.zeros_like(datum.loss_mask)
        datum.loss_inputs["old_values"][mask] = v
        all_adv.append(adv)
    flat = torch.cat(all_adv)
    mean, std = flat.mean(), flat.std() + 1e-6
    for datum in datums:
        mask = datum.loss_mask.bool()
        datum.advantages[mask] = (datum.advantages[mask] - mean) / std
    tokens = error = clipped = 0.0
    grad_norms = []
    for step_datums in chunks(datums, args.steps_per_round):
        for call_datums in chunks(step_datums, args.calls_per_step):
            out = critic.forward_backward(without_routing(call_datums), loss_kwargs={"clip": args.value_clip})
            tokens += out.num_loss_tokens
            error += out.metrics["value_abs_error_sum"]
            clipped += out.metrics["value_clipped_tokens"]
        result = critic.optim_step(AdamParams(learning_rate=args.critic_lr))
        grad_norms.append(result["grad_norm"] if result["update_successful"] else None)
    critic.cpu()
    return {
        "value_abs_error": error / max(tokens, 1),
        "value_clipped_fraction": clipped / max(tokens, 1),
        "critic_grad_norms": grad_norms,
        "advantage_raw_mean": float(mean),
        "critic_seconds": time.perf_counter() - start,
    }


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = parse_grpo_args(_ppo_arguments, description=__doc__)
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipe.router_replay = args.router_replay  # only the policy receives Datums with routing
    prompts, eval_prompts = load_data(args, recipe)

    policy = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint)
    policy.cpu()
    reference = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=False)
    reference.cpu()
    critic = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, output_head="value")
    critic.cpu()
    rollout = VllmRollout(
        recipe.hf_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_tokens + 64,
        enforce_eager=not args.vllm_cuda_graphs,
        return_routed_experts=args.router_replay,
    )
    rollout.sleep()
    engines = {"policy": policy, "critic": critic}  # the reference never changes, so it is not saved
    first_round = resume_round(engines, args)
    logger = RunLogger(args, "ppo")

    last_round_seconds = None
    for round_index in range(first_round, args.rounds):
        if out_of_wall_time(args, last_round_seconds):
            print_rank_0(f"[wall] stopping before round {round_index}; saving for the next job")
            save_round(engines, args, round_index)
            break
        round_prompts = prompts[round_index * args.prompts_per_round : (round_index + 1) * args.prompts_per_round]
        start = time.perf_counter()
        policy.cuda()  # the refresh exports from the resident policy
        samples, eval_stats = rollout_round(
            rollout, policy, round_prompts, args, eval_prompts=eval_prompts_for(round_index, args, eval_prompts)
        )
        stats = {"round": round_index, "rollout_seconds": time.perf_counter() - start, **summarize_samples(samples)}
        stats.update(eval_stats)
        datums = [to_datum(s) for s in samples]

        reference.cuda()
        ref_logprobs = reference.forward_only(without_routing(datums)).logprobs
        reference.cpu()
        for datum, lp in zip(datums, ref_logprobs):
            datum.reference_logprobs = lp.cpu()

        stats.update(critic_round(critic, samples, datums, args))
        stats.update(train_round(policy, datums, args, kl_beta=args.kl_beta))
        last_round_seconds = time.perf_counter() - start
        stats["round_seconds"] = last_round_seconds
        logger.log(stats)
        if args.save_every and (round_index + 1) % args.save_every == 0 and round_index + 1 < args.rounds:
            save_round(engines, args, round_index + 1)
    logger.finish()
    for engine in (policy, reference, critic):
        engine.close()
    print_rank_0("PPO EXAMPLE PASSED")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
