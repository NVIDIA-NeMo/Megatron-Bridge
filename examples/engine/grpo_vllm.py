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
"""GRPO with truncated importance sampling on DAPO-Math, vllm colocated with the policy engine.

Per round: refresh vllm's weights from the policy (round 0 uses the HF weights the checkpoint was
converted from), offload the policy, sample ``group_size`` responses per prompt, score them with the
DAPO rule, normalize rewards within each group, bring the policy back, record its logprobs as
``old_logprobs``, and run ``steps_per_round`` PPO-clip updates (each accumulated over
``calls_per_step`` forward_backward calls). The ratio denominator is the policy before the round; the
truncated importance weight ``min(exp(old - rollout), tis_coef)`` corrects for vllm's sampling
distribution. Everything runs on every rank; vllm generates per rank and the samples are gathered.
"""

import argparse
import json
import logging
import time

import torch
from dapo_math import Sample, group_advantages, load_prompts, response_slice, score, to_datum
from transformers import AutoTokenizer
from vllm_rollout import VllmRollout

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, Datum, TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


def parse_args(add_arguments=None, description: str = __doc__) -> argparse.Namespace:
    """Common RL example arguments; ``add_arguments(parser)`` lets the PPO example add its own."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if add_arguments is not None:
        add_arguments(parser)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True, help="DAPO-Math-17k parquet")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompts-per-round", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max response tokens (32K for the long run)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps-per-round", type=int, default=2)
    parser.add_argument("--calls-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--tis-coef", type=float, default=2.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-cuda-graphs", action="store_true", help="Let vllm compile and capture CUDA graphs")
    parser.add_argument(
        "--router-replay",
        action="store_true",
        help="Rollout routing replay: vllm returns each token's experts and the policy replays them when scoring "
        "and training (recipe router_replay=True; the reference and critic keep their own routing)",
    )
    parser.add_argument("--prompt-offset", type=int, default=0)
    # Convergence-run extras: W&B logging, a fixed held-out prompt set, periodic checkpoints and resume.
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--eval-every", type=int, default=0, help="Score the held-out set every N rounds (0: never)")
    parser.add_argument("--eval-prompts", type=int, default=32)
    parser.add_argument("--eval-offset", type=int, default=17000, help="Parquet row the held-out set starts at")
    parser.add_argument("--eval-samples", type=int, default=2, help="Samples per held-out prompt")
    parser.add_argument("--save-every", type=int, default=0, help="save_state every N rounds (0: never)")
    parser.add_argument("--save-dir", default=None, help="Checkpoint root; one subdirectory per engine")
    parser.add_argument("--resume-from", default=None, help="Checkpoint root written by --save-dir")
    parser.add_argument(
        "--wall-seconds",
        type=float,
        default=0.0,
        help="Stop before this much wall time has passed since launch: save to --save-dir and exit (0: never)",
    )
    return parser.parse_args()


def chunks(items: list, n: int) -> list[list]:
    """Split into ``n`` nearly equal consecutive chunks (groups stay consecutive)."""
    size = -(-len(items) // n)
    return [items[i : i + size] for i in range(0, len(items), size)]


def consistency(datums: list[Datum], logprobs: list[torch.Tensor]) -> float:
    """Mean |policy logprob - vllm logprob| over response tokens: the train/inference agreement."""
    total, count = 0.0, 0
    for datum, lp in zip(datums, logprobs):
        diff = (response_slice(datum, lp.cpu()) - response_slice(datum, datum.rollout_logprobs)).abs()
        total += float(diff.sum())
        count += diff.numel()
    return total / max(count, 1)


def rollout_round(
    rollout: VllmRollout, policy: TrainingEngine, prompts, args, *, eval_prompts=None
) -> tuple[list[Sample], dict[str, float]]:
    """Refresh weights (policy resident), offload the policy, generate, sleep vllm, score and normalize.

    The refresh runs every round, including the first: ``sleep(level=2)`` discards vllm's weights, so
    they must be rewritten from the policy before each generation. With ``eval_prompts`` the held-out
    set is sampled too, with the same weights, and its statistics are returned under ``eval_*``.
    """
    rollout.refresh_from(policy)
    policy.cpu()
    print_rank_0("[round] policy offloaded; waking vllm's KV cache")
    rollout.wake_kv_cache()
    print_rank_0(f"[round] generating {args.group_size} samples for each of {len(prompts)} prompts")
    samples = rollout.generate(
        prompts, group_size=args.group_size, max_tokens=args.max_tokens, temperature=args.temperature
    )
    eval_stats: dict[str, float] = {}
    if eval_prompts:
        print_rank_0(f"[round] scoring the held-out set: {len(eval_prompts)} prompts x {args.eval_samples}")
        held_out = rollout.generate(
            eval_prompts, group_size=args.eval_samples, max_tokens=args.max_tokens, temperature=args.temperature
        )
        for sample in held_out:
            sample.reward = score(sample)
        eval_stats = {f"eval_{k}": v for k, v in summarize_samples(held_out).items()}
    rollout.sleep()
    for sample in samples:
        sample.reward = score(sample)
    group_advantages(samples, group_size=args.group_size)
    return samples, eval_stats


def summarize_samples(samples: list[Sample]) -> dict[str, float]:
    """Reward, accuracy, response-length and truncation statistics of one round's samples."""
    lengths = [len(s.response_ids) for s in samples]
    return {
        "reward_mean": sum(s.reward for s in samples) / len(samples),
        "accuracy": sum(s.reward > 0 for s in samples) / len(samples),
        "response_tokens_mean": sum(lengths) / len(lengths),
        "response_tokens_max": max(lengths),
        "truncated_fraction": sum(s.extras.get("finish_reason") == "length" for s in samples) / len(samples),
    }


def train_round(policy: TrainingEngine, datums: list[Datum], args, *, kl_beta: float = 0.0) -> dict[str, float]:
    """old_logprobs from the current policy, then steps_per_round clipped updates with TIS (and KL if set)."""
    policy.cuda()
    start = time.perf_counter()
    scored = policy.forward_only(datums)
    for datum, lp in zip(datums, scored.logprobs):
        datum.old_logprobs = lp.cpu()
    stats = {"train_vs_vllm_abs_logprob_diff": consistency(datums, scored.logprobs)}
    loss_kwargs = {
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "tis_coef": args.tis_coef,
        "kl_beta": kl_beta,
    }
    tokens = clipped = truncated = kl = 0.0
    loss_sum = 0.0
    grad_norms = []
    for step_datums in chunks(datums, args.steps_per_round):
        for call_datums in chunks(step_datums, args.calls_per_step):
            out = policy.forward_backward(call_datums, loss_fn="ppo_clip", loss_kwargs=loss_kwargs)
            tokens += out.num_loss_tokens
            loss_sum += out.loss_sum
            clipped += out.metrics["clipped_tokens"]
            truncated += out.metrics["tis_truncated_tokens"]
            kl += out.metrics.get("kl_sum", 0.0)
        result = policy.optim_step(AdamParams(learning_rate=args.lr))
        grad_norms.append(result["grad_norm"] if result["update_successful"] else None)
    stats.update(
        {
            "loss_per_token": loss_sum / max(tokens, 1),
            "clipped_fraction": clipped / max(tokens, 1),
            "tis_truncated_fraction": truncated / max(tokens, 1),
            "grad_norms": grad_norms,
            "skipped_updates": sum(g is None for g in grad_norms),
            "train_seconds": time.perf_counter() - start,
        }
    )
    if kl_beta > 0.0:
        stats["kl_per_token"] = kl / max(tokens, 1)
    return stats


class RunLogger:
    """One JSON line per round on stdout, mirrored to W&B (rank 0) when a project is given."""

    def __init__(self, args, tag: str):
        self.tag = tag
        self.run = None
        if args.wandb_project and torch.distributed.get_rank() == 0:
            import wandb

            # The run name doubles as the run id so chained (resumed) jobs append to one W&B run.
            self.run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                id=args.wandb_name,
                resume="allow",
                config=vars(args),
            )

    def log(self, stats: dict) -> None:
        print_rank_0(f"[{self.tag}] " + json.dumps(stats))
        if self.run is None:
            return
        flat = {}
        for key, value in stats.items():
            if isinstance(value, list):
                finite = [v for v in value if v is not None]
                if finite:
                    flat[f"{key}_mean"] = sum(finite) / len(finite)
                    flat[f"{key}_max"] = max(finite)
            else:
                flat[key] = value
        self.run.log(flat, step=stats["round"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def load_data(args, recipe) -> tuple[list, list]:
    """Training prompts for all rounds plus the held-out set (disjoint from the training prompts)."""
    tokenizer = AutoTokenizer.from_pretrained(recipe.hf_path)
    prompts = load_prompts(
        args.parquet,
        tokenizer,
        limit=args.rounds * args.prompts_per_round,
        offset=args.prompt_offset,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    eval_prompts: list = []
    if args.eval_every > 0:
        seen = {tuple(p.prompt_ids) for p in prompts}
        candidates = load_prompts(
            args.parquet,
            tokenizer,
            limit=2 * args.eval_prompts,
            offset=args.eval_offset,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        eval_prompts = [p for p in candidates if tuple(p.prompt_ids) not in seen][: args.eval_prompts]
    print_rank_0(f"[data] {len(prompts)} training prompts for {args.rounds} rounds, {len(eval_prompts)} held out")
    return prompts, eval_prompts


def one_engine_at_a_time(engines: dict[str, TrainingEngine], action) -> dict[str, object]:
    """Run ``action(name, engine)`` on each engine while it alone is resident, then restore residency.

    The examples only fit because engines take turns on the GPU, so a checkpoint or restore must not
    bring a second engine up next to the one that is already resident.
    """
    was_resident = {name: engine.device == "cuda" for name, engine in engines.items()}
    results = {}
    for name, engine in engines.items():
        for other_name, other in engines.items():
            if other_name != name and other.device == "cuda":
                other.cpu()
        engine.cuda()
        results[name] = action(name, engine)
    # Restore in two passes: everything that belongs on the host leaves first, so an engine coming
    # back never sits next to the last one acted on.
    for name, engine in engines.items():
        if not was_resident[name] and engine.device == "cuda":
            engine.cpu()
    for name, engine in engines.items():
        if was_resident[name] and engine.device == "cpu":
            engine.cuda()
    return results


def resume_round(engines: dict[str, TrainingEngine], args) -> int:
    """Restore every engine from ``--resume-from/<name>``; returns the round to continue from."""
    if not args.resume_from:
        return 0
    states = one_engine_at_a_time(engines, lambda name, engine: engine.load_state(f"{args.resume_from}/{name}"))
    rounds = {int(state["round"]) for state in states.values()}
    assert len(rounds) == 1, f"engines were saved at different rounds: {rounds}"
    next_round = rounds.pop()
    print_rank_0(f"[ckpt] resumed from {args.resume_from}; continuing at round {next_round}")
    return next_round


def save_round(engines: dict[str, TrainingEngine], args, next_round: int) -> None:
    """Checkpoint every trainable engine under ``--save-dir/<name>`` with the next round in user_state."""
    one_engine_at_a_time(
        engines,
        lambda name, engine: engine.save_state(f"{args.save_dir}/{name}", user_state={"round": next_round}),
    )
    print_rank_0(f"[ckpt] saved {sorted(engines)} to {args.save_dir} (next round {next_round})")


LAUNCHED_AT = time.time()


def out_of_wall_time(args, last_round_seconds: float | None) -> bool:
    """True when another round (estimated from the last one, with margin) would cross ``--wall-seconds``."""
    if not args.wall_seconds or last_round_seconds is None:
        return False
    return time.time() - LAUNCHED_AT + 1.25 * last_round_seconds > args.wall_seconds


def eval_prompts_for(round_index: int, args, eval_prompts: list) -> list | None:
    """The held-out set on eval rounds (the first round and every ``--eval-every`` after), else None."""
    if args.eval_every > 0 and round_index % args.eval_every == 0:
        return eval_prompts
    return None


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipe.router_replay = args.router_replay
    prompts, eval_prompts = load_data(args, recipe)

    policy = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint)
    policy.cpu()  # vllm measures free memory at construction
    rollout = VllmRollout(
        recipe.hf_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_tokens + 64,
        enforce_eager=not args.vllm_cuda_graphs,
        return_routed_experts=args.router_replay,
        moe_layers=policy.router_replay_layers if args.router_replay else None,
    )
    rollout.sleep()
    policy.cuda()
    engines = {"policy": policy}
    first_round = resume_round(engines, args)
    logger = RunLogger(args, "grpo")
    print_rank_0("[setup] vllm asleep, policy resident; starting rounds")

    last_round_seconds = None
    for round_index in range(first_round, args.rounds):
        if out_of_wall_time(args, last_round_seconds):
            print_rank_0(f"[wall] stopping before round {round_index}; saving for the next job")
            save_round(engines, args, round_index)
            break
        round_prompts = prompts[round_index * args.prompts_per_round : (round_index + 1) * args.prompts_per_round]
        start = time.perf_counter()
        samples, eval_stats = rollout_round(
            rollout, policy, round_prompts, args, eval_prompts=eval_prompts_for(round_index, args, eval_prompts)
        )
        stats = {"round": round_index, "rollout_seconds": time.perf_counter() - start, **summarize_samples(samples)}
        stats.update(eval_stats)
        datums = [to_datum(s) for s in samples]
        stats.update(train_round(policy, datums, args))
        last_round_seconds = time.perf_counter() - start
        stats["round_seconds"] = last_round_seconds
        logger.log(stats)
        if args.save_every and (round_index + 1) % args.save_every == 0 and round_index + 1 < args.rounds:
            save_round(engines, args, round_index + 1)
    logger.finish()
    policy.close()
    print_rank_0("GRPO EXAMPLE PASSED")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
