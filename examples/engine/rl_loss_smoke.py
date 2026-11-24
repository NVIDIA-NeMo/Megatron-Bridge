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
"""Shape-and-reduction smoke for the RL losses on the real model, with synthetic RL fields.

``--output-head lm``: scores one batch of SFT packs with ``forward_only`` to get ``old_logprobs``
(also used as reference and, with noise, as rollout logprobs), draws per-token advantages, then runs
``ppo_clip`` and ``importance_sampling`` through ``forward_backward`` + ``optim_step``. On the first
call the policy equals the old policy, so the ratio is 1 everywhere: no clipping, zero KL. After one
update the same batch must show non-zero logprob drift.

``--output-head value``: a critic engine built from the same recipe; ``forward_only`` returns
values, the targets are ``values + 1`` and ``value_loss`` is run plain and clipped. This is API
validation only; advantages and returns are synthetic.
"""

from __future__ import annotations

import argparse
import json
import logging
import math

import torch
from sft_from_recipe import iter_pack_batches

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, Datum, TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--overrides", default="{}", help="Recipe fields by dotted path, as JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--packs-per-step", type=int, default=8)
    parser.add_argument("--output-head", choices=["lm", "value"], default="lm")
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def _seeded(length: int, seed: int, *, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return scale * torch.randn(length, generator=generator)


def _per_token(out, key: str) -> float:
    return out.metrics[key] / max(out.num_loss_tokens, 1)


def _report(label: str, out) -> None:
    metrics = " ".join(f"{k}={v:.4g}" for k, v in sorted(out.metrics.items()))
    print_rank_0(f"[{label}] loss_sum={out.loss_sum:.4f} tokens={out.num_loss_tokens} {metrics}")
    assert math.isfinite(out.loss_sum), f"{label}: loss is not finite"


def run_policy(engine: TrainingEngine, datums: list[Datum], lr: float) -> None:
    """ppo_clip then importance_sampling, each as forward_backward + optim_step on the same batch."""
    scored = engine.forward_only(datums)
    assert len(scored.logprobs) == len(datums)
    for index, (datum, logprobs) in enumerate(zip(datums, scored.logprobs)):
        old = logprobs.detach().cpu().float()
        assert old.shape == datum.target_ids.shape
        datum.old_logprobs = old
        datum.reference_logprobs = old.clone()
        datum.rollout_logprobs = old + _seeded(len(old), 7_000 + index, scale=0.5)
        datum.advantages = _seeded(len(old), 9_000 + index)
    active_tokens = sum(int(d.loss_mask.sum()) for d in datums)
    kwargs = {"clip_low": 0.2, "clip_high": 0.2, "kl_beta": 0.1, "tis_coef": 1.5}

    first = engine.forward_backward(datums, loss_fn="ppo_clip", loss_kwargs=kwargs)
    _report("ppo_clip call 1 (policy == old)", first)
    assert first.num_loss_tokens == active_tokens, "token count does not match the loss masks"
    assert first.metrics["clipped_tokens"] == 0, "ratio must be 1 before any update"
    assert _per_token(first, "abs_logprob_diff_sum") < 1e-2, "forward_backward logprobs drifted from forward_only"
    assert _per_token(first, "kl_sum") < 1e-4, "KL to an identical reference must vanish"
    assert first.metrics["tis_truncated_tokens"] > 0, "noisy rollout logprobs should truncate some weights"
    result = engine.optim_step(AdamParams(learning_rate=lr))
    print_rank_0(f"[ppo_clip step] grad_norm={result['grad_norm']:.4f} ok={result['update_successful']}")

    second = engine.forward_backward(datums, loss_fn="ppo_clip", loss_kwargs=kwargs)
    _report("ppo_clip call 2 (after one update)", second)
    assert _per_token(second, "abs_logprob_diff_sum") > 0, "the update did not move the policy"
    engine.optim_step(AdamParams(learning_rate=lr))

    third = engine.forward_backward(datums, loss_fn="importance_sampling", loss_kwargs={"tis_coef": 1.5})
    _report("importance_sampling", third)
    assert third.num_loss_tokens == active_tokens
    assert set(third.metrics) == {"abs_logprob_diff_sum", "tis_truncated_tokens"}
    engine.optim_step(AdamParams(learning_rate=lr))


def run_value(engine: TrainingEngine, datums: list[Datum], lr: float) -> None:
    """value_loss plain and clipped on a critic engine; targets are the current values plus one."""
    scored = engine.forward_only(datums)
    assert scored.logprobs is None and len(scored.values) == len(datums)
    for datum, values in zip(datums, scored.values):
        values = values.detach().cpu().float()
        assert values.shape == datum.target_ids.shape
        datum.loss_inputs = {"returns": values + 1.0, "old_values": values.clone()}
    active_tokens = sum(int(d.loss_mask.sum()) for d in datums)

    first = engine.forward_backward(datums)  # defaults to value_loss on a value-head engine
    _report("value_loss call 1 (targets = values + 1)", first)
    assert first.num_loss_tokens == active_tokens
    assert abs(first.loss_sum - 0.5 * active_tokens) < 1e-2 * active_tokens, "0.5 * 1^2 per active token"
    assert abs(_per_token(first, "value_abs_error_sum") - 1.0) < 1e-2
    result = engine.optim_step(AdamParams(learning_rate=lr))
    print_rank_0(f"[value step] grad_norm={result['grad_norm']:.4f} ok={result['update_successful']}")

    # Direction check only: Adam's first step moves every parameter by about lr, so the value shift can
    # overshoot a unit target gap on a 35B model; what must hold is that values moved up on average.
    after = engine.forward_only(datums)
    shift = torch.cat(
        [
            (values.detach().cpu().float() - datum.loss_inputs["old_values"])[datum.loss_mask.bool()]
            for values, datum in zip(after.values, datums)
        ]
    )
    print_rank_0(f"[value shift] mean={shift.mean():.4f} min={shift.min():.3f} max={shift.max():.3f}")
    assert shift.mean() > 0, "the update should push values toward the higher targets"

    second = engine.forward_backward(datums, loss_kwargs={"clip": 0.5})
    _report("value_loss call 2 (clip 0.5, after one update)", second)
    assert "value_clipped_tokens" in second.metrics
    engine.optim_step(AdamParams(learning_rate=lr))


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipes.apply_recipe_overrides(recipe, json.loads(args.overrides))
    engine = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, output_head=args.output_head)
    print_rank_0(f"[engine] built output_head={args.output_head}; parallel_info={engine.parallel_info}")
    datums = next(iter_pack_batches(args.parquet, args.packs_per_step))
    print_rank_0(f"[data] {len(datums)} Datums from {args.packs_per_step} packs")
    if args.output_head == "lm":
        run_policy(engine, datums, args.lr)
    else:
        run_value(engine, datums, args.lr)
    engine.close()
    print_rank_0("RL LOSS SMOKE PASSED")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
