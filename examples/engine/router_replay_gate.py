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
"""Router replay gate: vllm's expert choices replayed in the policy on 16 GPUs.

One small rollout with ``return_routed_experts``, then on the policy (recipe ``router_replay=True``):
  1. logprobs with native routing, with replayed routing, and the gap of each to vllm's logprobs;
  2. layout: the routing the policy's own routers record on a native pass overlaps the prepared
     ``routed_experts`` (a token-order bug would drop the overlap to about topk / num_experts);
  3. exactness: replaying the policy's own recorded routing reproduces the native logprobs;
  4. training with vllm's routing: the training forward equals the scoring forward (ratio exactly 1),
     every router replays once in the forward and once in the recompute per microbatch, the gradient
     is repeatable, and the forward loss is identical with and without MoE recompute (the gradient
     within the recompute's rounding). Numbers against native are reported only: the native
     training-mode MoE path routes some tokens differently from the scoring pass and from its own
     recompute, so native is not an exact reference. Advantages are synthetic +1/-1 so the policy
     gradient does not vanish, and the load-balancing loss is off by default (its global tracker
     keeps state across calls).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import replace

import torch
from dapo_math import to_datum, without_routing
from grpo_vllm import chunks, consistency, load_data, parse_args, rollout_round
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from vllm_rollout import VllmRollout

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, Datum, TrainingEngine
from megatron.bridge.engine.batching import all_gather_dp, unpack_sequences, unshard_from_cp
from megatron.bridge.engine.schedule import run_forward_only
from megatron.bridge.utils.common_utils import print_rank_0


def _gate_arguments(parser) -> None:
    parser.add_argument(
        "--aux-loss-coeff",
        type=float,
        default=0.0,
        help="MoE load-balancing loss coefficient for the gate (default 0: the recipe's global aux loss keeps "
        "state across calls, which makes repeated gradients differ; pass a negative value to keep the recipe's)",
    )


def _max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x.cpu() - y.cpu()).abs().max()) for x, y in zip(a, b))


def _mean_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    total = sum(float((x.cpu() - y.cpu()).abs().sum()) for x, y in zip(a, b))
    return total / max(sum(x.numel() for x in a), 1)


def _overlap(a: list[Datum], b: list[Datum]) -> float:
    """Mean per-token fraction of shared experts between two routings of the same Datums."""
    hits = total = 0.0
    for x, y in zip(a, b):
        shared = (x.routed_experts[..., :, None] == y.routed_experts[..., None, :]).any(-1).float()
        hits += float(shared.sum())
        total += shared.numel()
    return hits / max(total, 1)


def record_native_routing(
    policy: TrainingEngine, datums: list[Datum], *, train: bool = False, loss_kwargs: dict | None = None
) -> tuple[float, list[Datum]]:
    """Run a native pass with the routers recording; return the overlap with the prepared routing and
    Datums carrying the recorded routing (same order as ``datums``).

    ``train=False`` records in a scoring pass (eval mode, no grad); ``train=True`` records in a
    ``forward_backward`` whose gradients are discarded, so the routing is the one a training step
    sees (the kernels differ slightly between the two modes and near-tie top-k choices can flip).
    """
    microbatches = policy.preprocess(datums, training=train)
    policy.check_microbatches(microbatches)
    if len(microbatches) != 1:
        raise RuntimeError(f"the layout check needs one microbatch per rank, got {len(microbatches)}; fewer prompts")
    state = policy._router_replay
    for router in state.routers:
        router.set_router_replay_action(RouterReplayAction.RECORD)
    try:
        if train:
            policy.forward_backward(without_routing(datums), loss_fn="ppo_clip", loss_kwargs=loss_kwargs)
            policy.zero_grad()
        else:
            run_forward_only(
                policy.model,
                [replace(microbatches[0], routed_experts=None)],
                pg_collection=policy.process_groups,
                mtp_target_scope=policy.recipe.mtp_target_scope,
            )
        recorded = torch.stack([router.recorded_topk_idx for router in state.routers], dim=1)  # [T_local, L, K]
    finally:
        state.clear()
        for router in state.routers:
            router.recorded_topk_idx = None
    prepared = microbatches[0].routed_experts[0]  # [T_local, L, K], -1 at padding
    valid = prepared[:, 0, 0] >= 0
    hits = (recorded[valid, :, :, None] == prepared[valid, :, None, :]).any(-1).float()
    overlap = torch.tensor([hits.sum(), hits.numel()], device=hits.device, dtype=torch.float64)
    torch.distributed.all_reduce(overlap, group=policy.process_groups.dp_cp)
    # Recorded routing back to per-Datum tensors: undo the CP split and the packing, collect across DP.
    full = unshard_from_cp(microbatches[0], recorded, policy.process_groups.cp)
    entries = all_gather_dp(unpack_sequences(microbatches[0], full, local_only=False), policy.process_groups.dp)
    by_index = {entry.index: entry.values.to(torch.int64) for entry in entries}
    recorded_datums = [replace(datum, routed_experts=by_index[index]) for index, datum in enumerate(datums)]
    return float(overlap[0] / overlap[1]), recorded_datums


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args(_gate_arguments, description=__doc__)
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipe.router_replay = True
    if args.aux_loss_coeff >= 0.0:
        recipe.model.moe_aux_loss_coeff = args.aux_loss_coeff
    prompts, _ = load_data(args, recipe)
    prompts = prompts[: args.prompts_per_round]

    policy = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint)
    policy.cpu()
    rollout = VllmRollout(
        recipe.hf_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_tokens + args.max_tokens + 64,
        enforce_eager=not args.vllm_cuda_graphs,
        return_routed_experts=True,
        moe_layers=policy.router_replay_layers,
    )
    rollout.sleep()
    policy.cuda()
    print_rank_0(f"[gate] policy replays {policy._router_replay.num_layers} MoE layers")

    samples, _ = rollout_round(rollout, policy, prompts, args)
    policy.cuda()
    # Synthetic +1/-1 advantages: with two short samples per prompt the rewards usually tie, the GRPO
    # advantages are all zero, and the policy gradient would vanish, leaving only the auxiliary terms.
    for index, sample in enumerate(samples):
        sample.advantage = 1.0 if index % 2 == 0 else -1.0
    datums = [to_datum(s) for s in samples]
    plain = without_routing(datums)
    tokens = sum(d.num_tokens for d in datums)
    print_rank_0(
        f"[gate] {len(datums)} samples, {tokens} input positions, routing {tuple(datums[0].routed_experts.shape)}"
    )

    def timed(label: str, data: list[Datum]) -> list[torch.Tensor]:
        start = time.perf_counter()
        out = policy.forward_only(data).logprobs
        torch.cuda.synchronize()
        print_rank_0(f"[gate] {label}: {time.perf_counter() - start:.1f} s")
        return out

    native = timed("native forward_only", plain)
    native_again = timed("native forward_only (repeat)", plain)
    replay = timed("replay forward_only", datums)
    replay_again = timed("replay forward_only (repeat)", datums)
    for label, scores in (("native", native), ("replay", replay)):
        if not all(bool(torch.isfinite(lp).all()) for lp in scores):
            raise SystemExit(f"ROUTER REPLAY GATE FAILED: non-finite {label} logprobs")
    report = {
        "vllm_gap_native": consistency(datums, native),
        "vllm_gap_replay": consistency(datums, replay),
        "native_repeat_max_diff": _max_abs_diff(native, native_again),
        "replay_repeat_max_diff": _max_abs_diff(replay, replay_again),
        "replay_vs_native_mean_diff": _mean_abs_diff(replay, native),
    }

    overlap, recorded_datums = record_native_routing(policy, datums)
    recorded_replay = timed("replay of the policy's own routing", recorded_datums)
    report["routing_overlap_vllm_vs_native"] = overlap
    report["own_routing_replay_vs_native_max_diff"] = _max_abs_diff(recorded_replay, native)

    # Training path, two proofs. (a) Replaying the routing the policy itself recorded in a training
    # pass of the same packing reproduces the native gradient norm, with one call (one microbatch per
    # rank) and with four calls (recorded per call). (b) With vllm's routing, the gradient is the same
    # whether the MoE block is recomputed in the backward (queued replay) or kept from the forward.
    # One call versus four differ from each other even natively: the per-microbatch auxiliary losses
    # (MTP, load balancing) are not additive under re-packing and near-tie top-k choices flip with the
    # CP shard, so those numbers are reported, not asserted.
    loss_kwargs = {"clip_low": args.clip_low, "clip_high": args.clip_high, "tis_coef": args.tis_coef, "kl_beta": 0.0}
    for datum, lp in zip(datums, native):
        datum.old_logprobs = lp.cpu()
    plain = without_routing(datums)  # fresh copies so they carry old_logprobs
    own_one = record_native_routing(policy, datums, train=True, loss_kwargs=loss_kwargs)[1]
    own_four: list[Datum] = []
    for chunk in chunks(datums, 4):
        own_four.extend(record_native_routing(policy, chunk, train=True, loss_kwargs=loss_kwargs)[1])
    report["own_routing_overlap_eval_vs_train"] = _overlap(recorded_datums, own_one)
    report["own_routing_overlap_one_call_vs_four_calls"] = _overlap(own_one, own_four)

    def train(label: str, calls: list[list[Datum]]) -> float:
        """Accumulate the calls, step with lr 0; record the grad norm, the loss sum and the mean
        |train logprob - old logprob| (old = the scoring pass, so 0 means the training forward equals
        the scoring forward), plus how often the routers ran in each replay mode."""
        loss_sum = diff = tokens = 0.0
        counts: dict = {}
        original = RouterReplay.get_replay_topk

        def counting(self, *a, **k):
            counts[self.router_replay_action] = counts.get(self.router_replay_action, 0) + 1
            return original(self, *a, **k)

        RouterReplay.get_replay_topk = counting
        try:
            for call_datums in calls:
                out = policy.forward_backward(call_datums, loss_fn="ppo_clip", loss_kwargs=loss_kwargs)
                loss_sum += out.loss_sum
                diff += out.metrics["abs_logprob_diff_sum"]
                tokens += out.num_loss_tokens
        finally:
            RouterReplay.get_replay_topk = original
        result = policy.optim_step(AdamParams(learning_rate=0.0))
        norm = result["grad_norm"]
        # A skipped update or a non-finite number would make every tolerance comparison vacuously pass.
        if not result["update_successful"] or not all(math.isfinite(x) for x in (norm, loss_sum, diff)):
            raise SystemExit(
                f"ROUTER REPLAY GATE FAILED: {label}: update_successful={result['update_successful']}, "
                f"grad_norm={norm}, loss_sum={loss_sum}, logprob_diff={diff}"
            )
        report[f"grad_norm_{label}"] = norm
        report[f"loss_sum_{label}"] = loss_sum
        report[f"train_vs_score_logprob_diff_{label}"] = diff / max(tokens, 1)
        report[f"router_calls_{label}"] = {str(k).split(".")[-1]: v for k, v in counts.items()}
        return norm

    def set_moe_recompute(enabled: bool) -> None:
        for chunk in policy.model:
            for module in chunk.modules():
                if isinstance(module, MoELayer):
                    module.moe_layer_recompute = enabled

    # With recompute on, the training-pass recording is the recompute's routing (it records last);
    # without recompute it is the forward's. Their overlap says whether native recompute re-routes.
    set_moe_recompute(False)
    try:
        own_one_forward = record_native_routing(policy, datums, train=True, loss_kwargs=loss_kwargs)[1]
    finally:
        set_moe_recompute(True)
    report["own_routing_overlap_forward_vs_recompute"] = _overlap(own_one_forward, own_one)

    for datum, lp in zip(datums, replay):
        datum.old_logprobs = lp.cpu()
    native_one = train("native_one_call", [plain])
    train("native_one_call_repeat", [plain])
    native_four = train("native_four_calls", chunks(plain, 4))
    recorded_one = train("own_routing_one_call", [own_one])
    recorded_four = train("own_routing_four_calls", chunks(own_four, 4))
    train("vllm_routing_one_call", [datums])
    replay_four = train("vllm_routing_four_calls", chunks(datums, 4))
    replay_four_repeat = train("vllm_routing_four_calls_repeat", chunks(datums, 4))
    set_moe_recompute(False)
    try:
        train("native_four_calls_no_moe_recompute", chunks(plain, 4))
        replay_four_no_recompute = train("vllm_routing_four_calls_no_moe_recompute", chunks(datums, 4))
    finally:
        set_moe_recompute(True)
    backward_floor = abs(replay_four - replay_four_repeat)

    print_rank_0("[gate] " + json.dumps(report, indent=2))
    floor = max(report["native_repeat_max_diff"], 1e-4)
    failures = []
    if overlap < 0.9:
        failures.append(
            f"routing overlap {overlap:.3f} < 0.9: the prepared routing is not in the router's token order"
        )
    if report["own_routing_replay_vs_native_max_diff"] > 2 * floor:
        failures.append("replaying the policy's own routing does not reproduce the native logprobs")
    if report["replay_repeat_max_diff"] > floor:
        failures.append("replay is not repeatable")
    # Training forward under replay must equal the scoring forward under replay (ratio exactly 1).
    # Natively the two differ (the training-mode MoE path routes some tokens differently from the
    # scoring pass, see own_routing_overlap_forward_vs_recompute); that is reported, not asserted,
    # and so are the gradient norms against native, which is therefore not an exact reference.
    if report["train_vs_score_logprob_diff_vllm_routing_one_call"] > 1e-6:
        failures.append("training forward under replay differs from the scoring forward under replay")
    report["own_routing_vs_native_grad_norm_rel_diff"] = {
        "one_call": abs(recorded_one - native_one) / max(abs(native_one), 1e-12),
        "four_calls": abs(recorded_four - native_four) / max(abs(native_four), 1e-12),
    }
    # Every router must have replayed once in the forward and once in the recompute per microbatch.
    calls = report["router_calls_vllm_routing_four_calls"]
    if calls.get("REPLAY_FORWARD", 0) != calls.get("REPLAY_BACKWARD", 0) or not calls.get("REPLAY_FORWARD"):
        failures.append(f"router replay calls are unbalanced between forward and recompute: {calls}")
    if backward_floor > 1e-3 * abs(replay_four):
        failures.append(f"replayed training is not repeatable: {replay_four} vs {replay_four_repeat}")
    # Recompute replays the same experts, so forward losses agree exactly; gradients differ by the
    # recompute's own rounding (the combine order of expert outputs), well below one percent.
    if report["loss_sum_vllm_routing_four_calls"] != report["loss_sum_vllm_routing_four_calls_no_moe_recompute"]:
        failures.append("replayed forward loss differs with and without MoE recompute")
    if abs(replay_four - replay_four_no_recompute) > max(1e-2 * abs(replay_four), 2 * backward_floor):
        failures.append(
            f"replayed gradient differs with and without MoE recompute: {replay_four} vs {replay_four_no_recompute}"
        )
    policy.close()
    if failures:
        raise SystemExit("ROUTER REPLAY GATE FAILED:\n  - " + "\n  - ".join(failures))
    print_rank_0("ROUTER REPLAY GATE PASSED")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
