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
"""Pipeline-parallel gate: score the same Datums with and without pipeline stages and compare.

Run it twice on the same packed parquet data, with the recipe changed through ``--overrides``:

1. the reference layout (pipeline size 1) with ``--save scores.pt``;
2. the pipelined layout (same context and expert parallel sizes, twice the GPUs for two stages) with
   ``--compare scores.pt``.

Each run checks that every rank received the same gathered result, and that with
``broadcast_output=False`` the last stage has that result while the other stages get ``None``. The
second run then compares per-token logprobs and the mean NLL with the saved reference.

With ``--update`` the runs instead apply one optimizer update on a trainable engine, export the
weights, and save or compare a fixed sample of every tensor together with the loss sum and the
gradient norm. Use deterministic kernels: two runs of one layout then give identical weights, and
the pipelined layout differs from the reference in a few last-bit roundings only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from score_packed_parquet import _check_gathered, _mean_nll, packs_to_datums

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, ForwardOutput, TrainingEngine
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--overrides", default="{}", help="Recipe fields by dotted path, as JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True, help="Offline-packed parquet (training or validation split)")
    parser.add_argument("--num-packs", type=int, default=4)
    parser.add_argument("--save", help="Write the gathered logprobs here (rank 0)")
    parser.add_argument("--compare", help="Compare the gathered logprobs with this file (rank 0)")
    parser.add_argument("--token-tolerance", type=float, default=1e-3)
    parser.add_argument("--nll-tolerance", type=float, default=1e-4)
    parser.add_argument("--update", action="store_true", help="Compare one optimizer update instead of scoring")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sample-size", type=int, default=65536, help="Elements kept per exported tensor")
    parser.add_argument("--update-tolerance", type=float, default=1e-5, help="Largest fraction of changed elements")
    return parser.parse_args()


def _same_on_every_rank(output: ForwardOutput) -> None:
    """Every rank holds the same gathered scores: compare a float64 checksum across the world."""
    checksum = torch.stack([values.double().sum() for values in output.logprobs]).sum().reshape(1).cuda()
    gathered = [torch.empty_like(checksum) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, checksum)
    values = {float(value) for value in gathered}
    assert len(values) == 1, f"ranks hold different scores: checksums {sorted(values)}"


def _compare(reference: list[torch.Tensor], logprobs: list[torch.Tensor], token_tolerance: float) -> None:
    assert len(reference) == len(logprobs), f"{len(logprobs)} datums, reference has {len(reference)}"
    worst, above, total = 0.0, 0, 0
    for index, (expected, values) in enumerate(zip(reference, logprobs)):
        assert expected.shape == values.shape, f"datum {index}: shape {tuple(values.shape)} vs {tuple(expected.shape)}"
        diff = (expected - values).abs()
        worst = max(worst, diff.max().item())
        above += int((diff >= token_tolerance).sum())
        total += diff.numel()
    print_rank_0(f"[compare] {total:,} tokens: max |diff| {worst:.2e}, {above} at or above {token_tolerance:.0e}")
    assert above == 0, f"{above} of {total} token logprobs differ from the reference by {token_tolerance} or more"


def _check_update(args: argparse.Namespace, recipe, datums: list) -> None:
    """Apply one optimizer update and save, or compare, a fixed sample of every exported weight tensor."""
    engine = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True)
    print_rank_0(f"[engine] built; parallel_info={engine.parallel_info}")
    summary = engine.forward_backward(datums)
    result = engine.optim_step(AdamParams(learning_rate=args.learning_rate, weight_decay=0.1))
    grad_norm = result["grad_norm"]
    print_rank_0(
        f"[update] loss_sum {summary.loss_sum:.4f} tokens {summary.num_loss_tokens} grad_norm {grad_norm:.6f}"
    )
    samples: dict[str, torch.Tensor] = {}
    for name, tensor in engine.export(streaming=True):  # every rank drains the stream; rank 0 keeps the sample
        if get_rank_safe() == 0:
            flat = tensor.flatten()
            samples[name] = flat[:: max(1, flat.numel() // args.sample_size)].float().cpu()
    if get_rank_safe() == 0:
        state = {"loss_sum": summary.loss_sum, "tokens": summary.num_loss_tokens, "grad_norm": grad_norm}
        if args.save:
            torch.save({**state, "samples": samples}, args.save)
            print_rank_0(f"[save] wrote {args.save} ({len(samples)} tensors)")
        if args.compare:
            reference = torch.load(args.compare)
            assert reference["tokens"] == state["tokens"], "token counts differ from the reference"
            norm_diff = abs(state["grad_norm"] - reference["grad_norm"]) / reference["grad_norm"]
            print_rank_0(
                f"[compare] loss_sum {state['loss_sum']:.4f} vs {reference['loss_sum']:.4f}; "
                f"grad_norm relative difference {norm_diff:.2e}"
            )
            assert samples.keys() == reference["samples"].keys(), "exported tensor names differ from the reference"
            changed = {name: (values != reference["samples"][name]) for name, values in samples.items()}
            total = sum(mask.numel() for mask in changed.values())
            count = sum(int(mask.sum()) for mask in changed.values())
            worst = sorted(changed, key=lambda name: float(changed[name].float().mean()), reverse=True)[:8]
            print_rank_0(f"[compare] {count} of {total} sampled weights differ ({count / total:.2e})")
            for name in worst:
                gap = (samples[name] - reference["samples"][name]).abs().max().item()
                print_rank_0(
                    f"[compare]   {name}: {float(changed[name].float().mean()):.2e} differ, max |diff| {gap:.2e}"
                )
            assert norm_diff < 1e-5, f"gradient norm differs from the reference by {norm_diff:.2e} (relative)"
            assert count / total <= args.update_tolerance, "the update differs from the reference"
    engine.close()


def main() -> None:
    """Run the pipeline-parallel scoring checks on every rank."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    recipes.apply_recipe_overrides(recipe, json.loads(args.overrides))
    datums = packs_to_datums(args.parquet, args.num_packs)
    print_rank_0(f"[data] {len(datums)} datums, {sum(d.num_tokens for d in datums):,} tokens")
    if args.update:
        _check_update(args, recipe, datums)
        torch.distributed.barrier()
        print_rank_0("PIPELINE PARALLEL GATE PASSED")
        torch.distributed.destroy_process_group()
        return

    engine = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=False)
    info = engine.parallel_info
    print_rank_0(f"[engine] built; parallel_info={info}")
    last_stage = info["pp_rank"] == info["pp_size"] - 1

    start = time.time()
    gathered = engine.forward_only(datums)
    torch.distributed.barrier()
    print_rank_0(f"[score] {len(gathered.logprobs)} entries in {time.time() - start:.1f}s")
    _check_gathered(datums, gathered)
    _same_on_every_rank(gathered)
    total, count = _mean_nll(datums, gathered)
    print_rank_0(f"[score] mean NLL over supervised tokens: {total / count:.6f} ({count:.0f} tokens)")

    held = engine.forward_only(datums, broadcast_output=False)
    if last_stage:
        worst = max((a - b).abs().max().item() for a, b in zip(held.logprobs, gathered.logprobs))
        assert worst < args.token_tolerance, f"a second scoring call differs by {worst}"
    else:
        assert held is None, "broadcast_output=False returned a result off the last stage"
    torch.distributed.barrier()
    print_rank_0(f"[score] broadcast_output=False: result on the last stage only ({info['pp_size']} stages)")

    if get_rank_safe() == 0:
        if args.save:
            torch.save({"logprobs": gathered.logprobs, "nll_sum": total, "nll_count": count}, args.save)
            print_rank_0(f"[save] wrote {args.save}")
        if args.compare:
            reference = torch.load(args.compare)
            _compare(reference["logprobs"], gathered.logprobs, args.token_tolerance)
            nll_diff = abs(total / count - reference["nll_sum"] / reference["nll_count"])
            print_rank_0(f"[compare] mean NLL differs from the reference by {nll_diff:.2e}")
            assert nll_diff < args.nll_tolerance, f"mean NLL differs from the reference by {nll_diff}"

    engine.close()
    torch.distributed.barrier()
    print_rank_0("PIPELINE PARALLEL GATE PASSED")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
