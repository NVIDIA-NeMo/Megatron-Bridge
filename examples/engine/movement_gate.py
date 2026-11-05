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
"""Movement gate: cpu()/cuda() between accumulation calls must not change the update.

Two trainable engines are built from the same checkpoint and accumulate the same four calls (one
original parquet pack per microbatch, via ``PackReplayEngine``). The second engine is moved to host
memory after the second call, a frozen reference engine is built and scores the same packs while the
policy is away, is closed again, and the policy returns with ``cuda()``. The accumulated gradient
buffers, the gradient norms and the master parameters after the step must match the uninterrupted
engine. An idle round trip after the step and a second update then confirm the optimizer state
survived. Freed GPU memory and transfer times are printed.
"""

import argparse
import json
import logging
import time

import torch
from sft_from_recipe import PackReplayEngine, _chunks, _grad_delta, _param_delta, iter_pack_batches

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import AdamParams, TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--packs-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def _timed_move(engine: TrainingEngine, direction: str) -> None:
    """Move and check that the GPU allocation changes by what the engine reports it moved (within 2%)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    engine.cpu() if direction == "cpu" else engine.cuda()
    torch.cuda.synchronize()
    delta = torch.cuda.memory_allocated() - before
    expected = -engine.offloaded_bytes if direction == "cpu" else None
    print_rank_0(
        f"[move] {direction}(): {time.perf_counter() - start:.1f} s, GPU allocated change {delta / 2**30:+.1f} GiB, "
        f"residual {torch.cuda.memory_allocated() / 2**30:.1f} GiB"
    )
    if expected is not None:
        assert abs(delta - expected) <= 0.02 * abs(expected), f"freed {delta} bytes, engine moved {expected}"


def _assert_matching_update(plain: TrainingEngine, moved: TrainingEngine, adam: AdamParams, label: str) -> None:
    grad_diff, grad_rel = _grad_delta(plain, moved)
    print_rank_0(f"[gate:{label}] DP-reduced grad sums: max |diff|={grad_diff:.3e} (relative {grad_rel:.3e})")
    step_plain = plain.optim_step(adam)
    step_moved = moved.optim_step(adam)
    assert step_plain["update_successful"] and step_moved["update_successful"]
    rel = abs(step_plain["grad_norm"] - step_moved["grad_norm"]) / max(step_plain["grad_norm"], 1e-12)
    diff, fraction = _param_delta(plain, moved, lr=adam.learning_rate)
    print_rank_0(
        f"[gate:{label}] grad_norm plain={step_plain['grad_norm']:.6f} moved={step_moved['grad_norm']:.6f} "
        f"rel diff={rel:.3e}; max |master param diff|={diff:.3e}; fraction with a different update={fraction:.3e}"
    )
    assert grad_rel < 1e-4, f"reduced gradients differ by {grad_rel:.3e} after movement"
    assert rel < 1e-4, f"grad norm differs by {rel:.3e} after movement"
    assert fraction < 1e-5, f"{fraction:.3e} of elements took a different update after movement"


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    adam = AdamParams(learning_rate=args.lr)
    datums = next(iter_pack_batches(args.parquet, args.packs_per_step))
    chunks = _chunks(datums, 4)

    plain = PackReplayEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True)
    moved = PackReplayEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True)
    for chunk in chunks:
        plain.forward_backward(chunk)
    for index, chunk in enumerate(chunks):
        out = moved.forward_backward(chunk)
        if index == 1:
            _timed_move(moved, "cpu")  # open window: live gradients travel with the engine
            assert moved.device == "cpu"
            try:
                moved.forward_only(chunk)
            except RuntimeError as error:
                print_rank_0(f"[move] offloaded engine rejects compute as expected: {error}")
            else:
                raise AssertionError("forward_only on an offloaded engine did not raise")
            reference = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=False)
            scored = reference.forward_only(chunk)
            assert all(torch.isfinite(lp).all() for lp in scored.logprobs)
            print_rank_0(f"[move] reference scored {len(scored.logprobs)} Datums while the policy was offloaded")
            reference.close()
            _timed_move(moved, "cuda")
            assert moved.device == "cuda"
    assert out.num_loss_tokens > 0
    _assert_matching_update(plain, moved, adam, "mid-window move")

    _timed_move(moved, "cpu")  # idle engine: empty gradient storage is simply released
    _timed_move(moved, "cuda")
    plain.forward_backward(datums)
    moved.forward_backward(datums)
    _assert_matching_update(plain, moved, adam, "after idle move")

    plain.close()
    moved.close()
    print_rank_0("MOVEMENT GATE PASSED")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
