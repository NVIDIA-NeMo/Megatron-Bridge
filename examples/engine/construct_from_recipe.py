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
"""Construction gate: construct TrainingEngines from an engine recipe on every rank.

Builds a trainable actor and a frozen reference from the same recipe, prints parallel info,
parameter counts and memory, closes the actor, verifies the reference still works and that the
actor's model was actually released. Run under torchrun / srun with the recipe's world size, e.g.::

    python examples/engine/construct_from_recipe.py \
        --recipe qwen35_text_35b_a3b_rl_16gpu_gb200_bf16 \
        --recipe-kwargs '{"hf_path": "/path/to/local/hf_text_model"}' \
        --checkpoint /path/to/megatron_ckpt \
        [key=value ...]   # optional Hydra-style recipe overrides, e.g. model.context_parallel_size=4

or by model, through ``TrainingEngine(model, seq_length=...)`` (the recipe is resolved from the HF config,
the detected GPU family and the context length; ``key=value`` pairs become constructor overrides)::

    python examples/engine/construct_from_recipe.py \
        --model /path/to/local/hf_text_model --seq-length 32768 \
        --checkpoint /path/to/megatron_ckpt [optimizer.lr=1e-6 ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
import weakref

import torch
import yaml

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import TrainingEngine
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default=None, help="Recipe function name in megatron.bridge.engine.recipes")
    parser.add_argument("--recipe-kwargs", default="{}", help="JSON dict of keyword arguments for the recipe function")
    parser.add_argument("--model", default=None, help="HF model ID or directory; builds via TrainingEngine(model)")
    parser.add_argument("--seq-length", type=int, default=None, help="Packed tokens per microbatch, with --model")
    parser.add_argument("--hardware", default=None, help="GPU family for --model, e.g. GB300 (default: detect)")
    parser.add_argument("--checkpoint", default=None, help="Pretrained Megatron/HF checkpoint for both engines")
    parser.add_argument("--skip-reference", action="store_true", help="Only build the trainable actor")
    args, overrides = parser.parse_known_args()
    return args, overrides


def _count_params(model: list[torch.nn.Module]) -> tuple[int, int]:
    total = sum(p.numel() for chunk in model for p in chunk.parameters())
    trainable = sum(p.numel() for chunk in model for p in chunk.parameters() if p.requires_grad)
    return total, trainable


def _memory_gb() -> str:
    allocated = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    return f"allocated {allocated:.1f} GB, reserved {reserved:.1f} GB"


def _assert_released(ref: weakref.ReferenceType, label: str) -> None:
    """Fail if a closed engine's model chunk is still alive; print who holds it for debugging."""
    gc.collect()
    chunk = ref()
    if chunk is None:
        return
    holders = sorted({type(r).__name__ for r in gc.get_referrers(chunk)})
    raise RuntimeError(f"{label} model chunk still referenced after close() by: {holders}")


def _param_checksum(model: list[torch.nn.Module]) -> float:
    """Sum |p| over all parameters of all ranks: a collective that proves the model and runtime are usable."""
    local = torch.zeros((), dtype=torch.float64, device="cuda")
    for chunk in model:
        for parameter in chunk.parameters():
            local += parameter.detach().abs().sum().double()
    torch.distributed.all_reduce(local)
    return local.item()


def _override_dict(overrides: list[str]) -> dict:
    """``key=value`` pairs as a dotted-path dict for ``TrainingEngine(overrides=...)``; YAML scalar values."""
    result = {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"override {item!r} is not key=value")
        result[key] = yaml.safe_load(value)
    return result


def _make_builder(args: argparse.Namespace, overrides: list[str]):
    """Return ``build(trainable)`` for the chosen construction path."""
    if (args.model is None) == (args.recipe is None):
        raise SystemExit("pass exactly one of --model or --recipe")
    if args.model is not None:
        if args.seq_length is None:
            raise SystemExit("--seq-length is required with --model")
        override_dict = _override_dict(overrides)

        def build(trainable: bool) -> TrainingEngine:
            return TrainingEngine(
                args.model,
                seq_length=args.seq_length,
                hardware=args.hardware,
                checkpoint_path=args.checkpoint,
                overrides=override_dict,
                trainable=trainable,
            )

        return build
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    if overrides:
        recipe = process_config_with_overrides(recipe, cli_overrides=overrides)
    return lambda trainable: TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=trainable)


def main() -> None:
    """Build an actor and a frozen reference engine, close the actor, and verify the reference survives."""
    logging.basicConfig(level=logging.INFO)
    args, overrides = _parse_args()
    build = _make_builder(args, overrides)

    start = time.time()
    actor = build(trainable=True)
    torch.distributed.barrier()
    total, trainable = _count_params(actor.model)
    print_rank_0(f"[actor] built in {time.time() - start:.1f}s; parallel_info={actor.parallel_info}")
    print_rank_0(f"[actor] local params total={total:,} trainable={trainable:,}; {_memory_gb()}")
    print_rank_0(
        f"[actor] optimizer={type(actor.optimizer).__name__}; param checksum={_param_checksum(actor.model):.6e}"
    )

    actor_chunk_ref = weakref.ref(actor.model[0])
    if args.skip_reference:
        actor.close()
        _assert_released(actor_chunk_ref, "actor")
        torch.distributed.barrier()
        print_rank_0(f"[actor] closed and released; {_memory_gb()}")
        torch.distributed.destroy_process_group()
        return

    start = time.time()
    reference = build(trainable=False)
    torch.distributed.barrier()
    total, trainable = _count_params(reference.model)
    print_rank_0(
        f"[reference] built in {time.time() - start:.1f}s; local params total={total:,} trainable={trainable:,}"
    )
    print_rank_0(f"[reference] optimizer={reference.optimizer}; {_memory_gb()}")
    checksum_before = _param_checksum(reference.model)

    actor.close()
    _assert_released(actor_chunk_ref, "actor")
    torch.distributed.barrier()
    print_rank_0(f"[actor] closed and released; {_memory_gb()}")

    checksum_after = _param_checksum(reference.model)
    if checksum_after != checksum_before:
        raise RuntimeError(
            f"reference parameters changed after closing the actor: {checksum_before} -> {checksum_after}"
        )
    print_rank_0(f"[reference] still usable after actor.close(); checksum={checksum_after:.6e}; closed={actor.closed}")

    reference_chunk_ref = weakref.ref(reference.model[0])
    reference.close()
    _assert_released(reference_chunk_ref, "reference")
    torch.distributed.barrier()
    print_rank_0(f"[reference] closed and released; {_memory_gb()}; CONSTRUCTION GATE PASSED")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
