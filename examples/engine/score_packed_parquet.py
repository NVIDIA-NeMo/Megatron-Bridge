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
"""Scoring gate: score packed SFT data with ``TrainingEngine.forward_only`` and check the output contract.

Reads the first ``--num-packs`` rows of an offline-packed parquet file (columns ``input_ids``,
``seq_start_id``, ``loss_mask``), expands every pack into one Datum per sequence (inputs are the
sequence minus its last token, targets the sequence shifted by one, loss mask as stored), and then:

1. scores them with ``gather_output=True`` and checks every Datum comes back in order with all positions;
2. scores with ``gather_output=False`` and checks the local shards agree with the gathered result;
3. scores the Datums in reverse order and checks per-Datum logprobs are unchanged;
4. prints the mean NLL over supervised tokens for comparison with the baseline's validation loss;
5. optionally repeats the scoring on a trainable engine and checks its gradient buffers stay zero.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pyarrow.parquet as pq
import torch

import megatron.bridge.engine.recipes as recipes
from megatron.bridge.engine import Datum, ForwardOutput, TrainingEngine
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--recipe-kwargs", default="{}")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parquet", required=True, help="Offline-packed parquet (training or validation split)")
    parser.add_argument("--num-packs", type=int, default=4)
    parser.add_argument("--with-trainable", action="store_true", help="Also score through a trainable engine")
    return parser.parse_args()


def packs_to_datums(path: str, num_packs: int) -> list[Datum]:
    """Expand packed rows into one Datum per logical sequence (same shift and mask as the SFT collate)."""
    table = pq.read_table(path).slice(0, num_packs)
    datums: list[Datum] = []
    for row in table.to_pylist():
        ids = torch.tensor(row["input_ids"], dtype=torch.int64)
        mask = torch.tensor(row["loss_mask"], dtype=torch.float32)
        boundaries = list(row["seq_start_id"]) + [len(ids)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start < 2:
                continue
            datums.append(
                Datum(input_ids=ids[start : end - 1], target_ids=ids[start + 1 : end], loss_mask=mask[start : end - 1])
            )
    return datums


def _check_gathered(datums: list[Datum], output: ForwardOutput) -> None:
    assert output.sample_indices is None and len(output.logprobs) == len(datums), "gathered output size mismatch"
    for index, (datum, values) in enumerate(zip(datums, output.logprobs)):
        assert values.shape == (datum.num_tokens,), f"datum {index}: shape {tuple(values.shape)}"
        assert torch.isfinite(values).all(), f"datum {index}: non-finite logprobs"
        assert (values <= 1e-3).all(), f"datum {index}: logprob above zero ({values.max():.4f})"


def _check_local_matches(gathered: ForwardOutput, local: ForwardOutput) -> int:
    checked = 0
    for index, positions, values in zip(local.sample_indices, local.token_positions, local.logprobs):
        reference = gathered.logprobs[index][positions]
        max_diff = (reference - values).abs().max().item() if positions.numel() else 0.0
        assert max_diff < 1e-3, f"local shard for datum {index} differs from gathered result by {max_diff}"
        checked += positions.numel()
    return checked


def _mean_nll(datums: list[Datum], output: ForwardOutput) -> tuple[float, float]:
    total, count = 0.0, 0.0
    for datum, values in zip(datums, output.logprobs):
        total += float((-values * datum.loss_mask).sum())
        count += float(datum.loss_mask.sum())
    return total, count


def main() -> None:
    """Run the scoring checks on every rank."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    recipe = getattr(recipes, args.recipe)(**json.loads(args.recipe_kwargs))
    datums = packs_to_datums(args.parquet, args.num_packs)
    tokens = sum(d.num_tokens for d in datums)
    print_rank_0(f"[data] {len(datums)} datums from {args.num_packs} packs, {tokens:,} tokens")

    engine = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=False)
    print_rank_0(f"[reference] built; parallel_info={engine.parallel_info}")

    start = time.time()
    gathered = engine.forward_only(datums, gather_output=True)
    torch.distributed.barrier()
    print_rank_0(f"[score] gathered: {len(gathered.logprobs)} entries in {time.time() - start:.1f}s")
    _check_gathered(datums, gathered)
    total, count = _mean_nll(datums, gathered)
    print_rank_0(f"[score] mean NLL over supervised tokens: {total / count:.4f} ({count:.0f} tokens)")

    local = engine.forward_only(datums, gather_output=False)
    checked = _check_local_matches(gathered, local)
    counts = torch.tensor([checked], device="cuda")
    torch.distributed.all_reduce(counts)
    tp = engine.parallel_info["tp_size"]
    assert counts.item() == tokens * tp, f"local shards cover {counts.item()} positions, expected {tokens * tp}"
    print_rank_0(f"[score] local shards reconstruct the gathered result ({checked} positions on rank 0)")

    reversed_out = engine.forward_only(list(reversed(datums)), gather_output=True)
    for reversed_index, values in enumerate(reversed_out.logprobs):
        reference = gathered.logprobs[len(datums) - 1 - reversed_index]
        max_diff = (values - reference).abs().max().item()
        assert max_diff < 1e-3, f"reordering changed logprobs by {max_diff}"
    print_rank_0("[score] reversed input order gives the same per-datum logprobs (tolerance 1e-3)")

    if args.with_trainable:
        engine.close()
        actor = TrainingEngine.from_recipe(recipe, checkpoint_path=args.checkpoint, trainable=True)
        out = actor.forward_only(datums[: max(1, len(datums) // 4)], gather_output=True)
        grads = sum(
            float(p.main_grad.abs().sum())
            for chunk in actor.model
            for p in chunk.parameters()
            if getattr(p, "main_grad", None) is not None
        )
        assert grads == 0.0, f"forward_only on a trainable engine touched gradient buffers: {grads}"
        print_rank_0(f"[actor] forward_only ok on {len(out.logprobs)} datums; grad buffers untouched")
        actor.close()
    else:
        engine.close()
    torch.distributed.barrier()
    print_rank_0("SCORING GATE PASSED")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
