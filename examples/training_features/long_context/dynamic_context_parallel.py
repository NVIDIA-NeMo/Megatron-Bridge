#!/usr/bin/env python3
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

"""Show how Megatron-Core Dynamic CP (DCP v2) packs variable-length samples.

This is intentionally not a training script. It uses Megatron-Core's
``DefaultDynamicCPScheduler`` to schedule a few toy sequence lengths, then
prints the packed THD metadata that the scheduled data iterator would yield:
``tokens``, ``cu_seqlens``, ``cu_seqlens_padded``, ``max_seqlen``, and
``local_cp_size``, together with the scheduler's own estimate of the work each
rank receives.

It runs on the Megatron-Core ``main`` pin this repository ships, so no
submodule switch is needed:

    uv run python examples/training_features/long_context/dynamic_context_parallel.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToySample:
    """One variable-length sample before DCP scheduling and packing."""

    sample_id: int
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    original_seq_len: torch.Tensor
    padded_seq_len: torch.Tensor


def load_dynamic_cp_helpers() -> tuple[type[Any], Callable[[int], list[int]], Callable[[int, int, bool], int]]:
    """Load the Megatron-Core Dynamic CP scheduler and its topology helpers."""
    try:
        from megatron.core.context_parallel.groups import get_valid_dynamic_context_parallel_group_sizes
        from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
        from megatron.core.datasets.data_schedule_utils import get_packed_sequence_length
    except ImportError as exc:
        raise SystemExit(
            "This example needs the Megatron-Core Dynamic CP v2 scheduler, which the Megatron-Core main pin of "
            "this repository provides. Run `uv sync` and try again."
        ) from exc

    return DefaultDynamicCPScheduler, get_valid_dynamic_context_parallel_group_sizes, get_packed_sequence_length


def make_sample(sample_id: int, length: int) -> ToySample:
    """Create one deterministic toy sample."""
    tokens = torch.arange(sample_id * 1000, sample_id * 1000 + length, dtype=torch.long)
    labels = torch.roll(tokens, shifts=-1, dims=0)
    loss_mask = torch.ones(length, dtype=torch.float32)
    loss_mask[-1] = 0.0

    return ToySample(
        sample_id=sample_id,
        tokens=tokens,
        labels=labels,
        loss_mask=loss_mask,
        position_ids=torch.arange(length, dtype=torch.long),
        original_seq_len=torch.tensor([length], dtype=torch.int32),
        padded_seq_len=torch.tensor([length], dtype=torch.int32),
    )


def pack_samples(samples: list[ToySample], local_cp_size: int) -> dict[str, torch.Tensor]:
    """Pack samples into the THD-style tensors consumed by Dynamic CP."""
    lengths = [int(sample.original_seq_len.item()) for sample in samples]
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)

    return {
        "tokens": torch.cat([sample.tokens for sample in samples], dim=0),
        "labels": torch.cat([sample.labels for sample in samples], dim=0),
        "loss_mask": torch.cat([sample.loss_mask for sample in samples], dim=0),
        "position_ids": torch.cat([sample.position_ids for sample in samples], dim=0),
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_padded": cu_seqlens.clone(),
        "max_seqlen": torch.tensor(max(lengths), dtype=torch.int32),
        "local_cp_size": torch.tensor(local_cp_size, dtype=torch.int32),
    }


def local_cp_size_for_rank(rank_sample_ids: list[int], microbatch_group: list[list[int]]) -> int:
    """Mirror MCore's local CP size attached to each packed microbatch."""
    if not rank_sample_ids:
        return 0

    first_sample_id = rank_sample_ids[0]
    return sum(1 for sample_ids in microbatch_group if first_sample_id in sample_ids)


def minimum_group_size(
    length: int,
    *,
    group_sizes: list[int],
    max_seqlen_per_rank: int,
    get_packed_sequence_length: Callable[[int, int, bool], int],
) -> int:
    """Return the smallest enabled CP group whose ranks can each hold their share of ``length`` tokens."""
    for size in group_sizes:
        if get_packed_sequence_length(length, size, False) / size <= max_seqlen_per_rank:
            return size
    raise ValueError(
        f"A sample of {length} tokens does not fit any enabled group {group_sizes} at {max_seqlen_per_rank} "
        "tokens per rank."
    )


def estimated_rank_work(sample_lengths: list[int], local_cp_size: int) -> float:
    """Return the scheduler's relative cost model for one rank: the sum of ``length ** 2 / group size``."""
    return sum(length * length for length in sample_lengths) / local_cp_size


def print_bridge_config_knobs(args: argparse.Namespace) -> None:
    """Print the Bridge config fields users set when enabling DCP for training."""
    logger.info("Bridge config knobs for a real run:")
    logger.info("  cfg.dataset.enable_global_batch_packing = True")
    logger.info("  cfg.model.dynamic_context_parallel = True  # selects the default_dynamic_cp scheduler")
    logger.info("  cfg.model.max_seqlen_per_dp_cp_rank = %s", args.max_seqlen_per_rank)
    logger.info("  cfg.model.min_dynamic_context_parallel_size = %s", args.min_dynamic_cp_size)
    logger.info("  cfg.train.micro_batch_size = 1")
    logger.info("")


def print_input_lengths(
    args: argparse.Namespace,
    *,
    valid_group_sizes: list[int],
    configured_group_sizes: list[int],
    get_packed_sequence_length: Callable[[int, int, bool], int],
) -> None:
    """Print the unscheduled sample lengths and the smallest CP group each one needs."""
    enabled_sizes = [size for size in configured_group_sizes if size >= args.min_dynamic_cp_size]
    logger.info("Valid group sizes for the dp x cp = %s pool: %s", args.dp_size * args.cp_size, valid_group_sizes)
    logger.info(
        "Enabled sizes (configured and >= min_dynamic_cp_size=%s): %s", args.min_dynamic_cp_size, enabled_sizes
    )
    logger.info("Input samples:")
    for sample_id, length in enumerate(args.lengths):
        needed = minimum_group_size(
            length,
            group_sizes=enabled_sizes,
            max_seqlen_per_rank=args.max_seqlen_per_rank,
            get_packed_sequence_length=get_packed_sequence_length,
        )
        logger.info("  sample %s: length=%s, min_group_size=%s", sample_id, length, needed)
    logger.info("")


def print_scheduled_packing(args: argparse.Namespace, sample_id_groups: list[list[list[int]]]) -> None:
    """Print packed tensors and estimated work for every scheduled microbatch and DPxCP rank."""
    samples = {sample_id: make_sample(sample_id, length) for sample_id, length in enumerate(args.lengths)}

    logger.info("Scheduled packed microbatches over dp_size=%s, cp_size=%s:", args.dp_size, args.cp_size)
    for microbatch_id, microbatch_group in enumerate(sample_id_groups):
        logger.info("microbatch %s:", microbatch_id)
        rank_work: list[float] = []
        for dcp_rank, sample_ids in enumerate(microbatch_group):
            if not sample_ids:
                logger.info("  dpxcp_rank %s: idle", dcp_rank)
                rank_work.append(0.0)
                continue

            rank_samples = [samples[sample_id] for sample_id in sample_ids]
            local_cp_size = local_cp_size_for_rank(sample_ids, microbatch_group)
            packed = pack_samples(rank_samples, local_cp_size)
            lengths = [int(sample.original_seq_len.item()) for sample in rank_samples]
            token_preview = packed["tokens"][: min(8, packed["tokens"].numel())].tolist()
            rank_work.append(estimated_rank_work(lengths, local_cp_size))

            logger.info(
                "  dpxcp_rank %s: sample_ids=%s, lengths=%s, local_cp_size=%s, estimated_work=%.0f",
                dcp_rank,
                sample_ids,
                lengths,
                int(packed["local_cp_size"].item()),
                rank_work[-1],
            )
            logger.info(
                "    tokens.shape=%s, cu_seqlens=%s, max_seqlen=%s, tokens[:8]=%s",
                tuple(packed["tokens"].shape),
                packed["cu_seqlens"].tolist(),
                int(packed["max_seqlen"].item()),
                token_preview,
            )
        logger.info("  critical path (busiest rank, relative units): %.0f", max(rank_work))
    logger.info("")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the packing demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 96, 64, 48, 32, 24, 16, 8],
        help="Toy sample lengths to schedule and pack.",
    )
    parser.add_argument("--max-seqlen-per-rank", type=int, default=64)
    parser.add_argument("--min-dynamic-cp-size", type=int, default=1)
    parser.add_argument(
        "--cp-group-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the scheduler to a subset of the valid group sizes (default: all valid sizes).",
    )
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Run the local Dynamic CP packing demo."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.dp_size < 1 or args.cp_size < 1:
        raise ValueError("dp-size and cp-size must be positive")

    DefaultDynamicCPScheduler, get_valid_group_sizes, get_packed_sequence_length = load_dynamic_cp_helpers()
    valid_group_sizes = get_valid_group_sizes(args.dp_size * args.cp_size)
    try:
        scheduler = DefaultDynamicCPScheduler(
            args.max_seqlen_per_rank,
            args.cp_size,
            args.dp_size,
            None,
            min_cp_size=args.min_dynamic_cp_size,
            cp_group_sizes=args.cp_group_sizes,
        )
        sample_id_groups = scheduler.get_groups_and_subsamples(list(enumerate(args.lengths)))
    except ValueError as exc:
        raise SystemExit(f"The scheduler rejected this configuration: {exc}") from exc

    print_bridge_config_knobs(args)
    print_input_lengths(
        args,
        valid_group_sizes=valid_group_sizes,
        configured_group_sizes=list(scheduler.cp_group_sizes),
        get_packed_sequence_length=get_packed_sequence_length,
    )
    print_scheduled_packing(args, sample_id_groups)
    logger.info("Forward step input: Bridge calls MCore get_batch_on_this_rank_for_sequence_packing(dynamic_cp=True).")
    logger.info("It reads local_cp_size, picks the matching pre-created dynamic CP group, then slices this THD batch.")


if __name__ == "__main__":
    main()
