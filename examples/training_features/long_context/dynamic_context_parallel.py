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

"""Show how Megatron-Core Dynamic CP packs variable-length samples.

This is intentionally not a training script. It uses Megatron-Core's
``DefaultDynamicCPScheduler`` to schedule a few toy sequence lengths, then
prints the packed THD metadata that the scheduled data iterator would yield:
``tokens``, ``cu_seqlens``, ``cu_seqlens_padded``, ``max_seqlen``, and
``local_cp_size``.

Run after switching the Megatron-Core submodule to a dev commit that contains
Dynamic CP:

    ./scripts/switch_mcore.sh dev
    uv sync
    uv run python examples/training_features/long_context/dynamic_context_parallel.py
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import Any

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


def load_dynamic_cp_helpers() -> tuple[type[Any], Any]:
    """Load the MCore dev Dynamic CP scheduler helpers."""
    try:
        from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
        from megatron.core.datasets.data_schedule_utils import dcp_gpus_needed
    except ImportError as exc:
        raise SystemExit(
            "This example needs the Megatron-Core dev Dynamic CP scheduler. "
            "Run `./scripts/switch_mcore.sh dev && uv sync` first."
        ) from exc

    return DefaultDynamicCPScheduler, dcp_gpus_needed


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


def print_bridge_config_knobs(args: argparse.Namespace) -> None:
    """Print the Bridge config fields users set when enabling DCP for training."""
    logger.info("Bridge config knobs for a real run:")
    logger.info("  cfg.model.dynamic_context_parallel = True")
    logger.info('  cfg.model.sequence_packing_scheduler = "default_dynamic_cp"')
    logger.info("  cfg.model.max_seqlen_per_dp_cp_rank = %s", args.max_seqlen_per_rank)
    logger.info("  cfg.model.min_dynamic_context_parallel_size = %s", args.min_dynamic_cp_size)
    logger.info("  cfg.train.micro_batch_size = 1")
    logger.info("")


def print_input_lengths(args: argparse.Namespace, dcp_gpus_needed: Any) -> None:
    """Print the unscheduled sample lengths and per-sample CP demand."""
    logger.info("Input samples:")
    for sample_id, length in enumerate(args.lengths):
        gpus_needed = dcp_gpus_needed(length, args.max_seqlen_per_rank, args.min_dynamic_cp_size)
        logger.info("  sample %s: length=%s, gpus_needed=%s", sample_id, length, gpus_needed)
    logger.info("")


def print_scheduled_packing(args: argparse.Namespace, sample_id_groups: list[list[list[int]]]) -> None:
    """Print packed tensors for every scheduled microbatch and DPxCP rank."""
    samples = {sample_id: make_sample(sample_id, length) for sample_id, length in enumerate(args.lengths)}

    logger.info("Scheduled packed microbatches over dp_size=%s, cp_size=%s:", args.dp_size, args.cp_size)
    for microbatch_id, microbatch_group in enumerate(sample_id_groups):
        logger.info("microbatch %s:", microbatch_id)
        for dcp_rank, sample_ids in enumerate(microbatch_group):
            if not sample_ids:
                logger.info("  dpxcp_rank %s: idle", dcp_rank)
                continue

            rank_samples = [samples[sample_id] for sample_id in sample_ids]
            local_cp_size = local_cp_size_for_rank(sample_ids, microbatch_group)
            packed = pack_samples(rank_samples, local_cp_size)
            lengths = [int(sample.original_seq_len.item()) for sample in rank_samples]
            token_preview = packed["tokens"][: min(8, packed["tokens"].numel())].tolist()

            logger.info(
                "  dpxcp_rank %s: sample_ids=%s, lengths=%s, local_cp_size=%s",
                dcp_rank,
                sample_ids,
                lengths,
                int(packed["local_cp_size"].item()),
            )
            logger.info(
                "    tokens.shape=%s, cu_seqlens=%s, max_seqlen=%s, tokens[:8]=%s",
                tuple(packed["tokens"].shape),
                packed["cu_seqlens"].tolist(),
                int(packed["max_seqlen"].item()),
                token_preview,
            )
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
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Run the local Dynamic CP packing demo."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.dp_size < 1 or args.cp_size < 1:
        raise ValueError("dp-size and cp-size must be positive")

    DefaultDynamicCPScheduler, dcp_gpus_needed = load_dynamic_cp_helpers()
    scheduler = DefaultDynamicCPScheduler(
        args.max_seqlen_per_rank,
        args.cp_size,
        args.dp_size,
        None,
        min_cp_size=args.min_dynamic_cp_size,
    )
    sample_id_seqlens = list(enumerate(args.lengths))
    sample_id_groups = scheduler.get_groups_and_subsamples(sample_id_seqlens)

    print_bridge_config_knobs(args)
    print_input_lengths(args, dcp_gpus_needed)
    print_scheduled_packing(args, sample_id_groups)
    logger.info("Forward step input: call MCore get_batch_on_this_rank_for_sequence_packing(..., dynamic_cp=True).")
    logger.info("It reads local_cp_size, chooses the matching dynamic CP group, then slices this THD batch.")


if __name__ == "__main__":
    main()
