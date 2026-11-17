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

"""Prepare a GPT-SFT recipe's training data and record the tokens of every item training reads.

The dataset is prepared as rank 0 of a training run prepares it (download, JSONL and, with offline packing, the
packed rows), so running this first keeps tokenization off the GPU nodes. Every training item is then read in
dataloader order and the token count of each sequence it holds is written as JSON: one list per item, with one
sequence per sample, or the sequences of one packed row with offline packing. Alignment padding is not counted.

``compare_runs.py`` turns the file into runtime tokens per second for a run with ``dataloader_type="single"``, which
reads items in this order.

Example::

    python examples/agentic_sft/item_tokens.py \\
        --recipe qwen35_text_35b_a3b_sft_long_context_8gpu_gb300_fp8mx_offline_dynamic_cp_config \\
        --output offline_items.json --num-workers 64 "dataset.hf_dataset.split='SWE_Rebench[:4000]'"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch


_RECIPE_RUNNER_DIR = Path(__file__).resolve().parents[2] / "scripts" / "training"
if str(_RECIPE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_RUNNER_DIR))

logger = logging.getLogger(__name__)


def item_sequence_tokens(
    dataset: torch.utils.data.Dataset, *, num_items: int | None = None, num_workers: int = 0
) -> list[list[int]]:
    """Return the token count of every sequence of the first ``num_items`` items of ``dataset``, in order.

    Args:
        dataset: A dataset whose ``collate_fn`` returns global-batch packing's unpacked samples.
        num_items: Items to read; all of them when ``None``.
        num_workers: Loader worker processes; lazily tokenized samples take seconds each.

    Returns:
        One list per item with the unpadded length of each of its sequences.
    """
    total = len(dataset) if num_items is None else min(num_items, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(total)),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
    )
    return [[int(sample["original_seq_len"]) for sample in samples] for samples in loader]


def capacity_shares(tokens: list[int], tokens_per_rank: int) -> dict[int, tuple[float, float]]:
    """Return, per smallest CP group size that fits, the share of sequences and of tokens that need it.

    Args:
        tokens: Token count of every sequence.
        tokens_per_rank: Tokens one rank holds, ``model.max_seqlen_per_dp_cp_rank``.

    Returns:
        ``{group size: (share of sequences, share of tokens)}`` in increasing group size.
    """
    counts: dict[int, list[int]] = {}
    for length in tokens:
        size = 1
        while size * tokens_per_rank < length:
            size *= 2
        entry = counts.setdefault(size, [0, 0])
        entry[0] += 1
        entry[1] += length
    total_tokens = sum(tokens)
    return {size: (n / len(tokens), t / total_tokens) for size, (n, t) in sorted(counts.items())}


def main(argv: list[str] | None = None) -> None:
    """Prepare the recipe's data and write its items' token counts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, help="Recipe with a GPTSFTDatasetConfig and global-batch packing.")
    parser.add_argument("--output", type=Path, required=True, help="JSON file to write.")
    parser.add_argument("--num-items", type=int, help="Items to read; default: the whole training split.")
    parser.add_argument("--num-workers", type=int, default=8, help="Tokenizer and loader worker processes.")
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides, as for run_recipe.py.")
    args = parser.parse_args(argv)

    from recipe_runner import (
        apply_cli_overrides,
        load_recipe,
        sync_model_dataset_sequence_length,
        sync_offline_packing_alignment,
    )

    from megatron.bridge.data.builders import GPTSFTDatasetBuilder, GPTSFTDatasetConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    cfg = apply_cli_overrides(load_recipe(args.recipe), args.overrides)
    cfg = sync_model_dataset_sequence_length(sync_offline_packing_alignment(cfg))
    dataset_config = cfg.dataset
    if not isinstance(dataset_config, GPTSFTDatasetConfig) or not dataset_config.enable_global_batch_packing:
        parser.error(f"{args.recipe} does not read a GPTSFTDatasetConfig with enable_global_batch_packing.")
    if dataset_config.enable_offline_packing and args.num_workers > 1:
        dataset_config.offline_packing_specs.num_tokenizer_workers = args.num_workers

    builder = GPTSFTDatasetBuilder(config=dataset_config, tokenizer=build_tokenizer(cfg.tokenizer))
    train_dataset = builder.build()[0]
    items = item_sequence_tokens(train_dataset, num_items=args.num_items, num_workers=args.num_workers)
    args.output.write_text(json.dumps(items))

    tokens = [length for item in items for length in item]
    logger.info(
        "%d items, %d sequences, %d tokens (%.0f per item, longest sequence %d) -> %s",
        len(items),
        len(tokens),
        sum(tokens),
        sum(tokens) / len(items),
        max(tokens),
        args.output,
    )
    shares = capacity_shares(tokens, cfg.model.max_seqlen_per_dp_cp_rank)
    for size, (sequence_share, token_share) in shares.items():
        logger.info("needs CP%d: %.1f%% of sequences, %.1f%% of tokens", size, 100 * sequence_share, 100 * token_share)


if __name__ == "__main__":
    main()
