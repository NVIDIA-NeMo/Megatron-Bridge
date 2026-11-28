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

"""Train a dynamic-CP recipe on mock variable-length data instead of its dataset, for benchmarking.

The recipe's model, parallelism, precision, optimizer and execution settings are kept. Only its data is replaced:
``MockGPTSFTDatasetConfig`` yields random-token samples whose lengths mix short and long sequences, as an agentic
tool-use workload does, so no dataset, tokenizer or checkpoint is read and the model trains from scratch. The loss
is not meaningful.

``--packing online`` packs each step's samples with the global-batch scheduler. ``--packing offline`` first packs
the samples into rows of the recipe's sequence length; a step then takes a global batch of rows and the scheduler
re-bins their samples.

Examples::

    srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \\
        python examples/agentic_sft/mock_data.py --packing online train.train_iters=200

    # Static CP on the same samples
    srun --nodes=4 --ntasks-per-node=4 --gpus-per-node=4 \\
        python examples/agentic_sft/mock_data.py --packing online train.train_iters=200 \\
        model.dynamic_context_parallel=false
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer

_TRAINING_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "training"
if str(_TRAINING_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_SCRIPTS_DIR))

DEFAULT_RECIPE = "qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_dynamic_cp_config"
PACKING_MODES = ("online", "offline")
LENGTH_BUCKET_SAMPLE_SHARES = (0.4, 0.3, 0.2, 0.1)
"""Share of samples that need at most a CP1, CP2, CP4 and CP8 group of a CP8 recipe."""
COMPLETION_FRACTION = 0.2
RANDOM_SEED = 1234


def use_mock_data(cfg: ConfigContainer, *, packing: str) -> ConfigContainer:
    """Replace a dynamic-CP recipe's dataset and tokenizer with mock variable-length samples.

    Args:
        cfg: A recipe with global-batch packing whose base context-parallel size is 8.
        packing: ``"online"``, or ``"offline"`` to pack the samples into rows of ``model.seq_length`` first.

    Returns:
        ``cfg``, reading ``MockGPTSFTDatasetConfig`` samples and running no validation.

    Raises:
        ValueError: If ``packing`` is unknown or the recipe's base CP size does not match the length buckets.
    """
    from megatron.bridge.data.builders import MockGPTSFTDatasetConfig
    from megatron.bridge.data.packing import PackedSequenceSpecs

    if packing not in PACKING_MODES:
        raise ValueError(f"packing must be one of {PACKING_MODES}, got {packing!r}.")
    model = cfg.model
    context_parallel_size = model.context_parallel_size
    if 2 ** (len(LENGTH_BUCKET_SAMPLE_SHARES) - 1) != context_parallel_size:
        raise ValueError(
            f"The mock length buckets are defined for a base CP size of "
            f"{2 ** (len(LENGTH_BUCKET_SAMPLE_SHARES) - 1)}, got {context_parallel_size}."
        )

    # Random token IDs, so no tokenizer is read; the null tokenizer only reports the vocabulary size.
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.hf_tokenizer_kwargs = {}
    cfg.tokenizer.vocab_size = model.vocab_size

    offline = packing == "offline"
    cfg.dataset = MockGPTSFTDatasetConfig(
        seq_length=model.seq_length,
        vocab_size=model.vocab_size,
        length_bucket_fractions=LENGTH_BUCKET_SAMPLE_SHARES,
        completion_fraction=COMPLETION_FRACTION,
        random_seed=RANDOM_SEED,
        # 256-wide attention heads run on FlashAttention, which needs bins without padding between sequences.
        fold_alignment_padding=True,
        dataloader_type="single",
        num_workers=2,
        enable_offline_packing=offline,
        # Pack on lengths aligned like the runtime samples: 2 x the base CP size.
        offline_packing_specs=PackedSequenceSpecs(
            packed_sequence_size=model.seq_length, pad_seq_to_mult=2 * context_parallel_size
        )
        if offline
        else None,
    )
    cfg.validation.eval_iters = 0
    # A benchmark from random weights on random tokens: nothing is loaded, and nothing is saved unless
    # checkpoint.save is set on the command line.
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.save = None
    return cfg


def main(argv: list[str] | None = None) -> None:
    """Train the recipe on mock data."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE, help="Dynamic-CP recipe to run on mock data.")
    parser.add_argument("--packing", choices=PACKING_MODES, required=True, help="Pack samples per step or offline.")
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides, as for run_recipe.py.")
    args = parser.parse_args(argv)

    from recipe_runner import (
        apply_cli_overrides,
        apply_runtime_environment,
        bootstrap_recipe_environment,
        load_forward_step,
        load_recipe,
        run_config,
        sync_model_dataset_sequence_length,
        sync_offline_packing_alignment,
    )
    from run_recipe import IMPORT_TIME_RECIPE_ENV_VARS

    cfg = apply_cli_overrides(use_mock_data(load_recipe(args.recipe), packing=args.packing), args.overrides)
    # Some recipe environment variables are read when a library is imported: re-execute with them in place.
    if any(name not in os.environ for name in IMPORT_TIME_RECIPE_ENV_VARS & set(cfg.env_vars or {})):
        cfg = bootstrap_recipe_environment(
            cfg, script_path=str(Path(__file__).resolve()), argv=list(argv) if argv is not None else sys.argv[1:]
        )
    else:
        cfg = apply_runtime_environment(cfg)
    cfg = sync_model_dataset_sequence_length(sync_offline_packing_alignment(cfg))
    run_config(config=cfg, mode="pretrain", step_func=load_forward_step("gpt_step", mode="pretrain"))


if __name__ == "__main__":
    main()
