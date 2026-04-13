# \!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Pre-pack SFT training data for a recipe that uses packed sequences.

Run this before submitting a training job so packing is not performed on
GPU compute nodes. Packed .parquet files are written to the dataset cache
directory defined by the recipe.

Usage (inside container with PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src):

    python examples/models/gpt_oss/pack_sft_data.py \\
        --recipe gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config

Set HF_HOME / NEMO_HOME if your dataset and model caches are not under ~/.cache.
"""

import argparse
import sys
from dataclasses import fields


def main() -> None:
    """Pre-pack SFT dataset for the given recipe."""
    parser = argparse.ArgumentParser(description="Pre-pack SFT dataset for a packed-sequence recipe.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe name, e.g. gpt_oss_20b_sft_openmathinstruct2_thinking_packed_config",
    )
    args = parser.parse_args()

    from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
    from megatron.bridge.data.builders.hf_dataset import HFDatasetBuilder, HFDatasetConfig
    from megatron.bridge.recipes.gpt_oss import gpt_oss as gpt_oss_recipes
    from megatron.bridge.training.config import DataloaderConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    recipe_fn = getattr(gpt_oss_recipes, args.recipe, None)
    if recipe_fn is None:
        sys.exit(
            f"Error: recipe '{args.recipe}' not found. Check available recipes in megatron.bridge.recipes.gpt_oss."
        )

    cfg = recipe_fn()

    if cfg.dataset is None:
        sys.exit("Error: recipe has no dataset configuration.")
    if cfg.dataset.packed_sequence_specs is None:
        sys.exit(f"Error: recipe '{args.recipe}' does not use packed sequences.")

    # Cap tokenizer workers to avoid /dev/shm OOM from multiprocessing shared memory.
    # Default is -1 (all CPUs) which exhausts /dev/shm even on CPU nodes.
    # Use 1 worker to avoid /dev/shm OOM: num_workers==1 runs single-threaded
    # with no multiprocessing shared memory (see packed_sequence._retrieve_tokenized).
    cfg.dataset.packed_sequence_specs.num_tokenizer_workers = 1

    print(f"Recipe:   {args.recipe}")
    print(f"Seq len:  {cfg.dataset.packed_sequence_specs.packed_sequence_size}")
    print(f"Workers:  {cfg.dataset.packed_sequence_specs.num_tokenizer_workers} (single-threaded, no /dev/shm)")
    print()

    print("Building tokenizer...")
    tokenizer = build_tokenizer(cfg.tokenizer)

    print("Packing dataset (skipped if already cached)...")
    dataset_config = cfg.dataset
    dataloader_field_names = {field.name for field in fields(DataloaderConfig)}

    BuilderClass = HFDatasetBuilder if isinstance(dataset_config, HFDatasetConfig) else FinetuningDatasetBuilder
    builder = BuilderClass(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field.name not in dataloader_field_names
        },
    )
    builder.prepare_packed_data()

    print("Done.")


if __name__ == "__main__":
    main()
