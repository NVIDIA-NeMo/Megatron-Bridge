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

"""Pre-pack SFT training data for a recipe that uses packed sequences.

Run this before submitting a training job so packing is not performed on
GPU compute nodes. Packed .parquet files are written to the dataset cache
directory defined by the recipe.

Usage (inside container with PYTHONPATH=/opt/megatron-lm:/opt/Megatron-Bridge/src):

    python scripts/training/pack_sft_data.py \\
        --recipe <recipe_name>

Set HF_HOME / NEMO_HOME if your dataset and model caches are not under ~/.cache.
"""

import argparse
import inspect
import sys
from dataclasses import fields
from pathlib import Path


def main() -> None:
    """Pre-pack SFT dataset for the given recipe."""
    parser = argparse.ArgumentParser(description="Pre-pack SFT dataset for a packed-sequence recipe.")
    parser.add_argument(
        "--recipe",
        required=True,
        help="Recipe name for a packed-sequence SFT dataset config.",
    )
    parser.add_argument("--seq-length", type=int, default=None, help="Optional sequence length override.")
    parser.add_argument("--hf-path", default=None, help="Optional Hugging Face model ID or local snapshot path.")
    parser.add_argument(
        "--train-input-path", default=None, help="Optional processed JSONL path for the training split."
    )
    parser.add_argument(
        "--val-input-path", default=None, help="Optional processed JSONL path for the validation split."
    )
    parser.add_argument(
        "--packed-train-data-path", default=None, help="Optional output path for packed training data."
    )
    parser.add_argument(
        "--packed-val-data-path", default=None, help="Optional output path for packed validation data."
    )
    parser.add_argument("--packed-metadata-path", default=None, help="Optional output path for packing metadata.")
    args = parser.parse_args()

    import megatron.bridge.recipes as all_recipes
    from megatron.bridge.data.builders.finetuning_dataset import FinetuningDatasetBuilder
    from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data
    from megatron.bridge.training.config import DataloaderConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    recipe_fn = getattr(all_recipes, args.recipe, None)
    if recipe_fn is None:
        sys.exit(f"Error: recipe '{args.recipe}' not found. Check available recipes in megatron.bridge.recipes.")

    sig = inspect.signature(recipe_fn)
    params = sig.parameters
    has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    recipe_kwargs = {}
    if args.seq_length is not None:
        if "seq_length" in params or has_var_keyword:
            recipe_kwargs["seq_length"] = args.seq_length
        else:
            sys.exit(f"Error: recipe '{args.recipe}' does not accept a 'seq_length' parameter.")
    if args.hf_path is not None:
        if "hf_path" in params or has_var_keyword:
            recipe_kwargs["hf_path"] = args.hf_path
        else:
            sys.exit(f"Error: recipe '{args.recipe}' does not accept an 'hf_path' parameter.")

    cfg = recipe_fn(**recipe_kwargs)

    if cfg.dataset is None:
        sys.exit("Error: recipe has no dataset configuration.")
    if not getattr(cfg.dataset, "enable_offline_packing", False):
        sys.exit(f"Error: recipe '{args.recipe}' does not enable offline packed sequences.")
    offline_packing_specs = getattr(cfg.dataset, "offline_packing_specs", None)
    if offline_packing_specs is None:
        sys.exit(f"Error: recipe '{args.recipe}' has no offline packing specs.")

    # Cap tokenizer workers to avoid /dev/shm OOM from multiprocessing shared memory.
    # Default is -1 (all CPUs) which exhausts /dev/shm even on CPU nodes.
    # Use 1 worker so serial materialization avoids multiprocessing shared memory
    # (see packed_sequence._materialize_dataset_items).
    offline_packing_specs.num_tokenizer_workers = 1

    print(f"Recipe:   {args.recipe}")
    print(f"Seq len:  {offline_packing_specs.packed_sequence_size}")
    print(f"Workers:  {offline_packing_specs.num_tokenizer_workers} (single-threaded, no /dev/shm)")
    print()

    print("Building tokenizer...")
    tokenizer = build_tokenizer(cfg.tokenizer)

    print("Packing dataset (skipped if already cached)...")
    dataset_config = cfg.dataset
    dataloader_field_names = {field.name for field in fields(DataloaderConfig)}

    builder = FinetuningDatasetBuilder(
        tokenizer=tokenizer,
        **{
            field.name: getattr(dataset_config, field.name)
            for field in fields(dataset_config)
            if field.name not in dataloader_field_names
        },
    )

    custom_pack_paths = [
        args.train_input_path,
        args.val_input_path,
        args.packed_train_data_path,
        args.packed_val_data_path,
        args.packed_metadata_path,
    ]
    if any(custom_pack_paths):
        if not args.train_input_path:
            sys.exit("Error: --train-input-path is required when using explicit pack paths.")
        if not args.packed_train_data_path:
            sys.exit("Error: --packed-train-data-path is required when using explicit pack paths.")

        packed_metadata_path = Path(args.packed_metadata_path) if args.packed_metadata_path else builder.pack_metadata
        prepare_packed_sequence_data(
            input_path=Path(args.train_input_path),
            output_path=Path(args.packed_train_data_path),
            output_metadata_path=packed_metadata_path,
            packed_sequence_size=offline_packing_specs.packed_sequence_size,
            tokenizer=tokenizer,
            max_seq_length=cfg.dataset.seq_length,
            seed=cfg.dataset.seed,
            dataset_kwargs=cfg.dataset.dataset_kwargs,
            pad_seq_to_mult=offline_packing_specs.pad_seq_to_mult,
            num_tokenizer_workers=offline_packing_specs.num_tokenizer_workers,
        )

        if args.val_input_path and args.packed_val_data_path:
            prepare_packed_sequence_data(
                input_path=Path(args.val_input_path),
                output_path=Path(args.packed_val_data_path),
                output_metadata_path=packed_metadata_path,
                packed_sequence_size=offline_packing_specs.packed_sequence_size,
                tokenizer=tokenizer,
                max_seq_length=cfg.dataset.seq_length,
                seed=cfg.dataset.seed,
                dataset_kwargs=cfg.dataset.dataset_kwargs,
                pad_seq_to_mult=offline_packing_specs.pad_seq_to_mult,
                num_tokenizer_workers=offline_packing_specs.num_tokenizer_workers,
            )
        elif args.val_input_path or args.packed_val_data_path:
            sys.exit("Error: --val-input-path and --packed-val-data-path must be provided together.")

        print("Done.")
        return

    builder.prepare_packed_data()

    print("Done.")


if __name__ == "__main__":
    main()
