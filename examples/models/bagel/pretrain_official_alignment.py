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

"""Run a short official BAGEL reference curve from an exported initialization."""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse official alignment arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run official BAGEL with controlled data/model RNGs and record its losses."""
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    bridge_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(bridge_root / "examples" / "models" / "bagel" / "data"))
    from dump_official_order import configure_official_data

    sys.path.insert(0, str(args.bagel_repo.resolve()))
    from data.dataset_info import DATASET_INFO
    from export_official_init import _configure_transformers_compatibility
    from train import pretrain_unified_navit as official

    _configure_transformers_compatibility(official)
    rank_seed = args.seed * int(os.environ["WORLD_SIZE"]) + int(os.environ["RANK"])
    original_safe_load = official.yaml.safe_load
    original_data_loader = official.DataLoader
    original_forward = official.Bagel.forward
    reference_rng_seeded = False

    def load_dataset_meta(stream):
        metadata = original_safe_load(stream)
        configure_official_data(args.data_root.resolve(), metadata, DATASET_INFO)
        return metadata

    def build_data_loader(*loader_args, **loader_kwargs):
        loader_kwargs["generator"] = torch.Generator().manual_seed(rank_seed)
        return original_data_loader(*loader_args, **loader_kwargs)

    def reference_forward(model, *model_args, **model_kwargs):
        nonlocal reference_rng_seeded
        official.wandb.log = record_metrics
        if not reference_rng_seeded:
            random.seed(rank_seed)
            np.random.seed(rank_seed % (2**32))
            torch.manual_seed(rank_seed)
            reference_rng_seeded = True
        return original_forward(model, *model_args, **model_kwargs)

    def skip_checkpoint(**_kwargs):
        return None

    def record_metrics(metrics, step):
        if int(os.environ["RANK"]) == 0:
            keys = ("ce", "mse", "lr", "total_norm", "total_mse_tokens", "total_ce_tokens", "total_samples")
            row = {"step": step, **{key: metrics[key] for key in keys}}
            with output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    official.yaml.safe_load = load_dataset_meta
    official.DataLoader = build_data_loader
    official.Bagel.forward = reference_forward
    official.FSDPCheckpoint.fsdp_save_ckpt = staticmethod(skip_checkpoint)
    official.wandb.log = record_metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    original_argv = sys.argv
    with tempfile.TemporaryDirectory(prefix="bagel-official-", dir=output.parent) as temporary_dir:
        temporary_path = Path(temporary_dir)
        sys.argv = [
            str(Path(official.__file__).resolve()),
            "--model_path",
            str(args.model_path.resolve()),
            "--finetune_from_hf",
            "True",
            "--resume_from",
            str(args.checkpoint.resolve().parent),
            "--resume_model_only",
            "True",
            "--global_seed",
            str(args.seed),
            "--max_latent_size",
            "64",
            "--dataset_config_file",
            str(args.dataset_config.resolve()),
            "--data_seed",
            str(args.seed),
            "--num_workers",
            "1",
            "--total_steps",
            str(args.train_iters),
            "--log_every",
            "1",
            "--sharding_strategy",
            "FULL_SHARD",
            "--text_cond_dropout_prob",
            "0.1",
            "--vit_cond_dropout_prob",
            "0.4",
            "--vae_cond_dropout_prob",
            "0.1",
            "--wandb_offline",
            "True",
            "--results_dir",
            str(temporary_path / "results"),
            "--checkpoint_dir",
            str(temporary_path / "checkpoints"),
        ]
        try:
            official.main()
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    main()
