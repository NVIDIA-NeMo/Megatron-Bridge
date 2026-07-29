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

"""Export BAGEL's official random initialization before FSDP or optimizer setup."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file


logger = logging.getLogger(__name__)


class _ExportComplete(Exception):
    """Stop the official entry point after the initialization is saved."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagel-repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _configure_transformers_compatibility(official) -> None:
    """Restore the Transformers 4.49 configuration semantics used by BAGEL."""
    from modeling.qwen2 import modeling_qwen2 as qwen2_modeling

    original_load_config = official.Qwen2Config.from_json_file

    def load_config(config_path):
        config = original_load_config(config_path)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None
        return config

    official.Qwen2Config.from_json_file = load_config
    if "default" not in qwen2_modeling.ROPE_INIT_FUNCTIONS:

        def default_rope(config, device, **_kwargs):
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
            exponent = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim
            return 1.0 / (config.rope_theta**exponent), 1.0

        qwen2_modeling.ROPE_INIT_FUNCTIONS["default"] = default_rope


def main() -> None:
    """Run official BAGEL initialization and save its complete native state."""
    args = parse_args()
    bagel_repo = args.bagel_repo.resolve()
    model_path = args.model_path.resolve()
    output = args.output.resolve()
    if args.seed <= 0:
        raise ValueError(f"Seed must be positive, got {args.seed}")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    for name in ("llm_config.json", "vit_config.json", "ae.safetensors", "tokenizer.json"):
        if not (model_path / name).is_file():
            raise ValueError(f"BAGEL model file is missing: {model_path / name}")

    sys.path.insert(0, str(bagel_repo))
    from train import pretrain_unified_navit as official

    _configure_transformers_compatibility(official)
    output.parent.mkdir(parents=True, exist_ok=True)
    original_argv = sys.argv
    with tempfile.TemporaryDirectory(prefix="bagel-init-", dir=output.parent) as temporary_dir:
        temporary_path = Path(temporary_dir)
        temporary_output = temporary_path / output.name

        def export_checkpoint(resume_from, _logger, model, _ema_model, **_kwargs):
            if resume_from is not None:
                raise ValueError("Official initialization export cannot resume a checkpoint")
            if dist.get_rank() == 0:
                state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
                save_file(
                    state,
                    str(temporary_output),
                    metadata={
                        "format_version": "1",
                        "model_seed": str(args.seed * dist.get_world_size()),
                        "world_size": str(dist.get_world_size()),
                    },
                )
                temporary_output.replace(output)
            dist.barrier()
            raise _ExportComplete

        def skip_unused_ema_copy(model):
            return model

        # The following checkpoint hook exits before EMA setup, so cloning the 7B model would only double host memory.
        official.deepcopy = skip_unused_ema_copy
        official.FSDPCheckpoint.try_load_ckpt = staticmethod(export_checkpoint)
        os.environ["WANDB_DIR"] = str(temporary_path)
        sys.argv = [
            str(Path(official.__file__).resolve()),
            "--model_path",
            str(model_path),
            "--finetune_from_hf",
            "True",
            "--global_seed",
            str(args.seed),
            "--max_latent_size",
            "64",
            "--wandb_offline",
            "True",
            "--results_dir",
            str(temporary_path / "results"),
            "--checkpoint_dir",
            str(temporary_path / "checkpoints"),
        ]
        try:
            official.main()
        except _ExportComplete:
            logger.info("Exported official BAGEL initialization to %s", output)
        finally:
            sys.argv = original_argv
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
