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
#
"""Merge Megatron-Bridge LoRA adapters back into dense model weights (model-agnostic).

The script expects two checkpoints:
1. A **LoRA fine-tuning checkpoint** that contains the adapter weights.
2. A **base/pre-trained checkpoint** that holds the original dense weights.

If the base path is not provided, the script will look for ``run_config.yaml``
inside the LoRA checkpoint and read ``checkpoint.pretrained_checkpoint``.

It works for **any model architecture** supported by ``AutoBridge`` and trained
with Megatron-Bridge's `LoRALinear` wrapper (e.g., Llama, Nemotron, Qwen,
DeepSeek, Phi, etc.).

Usage
-----
python merge_lora.py \
    --lora-checkpoint path/to/finetune_ckpt \
    --hf-model-path   path/to/hf_model \
    --output          path/to/merged_ckpt \
    [--pretrained path/to/base_ckpt]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from megatron.core import dist_checkpointing

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA, LoRAMerge, VLMLoRA
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    apply_peft_adapter_filter_to_state_dict,
)
from megatron.bridge.training.model_load_save import save_megatron_model
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge Megatron-Bridge LoRA adapters into base weights",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lora-checkpoint", required=True, help="LoRA fine-tuning checkpoint directory")
    parser.add_argument("--output", required=True, help="Where to store the merged checkpoint")
    parser.add_argument(
        "--hf-model-path",
        required=True,
        help="HuggingFace model name or local path supplying the config of the architecture.",
    )
    parser.add_argument(
        "--pretrained",
        help="Base (dense) checkpoint. If omitted, resolved from run_config.yaml in the LoRA checkpoint.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _resolve_pretrained(lora_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    cfg_path = lora_dir / "run_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("run_config.yaml not found in LoRA checkpoint and --pretrained not supplied")
    cfg = read_run_config(str(cfg_path))
    base = cfg.get("checkpoint", {}).get("pretrained_checkpoint")
    if base is None:
        raise ValueError("pretrained_checkpoint missing in run_config.yaml; pass --pretrained")
    return Path(base).expanduser().resolve()


# -----------------------------------------------------------------------------
# Merge routine
# -----------------------------------------------------------------------------


def merge_lora(
    base_dir: Path,
    lora_dir: Path,
    out_dir: Path,
    hf_model_path: str,
) -> None:
    """
    Merge LoRA adapter weights back into the base model.

    Args:
        base_dir (Path): Path to the directory containing the base model checkpoint (the dense, pre-trained model).
        lora_dir (Path): Path to the directory containing the LoRA fine-tuned checkpoint.
        out_dir (Path): Path to the directory where the merged model checkpoint should be saved.
        hf_model_path (str): HuggingFace model name or local path to the model architecture/configuration.

    This routine reconstructs the model architecture from HuggingFace config,
    loads the dense base model weights, then loads the LoRA adapter weights
    (optionally reading LoRA hyperparameters from run_config.yaml), and merges
    the LoRA deltas back into the model weights, resulting in a fully merged checkpoint.
    """
    print_rank_0(f"Loading base model from {base_dir}")
    bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)

    # Single-GPU context; set all parallel dims to 1
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.expert_model_parallel_size = 1
    model_provider.expert_tensor_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.initialize_model_parallel(seed=0)

    # 1) Load base model weights
    model = bridge.load_megatron_model(str(base_dir), wrap_with_ddp=False)

    # 2) Patch the model with LoRA adapter *structure* (no weights yet)
    # Load LoRA hyper-parameters from the fine-tuning run_config.yaml so we
    # recreate the exact adapter structure (rank, alpha, etc.) that was used
    # during training. Fallback to defaults when the config is missing.
    peft_cfg: dict = {}
    peft_class = LoRA
    cfg_file = lora_dir / "run_config.yaml"
    if cfg_file.exists():
        try:
            run_cfg_dict = read_run_config(str(cfg_file))
            peft_cfg = run_cfg_dict.get("peft", {}) or {}

            # Determine which PEFT class to use based on _target_ field
            target = peft_cfg.get("_target_", "")
            if "VLMLoRA" in target:
                peft_class = VLMLoRA

            allowed_keys = {
                "target_modules",
                "dim",
                "alpha",
                "dropout",
                "dropout_position",
                "freeze_language_model",
                "freeze_vision_model",
                "freeze_vision_projection",
            }
            peft_cfg = {k: v for k, v in peft_cfg.items() if k in allowed_keys}
        except Exception as err:
            logger.warning(f"Failed to read LoRA settings from {cfg_file}: {err}. Using defaults.")
    else:
        logger.warning(
            "run_config.yaml not found in LoRA checkpoint; using default LoRA settings for structure patching"
        )

    # Initialize the PEFT object with the loaded hyper-parameters
    print_rank_0(f"Using PEFT class: {peft_class.__name__}")
    lora_peft = peft_class(**peft_cfg)
    model = lora_peft(model, training=False)

    # 3) Load weights from the fine-tuned checkpoint
    print_rank_0(f"Loading LoRA adapter weights from {lora_dir}")
    # Generate full sharded_state_dict describing all model tensors
    sharded_state_dict = _generate_model_state_dict(model, {})
    # Keep only LoRA adapter tensors (and any other trainable parameters) so we don't read unnecessary dense weights.
    sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, lora_peft)

    # Load those tensors from the checkpoint directory
    loaded_sd = dist_checkpointing.load(sharded_state_dict, str(lora_dir))
    # dist_checkpointing.load returns the same nested dict structure; we need the model section
    model_section_key = "model" if "model" in loaded_sd else next(k for k in loaded_sd if k.startswith("model"))
    adapter_sd = loaded_sd[model_section_key]
    # Load adapter weights into the base model (strict=False so missing dense weights are ignored)
    model[0].load_state_dict(adapter_sd, strict=False)

    # 4) Merge adapters
    merge = LoRAMerge()
    merged_model = merge(model[0], training=False)
    for m in merged_model.modules():
        if hasattr(m, "adapter"):
            delattr(m, "adapter")

    # Recursively replace any remaining LoRALinear wrappers with their underlying linear modules
    def _unwrap_lora(module):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.to_wrap)
            else:
                _unwrap_lora(child)

    _unwrap_lora(merged_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    print_rank_0(f"Saving merged checkpoint to {out_dir}")
    save_megatron_model([merged_model], out_dir)

    print_rank_0("Merge complete ✔")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    """Main function to merge LoRA adapter weights back into the base model."""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    lora_dir = Path(args.lora_checkpoint).expanduser().resolve()
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_dir}")
    base_dir = _resolve_pretrained(lora_dir, args.pretrained)
    if not base_dir.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {base_dir}")
    merge_lora(
        base_dir=base_dir,
        lora_dir=lora_dir,
        out_dir=Path(args.output).expanduser().resolve(),
        hf_model_path=args.hf_model_path,
    )


if __name__ == "__main__":
    main()
