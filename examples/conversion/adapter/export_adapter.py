#!/usr/bin/env python3
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

"""
Export LoRA adapter weights from a Megatron-Bridge PEFT checkpoint to
HuggingFace PEFT format (``adapter_config.json`` + ``adapter_model.safetensors``).

The default path runs on CPU and is suitable for models whose architecture can be
materialized without CUDA. Large hybrid models can use the distributed GPU path
by passing TP/PP/EP settings that match the adapter checkpoint.

The output can be loaded directly with::

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("<hf-model-path>")
    model = PeftModel.from_pretrained(base, "./my_adapter")

Usage::

    uv run python examples/conversion/adapter/export_adapter.py \\
        --hf-model-path meta-llama/Llama-3.2-1B \\
        --lora-checkpoint /path/to/finetune_ckpt \\
        --output ./my_adapter \\
        --exclude-adapter-base-prefix mtp.layers
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing, parallel_state
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA, VLMLoRA
from megatron.bridge.peft.utils import enable_legacy_shared_expert_adapter_loading
from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    apply_peft_adapter_filter_to_state_dict,
)
from megatron.bridge.training.utils.checkpoint_utils import read_run_config
from megatron.bridge.utils.activation_map import str_to_dtype
from megatron.bridge.utils.common_utils import get_local_rank_preinit


logger = logging.getLogger(__name__)

_SUPPORTED_EXPORT_DTYPES = {torch.float32, torch.float16, torch.bfloat16}


def _parse_dtype(dtype: str) -> torch.dtype:
    try:
        parsed_dtype = str_to_dtype(dtype.lower())
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    if parsed_dtype not in _SUPPORTED_EXPORT_DTYPES:
        supported = ", ".join(sorted(str(dtype).replace("torch.", "") for dtype in _SUPPORTED_EXPORT_DTYPES))
        raise argparse.ArgumentTypeError(
            f"Unsupported adapter export dtype: {parsed_dtype}. Supported values: {supported}"
        )
    return parsed_dtype


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Megatron-Bridge LoRA adapter to HuggingFace PEFT format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-model-path",
        required=True,
        help="HuggingFace model name or local path (architecture + base weights).",
    )
    parser.add_argument(
        "--lora-checkpoint",
        required=True,
        help="Megatron-Bridge distributed checkpoint containing LoRA adapter weights.",
    )
    parser.add_argument("--output", type=Path, default=Path("./my_adapter"))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype",
        type=_parse_dtype,
        default=torch.float32,
        help="Dtype used to materialize the model for distributed GPU export.",
    )
    parser.add_argument(
        "--exclude-adapter-base-prefix",
        action="append",
        default=[],
        help=(
            "Megatron adapter base prefix to skip during export, before HF mapping lookup. "
            "Can be specified multiple times; e.g. `mtp.layers` excludes MTP adapters."
        ),
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for distributed GPU export.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size for distributed GPU export.")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size for distributed GPU export.")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size for distributed GPU export.")
    parser.add_argument("--sequence-parallel", action="store_true", help="Enable sequence parallelism.")
    return parser.parse_args()


def _load_lora_config(ckpt_path: Path) -> LoRA | VLMLoRA:
    peft_class: type[LoRA | VLMLoRA] = LoRA
    peft_cfg: dict = {}
    cfg_file = ckpt_path / "run_config.yaml"
    if not cfg_file.exists() and ckpt_path.parent != ckpt_path:
        cfg_file = ckpt_path.parent / "run_config.yaml"
    if cfg_file.exists():
        try:
            run_cfg_dict = read_run_config(str(cfg_file))
        except Exception as err:
            logger.warning("Failed to read LoRA settings from %s: %s. Using defaults.", cfg_file, err)
        else:
            peft_cfg = run_cfg_dict.get("peft", {}) or {}
            if "VLMLoRA" in peft_cfg.get("_target_", ""):
                peft_class = VLMLoRA
            vlm_only_keys = {"freeze_language_model", "freeze_vision_model", "freeze_vision_projection"}
            allowed_keys = {
                "target_modules",
                "exclude_modules",
                "dim",
                "alpha",
                "dropout",
                "dropout_position",
                "normalize_moe_lora",
                "share_expert_adapters",
            }
            if peft_class is VLMLoRA:
                allowed_keys |= vlm_only_keys
            peft_cfg = {key: value for key, value in peft_cfg.items() if key in allowed_keys}
    return peft_class(**peft_cfg)


def _get_loaded_model_key(loaded_sd: Mapping[str, object], ckpt_path: Path) -> str:
    if "model" in loaded_sd:
        return "model"

    model_key = next((key for key in loaded_sd if key.startswith("model")), None)
    if model_key is None:
        raise RuntimeError(f"Checkpoint at {ckpt_path} has no 'model' key. Available keys: {list(loaded_sd.keys())}")
    return model_key


def _uses_distributed_export(args: argparse.Namespace) -> bool:
    return args.tp > 1 or args.pp > 1 or args.ep > 1 or args.etp > 1


def _configure_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed adapter export requires CUDA. Use the default CPU path for TP=PP=EP=ETP=1.")
    local_rank = get_local_rank_preinit()
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _export_adapter_distributed(args: argparse.Namespace) -> None:
    device = _configure_cuda_device()
    ckpt_path = Path(args.lora_checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PEFT checkpoint not found: {ckpt_path}")
    config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
    bridge = AutoBridge.from_hf_config(config)
    lora = _load_lora_config(ckpt_path)

    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = args.tp
    provider.pipeline_model_parallel_size = args.pp
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = args.etp
    provider.sequence_parallel = args.sequence_parallel
    provider.pipeline_dtype = args.dtype
    provider.params_dtype = args.dtype
    provider.finalize()
    provider.register_pre_wrap_hook(lambda chunks: lora(chunks, training=False))
    try:
        provider.initialize_model_parallel(seed=0)

        model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=False,
            init_model_with_meta_device=False,
        )
        model = [chunk.to(device) for chunk in model]
        if len(model) != 1:
            raise RuntimeError(
                "Distributed adapter export currently supports exactly one local model chunk; "
                f"got {len(model)}. Use pipeline parallel size 1 without virtual pipeline parallelism."
            )

        sharded_state_dict = _generate_model_state_dict(model, {})
        sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, lora)
        legacy_shared_expert_adapter = enable_legacy_shared_expert_adapter_loading(
            model, sharded_state_dict, ckpt_path
        )
        if legacy_shared_expert_adapter:
            sharded_state_dict = _generate_model_state_dict(model, {})
            sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, lora)
        loaded_sd = dist_checkpointing.load(
            sharded_state_dict,
            str(ckpt_path),
            validate_access_integrity=not legacy_shared_expert_adapter,
        )
        model_key = _get_loaded_model_key(loaded_sd, ckpt_path)
        model[0].load_state_dict(loaded_sd[model_key], strict=False)

        bridge.save_hf_adapter(
            model,
            path=args.output,
            peft_config=lora,
            base_model_name_or_path=args.hf_model_path,
            exclude_adapter_base_prefixes=tuple(args.exclude_adapter_base_prefix),
        )
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    """Export a Megatron-Bridge PEFT checkpoint to HuggingFace PEFT format."""
    args = parse_args()

    if _uses_distributed_export(args):
        _export_adapter_distributed(args)
    else:
        if args.dtype != torch.float32:
            raise ValueError("--dtype is only supported by distributed GPU export; CPU export uses float32.")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model_path, trust_remote_code=args.trust_remote_code)
        bridge.export_adapter_ckpt(
            peft_checkpoint=args.lora_checkpoint,
            output_path=args.output,
            exclude_adapter_base_prefixes=tuple(args.exclude_adapter_base_prefix),
        )


if __name__ == "__main__":
    main()
