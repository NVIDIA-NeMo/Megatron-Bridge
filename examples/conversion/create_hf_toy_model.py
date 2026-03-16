#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""
Utility script that materializes a significantly smaller HuggingFace checkpoint
from an existing model configuration.  It is primarily intended to help bridge
functional / quantization tests (e.g., the Qwen3 MoE conversion suites) avoid
downloading extremely large public checkpoints.

Example:
    ```bash
    uv run python examples/conversion/create_hf_toy_model.py \
        --hf-model-id Qwen/Qwen3-30B-A3B \
        --output-dir /tmp/qwen3_toy \
        --num-hidden-layers 2 \
        --num-experts 4
    ```

The script works by:
1. Loading the original configuration via `AutoConfig` so that all model-specific
   attributes (e.g., gating settings, rotary params) stay in sync with the
   upstream release.
2. Overriding a handful of size-related knobs (hidden layers, number of experts,
   etc.) so that the instantiated model is tiny but structurally compatible.
3. Saving the resulting random-weight checkpoint alongside a tokenizer so tests
   can treat it like any other HF model directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reduced HuggingFace Causal LM checkpoint for tests.")
    parser.add_argument(
        "--hf-model-id",
        default="Qwen/Qwen3-30B-A3B",
        help="Source HuggingFace model id to pull the base config from.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Optional tokenizer model id. Defaults to --hf-model-id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the toy checkpoint will be saved.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of transformer layers to keep in the toy model.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=4,
        help="Total MoE experts per layer for the toy model.",
    )
    parser.add_argument(
        "--num-experts-per-tok",
        type=int,
        default=None,
        help="Experts routed per token. Defaults to --num-experts.",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Optional override for the MoE FFN size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Torch seed applied before checkpoint creation.",
    )
    parser.add_argument(
        "--disable-remote-code-trust",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code when loading from HuggingFace.",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def _adjust_config(
    config,
    *,
    num_hidden_layers: int,
    num_experts: int,
    num_experts_per_tok: Optional[int],
    moe_intermediate_size: Optional[int],
) -> None:
    """Mutate config(s) in-place so they match requested layer/expert topology."""

    def _adjust_one(cfg) -> None:
        cfg.num_hidden_layers = num_hidden_layers

        if hasattr(cfg, "max_window_layers"):
            cfg.max_window_layers = min(cfg.max_window_layers, num_hidden_layers)

        if hasattr(cfg, "layer_types"):
            cfg.layer_types = cfg.layer_types[:num_hidden_layers]

        mlp_only_layers = getattr(cfg, "mlp_only_layers", [])
        if isinstance(mlp_only_layers, (list, tuple)):
            cfg.mlp_only_layers = [layer for layer in mlp_only_layers if layer < num_hidden_layers]

        # Kimi-style configs may use n_routed_experts while many others use num_experts.
        for field in ("num_experts", "n_routed_experts"):
            if hasattr(cfg, field):
                setattr(cfg, field, num_experts)

        if hasattr(cfg, "num_experts_per_tok"):
            cfg.num_experts_per_tok = (
                num_experts_per_tok
                if num_experts_per_tok is not None
                else min(num_experts, getattr(cfg, "num_experts_per_tok", num_experts))
            )

        if hasattr(cfg, "router_top_k"):
            cfg.router_top_k = min(num_experts, getattr(cfg, "num_experts_per_tok", num_experts))

        if moe_intermediate_size is not None and hasattr(cfg, "moe_intermediate_size"):
            cfg.moe_intermediate_size = moe_intermediate_size

    _adjust_one(config)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        _adjust_one(text_config)


def _save_tokenizer(output_dir: Path, tokenizer_id: str, *, trust_remote_code: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


def _save_processor(output_dir: Path, model_id: str, *, trust_remote_code: bool) -> None:
    """Save the AutoProcessor alongside the model so VL toy models can process images."""
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    processor.save_pretrained(output_dir)
    print(f"  Processor ({type(processor).__name__}) saved to {output_dir}")


def main() -> None:
    """Main entry point."""
    args = _parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_id = args.tokenizer_id or args.hf_model_id
    trust_remote_code = bool(args.trust_remote_code)

    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(
        args.hf_model_id,
        trust_remote_code=trust_remote_code,
    )
    config.torch_dtype = torch.bfloat16

    _adjust_config(
        config,
        num_hidden_layers=args.num_hidden_layers,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        moe_intermediate_size=args.moe_intermediate_size,
    )

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    model = model.bfloat16()
    model.save_pretrained(output_dir, safe_serialization=True)

    _save_tokenizer(output_dir, tokenizer_id, trust_remote_code=trust_remote_code)

    # For VL models, save the processor so image inputs work with the toy model.
    if getattr(config, "vision_config", None) is not None:
        _save_processor(output_dir, args.hf_model_id, trust_remote_code=trust_remote_code)

    print(f"Toy HuggingFace checkpoint saved to: {output_dir}")
    print(f"  hidden_layers={args.num_hidden_layers}")
    print(f"  num_experts={args.num_experts}")
    effective_cfg = getattr(config, "text_config", config)
    print(f"  num_experts_per_tok={getattr(effective_cfg, 'num_experts_per_tok', 'N/A')}")
    print(f"  tokenizer_source={tokenizer_id}")


if __name__ == "__main__":
    main()