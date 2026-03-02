#!/usr/bin/env python3
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
import json
import math
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
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
        "--quantize-fp8",
        action="store_true",
        default=False,
        help="Post-process the saved checkpoint into FP8 (e4m3) block-wise "
        "format with scale_inv tensors, matching the DeepSeek-V3 / Kimi-K2.5 "
        "quantization convention.",
    )
    parser.add_argument(
        "--fp8-block-size",
        type=int,
        default=128,
        help="Block size for FP8 block-wise quantization (default: 128).",
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

    # Always strip quantization_config during model creation so
    # from_config instantiates plain bf16 weights.  If --quantize-fp8 is
    # requested the checkpoint is post-processed later.
    for cfg in (config, text_config):
        if cfg is not None and hasattr(cfg, "quantization_config"):
            del cfg.quantization_config


# FP8 e4m3 representable range
_FP8_E4M3_MAX = 448.0


def _rebuild_safetensors_index(output_dir: Path, st_files: list[Path]) -> None:
    """Regenerate model.safetensors.index.json from the current safetensors files."""
    index_path = output_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return

    weight_map: dict[str, str] = {}
    metadata: dict[str, str] = {}
    for st_path in st_files:
        tensors = load_file(str(st_path), device="cpu")
        for key in tensors:
            weight_map[key] = st_path.name
        total_bytes = sum(t.nelement() * t.element_size() for t in tensors.values())
        metadata[st_path.name] = str(total_bytes)

    index = {"metadata": {"total_size": sum(int(v) for v in metadata.values())}, "weight_map": weight_map}
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"  rebuilt {index_path.name} with {len(weight_map)} keys")


def _quantize_checkpoint_fp8(output_dir: Path, block_size: int = 128) -> None:
    """Convert saved bf16 safetensors in *output_dir* to FP8 block-wise format.

    For every 2-D weight tensor whose both dimensions are >= *block_size*,
    produce:
      - ``{name}`` in ``torch.float8_e4m3fn``
      - ``{name}_scale_inv`` with per-block dequantization scales (float32)

    Then inject a ``quantization_config`` into ``config.json``.
    """
    st_files = sorted(output_dir.glob("*.safetensors"))
    if not st_files:
        print("  WARNING: no safetensors found; skipping FP8 quantization")
        return

    for st_path in st_files:
        tensors = load_file(str(st_path))
        new_tensors: dict[str, torch.Tensor] = {}
        quantized_count = 0

        for name, tensor in tensors.items():
            if tensor.ndim == 2 and tensor.shape[0] >= block_size and tensor.shape[1] >= block_size:
                fp8_weight, scale_inv = _quantize_tensor_fp8(tensor.float(), block_size)
                new_tensors[name] = fp8_weight
                new_tensors[name + "_scale_inv"] = scale_inv
                quantized_count += 1
            else:
                new_tensors[name] = tensor

        save_file(new_tensors, str(st_path))
        print(f"  {st_path.name}: quantized {quantized_count} tensors to FP8")

    # Rebuild the safetensors index so that _scale_inv keys are discoverable
    # by lazy-loading state dict implementations (e.g. Megatron-Bridge).
    _rebuild_safetensors_index(output_dir, st_files)

    config_path = output_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        quant_cfg = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [block_size, block_size],
            "activation_scheme": "dynamic",
        }
        cfg["quantization_config"] = quant_cfg
        if "text_config" in cfg:
            cfg["text_config"]["quantization_config"] = quant_cfg
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print("  injected quantization_config into config.json")


def _quantize_tensor_fp8(
    tensor: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a single 2-D tensor to FP8 e4m3 with per-block scales.

    Returns ``(fp8_weight, scale_inv)``."""
    M, N = tensor.shape
    num_blocks_m = math.ceil(M / block_size)
    num_blocks_n = math.ceil(N / block_size)
    padded_M = num_blocks_m * block_size
    padded_N = num_blocks_n * block_size

    if M != padded_M or N != padded_N:
        padded = torch.zeros(padded_M, padded_N, dtype=tensor.dtype, device=tensor.device)
        padded[:M, :N] = tensor
    else:
        padded = tensor

    blocks = padded.reshape(num_blocks_m, block_size, num_blocks_n, block_size)
    abs_max = blocks.abs().amax(dim=(1, 3))  # [num_blocks_m, num_blocks_n]
    scale_inv = (abs_max / _FP8_E4M3_MAX).clamp(min=1e-12).to(torch.float32)

    scaled = blocks / scale_inv[:, None, :, None]
    scaled = scaled.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    scaled = scaled.reshape(padded_M, padded_N)

    if M != padded_M or N != padded_N:
        scaled = scaled[:M, :N].contiguous()

    fp8_weight = scaled.to(torch.float8_e4m3fn)
    return fp8_weight, scale_inv


def _save_tokenizer(output_dir: Path, tokenizer_id: str, *, trust_remote_code: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)


def _save_processor(output_dir: Path, model_id: str, *, trust_remote_code: bool) -> None:
    """Save the AutoProcessor alongside the model so VL toy models can process images."""
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        processor.save_pretrained(output_dir)
        print(f"  Processor ({type(processor).__name__}) saved to {output_dir}")
    except Exception as exc:
        print(f"  Processor not available for {model_id} ({exc}); skipping.")


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

    if args.quantize_fp8:
        print("Quantizing checkpoint to FP8 (e4m3) block-wise format...")
        _quantize_checkpoint_fp8(output_dir, block_size=args.fp8_block_size)

    _save_tokenizer(output_dir, tokenizer_id, trust_remote_code=trust_remote_code)

    # For VL models, save the processor so image inputs work with the toy model.
    if getattr(config, "vision_config", None) is not None:
        _save_processor(output_dir, args.hf_model_id, trust_remote_code=trust_remote_code)

    print(f"Toy HuggingFace checkpoint saved to: {output_dir}")
    print(f"  hidden_layers={args.num_hidden_layers}")
    print(f"  num_experts={args.num_experts}")
    effective_cfg = getattr(config, "text_config", config)
    print(f"  num_experts_per_tok={getattr(effective_cfg, 'num_experts_per_tok', 'N/A')}")
    print(f"  quantize_fp8={args.quantize_fp8}")
    print(f"  tokenizer_source={tokenizer_id}")


if __name__ == "__main__":
    main()
