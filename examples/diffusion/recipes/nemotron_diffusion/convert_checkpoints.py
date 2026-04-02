#!/usr/bin/env python3
"""
NemotronDiffusion HF <-> Megatron-Bridge checkpoint conversion.

Usage:
  # HF -> Megatron-Bridge
  python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py import \
    --hf-model /path/to/hf_model \
    --megatron-path /path/to/mb_checkpoint

  # Megatron-Bridge -> HF
  python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py export \
    --hf-model /path/to/hf_model \
    --megatron-path /path/to/mb_checkpoint \
    --hf-path /path/to/output_hf
"""

import argparse
import sys

import torch

from megatron.bridge import AutoBridge

# Register NemotronDiffusionBridge before using AutoBridge
from megatron.bridge.diffusion.conversion.nemotron_diffusion import nemotron_diffusion_bridge  # noqa: F401


def main():
    """Entry point for HF<->Megatron checkpoint conversion."""
    parser = argparse.ArgumentParser(description="NemotronDiffusion checkpoint conversion")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import", help="HF -> Megatron-Bridge")
    import_parser.add_argument("--hf-model", required=True)
    import_parser.add_argument("--megatron-path", required=True)
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    import_parser.add_argument("--device", default=None)

    export_parser = subparsers.add_parser("export", help="Megatron-Bridge -> HF")
    export_parser.add_argument("--hf-model", required=True)
    export_parser.add_argument("--megatron-path", required=True)
    export_parser.add_argument("--hf-path", required=True)
    export_parser.add_argument("--no-progress", action="store_true")
    export_parser.add_argument("--not-strict", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    if args.command == "import":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = dtype_map[args.torch_dtype]
        print(f"Importing {args.hf_model} -> {args.megatron_path}")
        AutoBridge.import_ckpt(
            hf_model_id=args.hf_model,
            megatron_path=args.megatron_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        print(f"Done. Checkpoint saved to {args.megatron_path}")

    elif args.command == "export":
        print(f"Exporting {args.megatron_path} -> {args.hf_path}")
        bridge = AutoBridge.from_hf_pretrained(args.hf_model)
        bridge.export_ckpt(
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
        )
        print(f"Done. HF model saved to {args.hf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
