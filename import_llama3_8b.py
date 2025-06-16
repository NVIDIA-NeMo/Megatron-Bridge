#!/usr/bin/env python3
"""
Script to convert HuggingFace Llama3-8B model to Megatron-LM format.
"""

import argparse
from pathlib import Path

from megatron.hub.converters.llama import HFLlamaImporter


def convert_llama3_hf_to_megatron(input_path: str, output_path: str) -> None:
    """Convert HuggingFace Llama3 model to Megatron format.

    Args:
        input_path: Path or name of HuggingFace model (e.g., "meta-llama/Meta-Llama-3-8B")
        output_path: Directory where to save the converted Megatron checkpoint
    """
    print(f"Converting {input_path} to Megatron format...")
    print(f"Output will be saved to: {output_path}")

    # Create the importer
    importer = HFLlamaImporter(input_path=input_path, output_path=output_path)

    # Run the conversion
    result_path = importer.apply()

    print("✅ Conversion completed successfully!")
    print(f"📁 Megatron checkpoint saved to: {result_path}")
    print("🔧 You can now use this for PEFT training with:")
    print(f"   config.checkpoint.pretrained_checkpoint = '{result_path}'")


def main():
    """Main function to convert HuggingFace Llama3 model to Megatron format."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Llama3 model to Megatron-LM format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, default="meta-llama/Meta-Llama-3-8B", help="HuggingFace model name or local path"
    )

    parser.add_argument("--output", type=str, required=True, help="Output directory for Megatron checkpoint")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output).mkdir(parents=True, exist_ok=True)

    try:
        convert_llama3_hf_to_megatron(args.input, args.output)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
