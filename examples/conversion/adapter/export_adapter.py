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

No GPU required -- runs entirely on CPU.

The output can be loaded directly with::

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("<hf-model-id>")
    model = PeftModel.from_pretrained(base, "./my_adapter")

Usage::

    uv run python examples/conversion/adapter/export_adapter.py \\
        --hf-model-id meta-llama/Llama-3.2-1B \\
        --megatron-peft-checkpoint /path/to/finetune_ckpt \\
        --output-hf-path ./my_adapter
"""

from __future__ import annotations

import argparse
from pathlib import Path

from megatron.bridge import AutoBridge


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Megatron-Bridge LoRA adapter to HuggingFace PEFT format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hf-model-id",
        required=True,
        help="HuggingFace model name or local path (architecture + base weights).",
    )
    parser.add_argument(
        "--megatron-peft-checkpoint",
        required=True,
        help="Megatron-Bridge distributed checkpoint containing LoRA adapter weights.",
    )
    parser.add_argument("--output-hf-path", type=Path, default=Path("./my_adapter"))
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Export a Megatron-Bridge PEFT checkpoint to HuggingFace PEFT format."""
    args = parse_args()

    bridge = AutoBridge.from_hf_pretrained(args.hf_model_id, trust_remote_code=args.trust_remote_code)
    bridge.export_adapter_ckpt(
        peft_checkpoint=args.megatron_peft_checkpoint,
        output_path=args.output_hf_path,
    )


if __name__ == "__main__":
    main()
