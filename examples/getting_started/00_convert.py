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
Getting Started — Step 0: HF ↔ Megatron Checkpoint Conversion

This is the simplest example: load a Hugging Face model, convert it to
Megatron format, and save it back to Hugging Face format.

Usage:
    python examples/getting_started/00_convert.py
    # or: uv run python examples/getting_started/00_convert.py
"""

from megatron.bridge import AutoBridge


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
OUTPUT_DIR = "./exports/llama32_1b"


def main():
    """Convert a HuggingFace model to Megatron format and back."""
    # Step 1: Create a bridge from any supported HF model
    print(f"Loading {HF_MODEL_ID}...")
    bridge = AutoBridge.from_hf_pretrained(HF_MODEL_ID)

    # Step 2: Convert to Megatron (single-GPU, no parallelism)
    print("Converting HF → Megatron...")
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)

    # Step 3: Export back to HF format (config + tokenizer + weights)
    print(f"Saving HF checkpoint to {OUTPUT_DIR}...")
    bridge.save_hf_pretrained(megatron_model, OUTPUT_DIR)

    print("Done! You can load the exported model with:")
    print("  from transformers import AutoModelForCausalLM")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{OUTPUT_DIR}")')


if __name__ == "__main__":
    main()
