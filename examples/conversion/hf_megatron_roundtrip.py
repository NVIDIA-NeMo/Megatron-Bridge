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
This example demonstrates how to use the AutoBridge to perform a round-trip
conversion between a Hugging Face model and a Megatron-LM model.

The process is as follows:
1. An AutoBridge is initialized from a pretrained Hugging Face model
    (e.g., "meta-llama/Llama-3.2-1B"). This downloads the model from the Hub and loads it.
2. The bridge's `to_megatron_model` method is called to get a Megatron-LM compatible model provider.
3. The model provider is used to instantiate the Megatron-LM model.
4. Finally, the `save_hf_pretrained` method is used to save the Megatron-LM
    model back into the Hugging Face format. A new directory, named after the
    model, will be created for the converted model files. By default, this
    directory is created in the current working directory, but a different
    parent directory can be specified via the `--output-dir` argument.
"""

import argparse
import os

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion import weights_verification_table


console = Console()
HF_MODEL_ID = "meta-llama/Llama-3.2-1B"


def main(hf_model_id: str = HF_MODEL_ID, output_dir: str = None) -> None:
    """Perform round-trip conversion between HuggingFace and Megatron-LM models."""
    model_name = hf_model_id.split("/")[-1]
    if output_dir:
        save_path = os.path.join(output_dir, model_name)
    else:
        # Default to outputs directory (not tracked by git)
        save_path = os.path.join("outputs", model_name)

    bridge = AutoBridge.from_hf_pretrained(hf_model_id)
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    console.print(weights_verification_table(bridge, megatron_model))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    console.print(f"Saving HF-ckpt in {save_path}...")
    bridge.save_hf_pretrained(megatron_model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert between HuggingFace and Megatron-LM model formats")
    parser.add_argument("--hf-model-id", type=str, default=HF_MODEL_ID, help="HuggingFace model ID to convert")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="The directory where the converted model directory will be created. Defaults to the current working directory.",
    )

    args = parser.parse_args()
    main(args.hf_model_id, args.output_dir)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
