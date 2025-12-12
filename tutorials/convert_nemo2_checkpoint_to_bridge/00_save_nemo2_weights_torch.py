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
Step 1 of the NeMo2 → Megatron Bridge conversion pipeline.

This script must run inside the **NeMo training environment** so it can import the
same modules that `nemo llm export` uses. It loads a NeMo2 checkpoint directory,
captures the model's `state_dict`, optionally moves all tensors to CPU, and
stores the result with `torch.save`.

Example
-------
    python tutorials/convert_nemo2_checkpoint_to_bridge/00_save_nemo2_weights_torch.py \
        --nemo-checkpoint /path/to/meta-llama/Meta-Llama-3-8B \
        --output /tmp/llama3_8b_weights.pt

The saved tensor file is the input to `01_torch_weights_to_megatron.py`.
"""

import argparse
from pathlib import Path

import torch  # type: ignore[import]
from nemo.lightning.io.connector import ModelConnector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a NeMo 2 checkpoint and save its model state dict with torch.save."
    )
    parser.add_argument(
        "--nemo-checkpoint",
        type=Path,
        required=True,
        help="Path to the NeMo2 checkpoint directory (the same path you would pass to `nemo llm export`).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the serialized state dict (e.g., /tmp/model_weights.pt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--keep-device",
        action="store_true",
        help="If set, keep tensors on their original devices instead of moving to CPU before saving.",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Load model + weights using the same path the exporters consume.
    model, _ = ModelConnector().nemo_load(checkpoint_path, cpu=True)
    model.eval()
    return model


def _maybe_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Move tensors to CPU to make the saved artifact device agnostic and portable.
    return {name: tensor.cpu() if torch.is_tensor(tensor) else tensor for name, tensor in state_dict.items()}


def main():
    """CLI entry point for dumping NeMo2 checkpoint weights with torch.save."""
    args = _parse_args()

    output_path: Path = args.output
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file {output_path} exists. Re-run with --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.nemo_checkpoint)

    state_dict = model.state_dict()
    if not args.keep_device:
        state_dict = _maybe_to_cpu(state_dict)

    torch.save(state_dict, output_path)
    print(f"Saved state dict to {output_path}")


if __name__ == "__main__":
    # torch.set_grad_enabled(False) avoids accidental grad tracking when running in notebooks.
    torch.set_grad_enabled(False)
    main()
