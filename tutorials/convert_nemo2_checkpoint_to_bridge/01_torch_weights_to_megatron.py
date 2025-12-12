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
Step 2 of the NeMo2 → Megatron Bridge conversion pipeline.

The weights saved by `00_save_nemo2_weights_torch.py` are loaded into a CPU
Megatron Bridge checkpoint. We bootstrap a Hugging Face config/tokenizer,
instantiate the bridge, and write a single-rank checkpoint that Megatron can
further process or shard.

Minimal example
---------------
    python tutorials/convert_nemo2_checkpoint_to_bridge/01_torch_weights_to_megatron.py \
        --hf-model-id meta-llama/Llama-3.2-1B \
        --torch-weights /path/to/model_weights.pt \
        --output-dir /tmp/megatron_ckpt

All arguments map directly to CLI flags. The script prints missing/unexpected
keys and saves the finished Megatron checkpoint inside `--output-dir`.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable

import torch
from rich.console import Console
from rich.table import Table

from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import temporary_distributed_context


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
console = Console()


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "f32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "f16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]


def _infer_dtype_from_state(state_dict: Dict[str, torch.Tensor]) -> torch.dtype:
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            return tensor.dtype
    return torch.float32


def _strip_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    stripped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        stripped[new_key] = value
    return stripped


def _load_torch_weights(path: Path) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(path, map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    if not state_dict:
        raise ValueError(f"No tensors found in checkpoint {path}")
    # Standardize common NeMo prefixes: module.* or model.*
    state_dict = _strip_prefixes(state_dict, prefixes=("module.", "model."))
    return OrderedDict(sorted(state_dict.items()))


def _default_save_dir(output_dir: Path | None, hf_model_id: str) -> Path:
    if output_dir is not None:
        return output_dir
    return Path.cwd() / f"{hf_model_id.split('/')[-1]}_mbridge"


def _print_summary_table(state_dict: Dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        sample_keys = list(state_dict.keys())[:5]
        table = Table(title="Loaded NeMo Weights (sample)")
        table.add_column("Parameter")
        table.add_column("Shape")
        table.add_column("dtype")
        for key in sample_keys:
            tensor = state_dict[key]
            table.add_row(key, str(tuple(tensor.shape)), str(tensor.dtype).replace("torch.", ""))
        table.add_row("...", "...", "...")
        console.print(table)
        console.print(f"[green]Detected dtype:[/green] {dtype}")


def _summarize_keys(keys: Iterable[str], limit: int = 8) -> str:
    key_list = list(keys)
    if not key_list:
        return "None"
    if len(key_list) <= limit:
        return ", ".join(key_list)
    return ", ".join(key_list[:limit]) + ", ..."


def parse_args() -> argparse.Namespace:
    """Build the CLI argument parser for the conversion helper script."""
    parser = argparse.ArgumentParser(
        description="Load torch weights (e.g., from a NeMo2 checkpoint) into a Megatron checkpoint using AutoBridge."
    )
    parser.add_argument("--hf-model-id", type=str, required=True, help="HuggingFace model ID providing the config.")
    parser.add_argument(
        "--torch-weights",
        type=Path,
        required=True,
        help="Path to the .pt file produced by 00_save_nemo2_weights_torch.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for the Megatron checkpoint (defaults to <cwd>/<model>_mbridge).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        help="Torch dtype to instantiate the HuggingFace model (auto-detected if not provided).",
    )
    parser.add_argument("--not-strict", action="store_true", help="Disable strict loading of torch weights.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom HF modeling code.")
    return parser.parse_args()


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def main(
    hf_model_id: str,
    torch_weights: Path,
    output_dir: Path | None = None,
    torch_dtype: str | None = None,
    not_strict: bool = False,
    trust_remote_code: bool = False,
) -> None:
    """Load torch weights and save them as a Megatron Bridge checkpoint."""
    weights_path = Path(torch_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Torch weights file not found: {weights_path}")

    save_path = _default_save_dir(output_dir, hf_model_id)
    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(f"Output directory {save_path} already exists and is not empty.")
    save_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading torch weights from[/bold] {weights_path}")
    nemo_state = _load_torch_weights(weights_path)
    detected_dtype = _infer_dtype_from_state(nemo_state)
    target_dtype = _dtype_from_name(torch_dtype) if torch_dtype else detected_dtype
    _print_summary_table(nemo_state, target_dtype)

    console.print(f"[bold]Building HuggingFace stub for[/bold] {hf_model_id}")

    bridge = AutoBridge.from_hf_pretrained(hf_model_id)

    console.print("[bold]Configuring Megatron provider (CPU single-rank)...[/bold]")
    with temporary_distributed_context(backend="gloo"):
        megatron_model = bridge.to_megatron_model(
            load_weights=False,
            wrap_with_ddp=False,
            use_cpu_initialization=True,
        )
        load_return = megatron_model[0].load_state_dict(nemo_state, strict=not not_strict)
        if _is_rank_zero():
            if load_return.missing_keys:
                console.print(
                    f"[yellow]Missing keys ({len(load_return.missing_keys)}): "
                    f"{_summarize_keys(load_return.missing_keys)}[/yellow]"
                )
            else:
                console.print("[green]Missing keys: none[/green]")

            if load_return.unexpected_keys:
                console.print(
                    f"[yellow]Unexpected keys ({len(load_return.unexpected_keys)}): "
                    f"{_summarize_keys(load_return.unexpected_keys)}[/yellow]"
                )
            else:
                console.print("[green]Unexpected keys: none[/green]")

        if _is_rank_zero():
            console.print(f"[green]Saving Megatron checkpoint to[/green] {save_path}")
        bridge.save_megatron_model(megatron_model, save_path)

    console.print("[bold green]✅ Conversion complete.[/bold green]")


if __name__ == "__main__":
    args = parse_args()
    main(
        hf_model_id=args.hf_model_id,
        torch_weights=args.torch_weights,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        not_strict=args.not_strict,
        trust_remote_code=args.trust_remote_code,
    )
