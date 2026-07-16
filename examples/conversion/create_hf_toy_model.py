#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

r"""Create a shallow Hugging Face toy checkpoint that preserves pretrained weights.

The tool retains the first transformer layers and rewrites safetensors files as
a byte stream, so even large or sharded checkpoints do not need to be loaded into
host memory. The retained tensors are byte-for-byte identical to the source.

The source can be a Hugging Face model ID:

.. code-block:: bash

   uv run python examples/conversion/create_hf_toy_model.py \
     Qwen/Qwen3-0.6B \
     /tmp/qwen3-0.6b-4layers \
     --num-hidden-layers 4

Or a local Hugging Face checkpoint directory:

.. code-block:: bash

   uv run python examples/conversion/create_hf_toy_model.py \
     /path/to/Qwen3-0.6B \
     /tmp/qwen3-0.6b-4layers \
     --num-hidden-layers 4 \
     --overwrite

The resulting directory can be passed directly to Bridge conversion tools:

.. code-block:: bash

   uv run python examples/conversion/hf_megatron_roundtrip.py \
     --hf-model-id /tmp/qwen3-0.6b-4layers \
     --output-dir /tmp/qwen3-roundtrip
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import struct
from pathlib import Path
from typing import TypedDict, cast


LOGGER = logging.getLogger(__name__)
_LAYER_KEY_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
_WEIGHT_FILE_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors")


class TensorInfo(TypedDict):
    """Safetensors header entry for one tensor."""

    dtype: str
    shape: list[int]
    data_offsets: list[int]


class RewriteResult(TypedDict):
    """Summary of one rewritten safetensors shard."""

    tensor_names: list[str]
    tensor_bytes: int
    removed_tensor_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Local Hugging Face checkpoint directory or Hub model ID.")
    parser.add_argument("output_dir", type=Path, help="Directory for the truncated checkpoint.")
    parser.add_argument("--num-hidden-layers", type=int, required=True, help="Number of leading layers to retain.")
    parser.add_argument("--revision", help="Optional Hugging Face Hub revision when source is a model ID.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output_dir if it already exists.")
    return parser.parse_args()


def _resolve_source(source: str, *, revision: str | None) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path.resolve()

    from huggingface_hub import snapshot_download

    LOGGER.info("Downloading Hugging Face checkpoint %s", source)
    return Path(snapshot_download(repo_id=source, revision=revision))


def _prepare_output(source_dir: Path, output_dir: Path, *, overwrite: bool) -> Path:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == source_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("output_dir must not overlap the source directory (parent, child, or same path)")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def _should_copy_metadata(path: Path) -> bool:
    if path.name == "config.json" or path.name.endswith(".index.json"):
        return False
    return not path.name.endswith(_WEIGHT_FILE_SUFFIXES)


def _copy_metadata_files(source_dir: Path, output_dir: Path) -> None:
    for source_path in source_dir.iterdir():
        if source_path.is_file() and _should_copy_metadata(source_path):
            shutil.copy2(source_path, output_dir / source_path.name, follow_symlinks=True)


def _truncate_config(source_dir: Path, output_dir: Path, *, num_hidden_layers: int) -> int:
    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hugging Face config not found: {config_path}")

    config = json.loads(config_path.read_text())
    if "num_hidden_layers" not in config:
        raise ValueError("config.json does not contain a top-level num_hidden_layers field")
    original_num_hidden_layers = int(config["num_hidden_layers"])
    if not 0 < num_hidden_layers <= original_num_hidden_layers:
        raise ValueError(
            f"num_hidden_layers must be between 1 and {original_num_hidden_layers}, got {num_hidden_layers}"
        )

    config["num_hidden_layers"] = num_hidden_layers
    if "max_window_layers" in config:
        config["max_window_layers"] = min(int(config["max_window_layers"]), num_hidden_layers)
    if isinstance(config.get("layer_types"), list):
        config["layer_types"] = config["layer_types"][:num_hidden_layers]
    if isinstance(config.get("mlp_only_layers"), list):
        config["mlp_only_layers"] = [layer for layer in config["mlp_only_layers"] if layer < num_hidden_layers]

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return original_num_hidden_layers


def _read_safetensors_header(source_file: Path) -> tuple[int, dict[str, object]]:
    with source_file.open("rb") as source:
        header_size_bytes = source.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header in {source_file}")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        try:
            header = json.loads(source.read(header_size))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid safetensors header JSON in {source_file}") from error
    return header_size, cast(dict[str, object], header)


def _retains_tensor(tensor_name: str, *, num_hidden_layers: int) -> bool:
    layer_match = _LAYER_KEY_PATTERN.search(tensor_name)
    return layer_match is None or int(layer_match.group(1)) < num_hidden_layers


def _rewrite_safetensors(
    source_file: Path,
    output_file: Path,
    *,
    num_hidden_layers: int,
) -> RewriteResult:
    old_header_size, old_header = _read_safetensors_header(source_file)
    metadata = old_header.get("__metadata__")
    tensor_entries: list[tuple[str, TensorInfo]] = []
    removed_tensor_count = 0
    for tensor_name, raw_info in old_header.items():
        if tensor_name == "__metadata__":
            continue
        if not _retains_tensor(tensor_name, num_hidden_layers=num_hidden_layers):
            removed_tensor_count += 1
            continue
        tensor_entries.append((tensor_name, cast(TensorInfo, raw_info)))
    tensor_entries.sort(key=lambda entry: entry[1]["data_offsets"][0])

    new_header: dict[str, object] = {}
    if metadata is not None:
        new_header["__metadata__"] = metadata
    tensor_bytes = 0
    for tensor_name, tensor_info in tensor_entries:
        old_start, old_end = tensor_info["data_offsets"]
        tensor_size = old_end - old_start
        new_header[tensor_name] = {**tensor_info, "data_offsets": [tensor_bytes, tensor_bytes + tensor_size]}
        tensor_bytes += tensor_size

    encoded_header = json.dumps(new_header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    old_data_start = 8 + old_header_size

    with source_file.open("rb") as source, output_file.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded_header)))
        output.write(encoded_header)
        for _, tensor_info in tensor_entries:
            old_start, old_end = tensor_info["data_offsets"]
            source.seek(old_data_start + old_start)
            remaining = old_end - old_start
            while remaining:
                block = source.read(min(16 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError(f"Unexpected end of safetensors data in {source_file}")
                output.write(block)
                remaining -= len(block)

    return {
        "tensor_names": [name for name, _ in tensor_entries],
        "tensor_bytes": tensor_bytes,
        "removed_tensor_count": removed_tensor_count,
    }


def _load_index(source_dir: Path, shard_files: list[Path]) -> dict[str, object] | None:
    index_path = source_dir / _SAFETENSORS_INDEX_NAME
    if not index_path.is_file():
        if len(shard_files) != 1:
            raise ValueError(f"Found {len(shard_files)} safetensors files but no {_SAFETENSORS_INDEX_NAME}")
        return None
    return cast(dict[str, object], json.loads(index_path.read_text()))


def _rewrite_checkpoint(
    source_dir: Path,
    output_dir: Path,
    *,
    num_hidden_layers: int,
) -> tuple[int, int, int]:
    shard_files = sorted(source_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {source_dir}")
    old_index = _load_index(source_dir, shard_files)
    old_weight_map = cast(dict[str, str], old_index["weight_map"]) if old_index is not None else None

    new_weight_map: dict[str, str] = {}
    total_tensor_bytes = 0
    total_tensor_count = 0
    total_removed_tensor_count = 0
    for source_file in shard_files:
        output_file = output_dir / source_file.name
        result = _rewrite_safetensors(source_file, output_file, num_hidden_layers=num_hidden_layers)
        total_removed_tensor_count += result["removed_tensor_count"]
        if not result["tensor_names"]:
            output_file.unlink()
            continue
        total_tensor_bytes += result["tensor_bytes"]
        total_tensor_count += len(result["tensor_names"])
        if old_weight_map is not None:
            for tensor_name in result["tensor_names"]:
                if old_weight_map.get(tensor_name) != source_file.name:
                    raise ValueError(f"Index maps {tensor_name} to an unexpected shard")
                new_weight_map[tensor_name] = source_file.name

    if old_index is not None:
        index_metadata = cast(dict[str, object], old_index.get("metadata", {})).copy()
        index_metadata["total_size"] = total_tensor_bytes
        new_index = {"metadata": index_metadata, "weight_map": new_weight_map}
        (output_dir / _SAFETENSORS_INDEX_NAME).write_text(json.dumps(new_index, indent=2) + "\n")
    return total_tensor_count, total_tensor_bytes, total_removed_tensor_count


def truncate_checkpoint(
    source: str,
    output_dir: Path,
    *,
    num_hidden_layers: int,
    revision: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a shallow checkpoint that retains pretrained weights from leading layers.

    Args:
        source: Local Hugging Face checkpoint directory or Hub model ID.
        output_dir: Directory where the truncated checkpoint is written.
        num_hidden_layers: Number of leading transformer layers to retain.
        revision: Optional Hub revision used when ``source`` is a model ID.
        overwrite: Whether to replace an existing output directory.

    Returns:
        Resolved path to the truncated checkpoint.
    """
    source_dir = _resolve_source(source, revision=revision)
    output_dir = _prepare_output(source_dir, output_dir, overwrite=overwrite)
    _copy_metadata_files(source_dir, output_dir)
    original_num_hidden_layers = _truncate_config(source_dir, output_dir, num_hidden_layers=num_hidden_layers)
    tensor_count, tensor_bytes, removed_tensor_count = _rewrite_checkpoint(
        source_dir, output_dir, num_hidden_layers=num_hidden_layers
    )
    if num_hidden_layers < original_num_hidden_layers and removed_tensor_count == 0:
        raise ValueError("No layer tensors were removed; expected tensor names containing layers.<index>")
    LOGGER.info(
        "Created %s from %s: layers=%d/%d, tensors=%d, tensor_bytes=%d",
        output_dir,
        source_dir,
        num_hidden_layers,
        original_num_hidden_layers,
        tensor_count,
        tensor_bytes,
    )
    return output_dir


def main() -> None:
    """Run checkpoint truncation from command-line arguments."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    truncate_checkpoint(
        args.source,
        args.output_dir,
        num_hidden_layers=args.num_hidden_layers,
        revision=args.revision,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
