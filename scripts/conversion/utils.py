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
"""Utilities shared by CPU and distributed GPU conversion backends."""

import shutil
from pathlib import Path

import torch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_dtype(name: str) -> torch.dtype:
    """Resolve a CLI dtype name.

    Args:
        name: User-facing dtype name.

    Returns:
        Matching PyTorch dtype.

    Raises:
        ValueError: If the dtype name is unsupported.
    """
    try:
        return DTYPE_MAP[name]
    except KeyError:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(DTYPE_MAP)}.") from None


def prepare_output_directory(path: str, *, overwrite: bool) -> Path:
    """Validate and optionally clear a conversion destination.

    Args:
        path: Destination directory.
        overwrite: Delete a non-empty destination when true.

    Returns:
        Destination as a ``Path``.

    Raises:
        FileExistsError: If the destination is non-empty and overwrite is false.
        ValueError: If overwrite targets the filesystem root.
    """
    output_path = Path(path).expanduser()
    if not output_path.exists() or not any(output_path.iterdir()):
        return output_path
    if not overwrite:
        raise FileExistsError(f"Destination is not empty: {output_path}. Pass --overwrite to replace it.")
    if output_path.resolve() == Path("/"):
        raise ValueError("Refusing to overwrite the filesystem root.")
    shutil.rmtree(output_path)
    return output_path
