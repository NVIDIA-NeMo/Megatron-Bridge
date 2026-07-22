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
from collections.abc import Iterable
from pathlib import Path

import torch


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _validate_hf_revision_target(hf_model: str, hf_revision: str | None) -> None:
    """Reject revision pinning for local model paths."""
    if hf_revision is not None and Path(hf_model).expanduser().exists():
        raise ValueError("--hf-revision applies only to Hugging Face Hub model IDs, not local paths.")


def resolve_hf_commit_revision(hf_model: str, hf_revision: str | None) -> str | None:
    """Resolve a Hub branch, tag, or commit to one immutable commit SHA.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hub branch, tag, or commit to resolve.

    Returns:
        The immutable Hub commit SHA, or ``None`` when no revision was supplied.

    Raises:
        ValueError: If a revision is paired with an existing local path.
        RuntimeError: If the Hub response does not contain a commit SHA.
    """
    _validate_hf_revision_target(hf_model, hf_revision)
    if hf_revision is None:
        return None

    from huggingface_hub import HfApi

    resolved_revision = HfApi().model_info(repo_id=hf_model, revision=hf_revision).sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face Hub did not return a commit SHA for {hf_model}@{hf_revision}.")
    return resolved_revision


def resolve_hf_model_revision(hf_model: str, hf_revision: str | None) -> str:
    """Resolve a remote Hugging Face model revision to an immutable local snapshot.

    Args:
        hf_model: Hugging Face model ID or local path.
        hf_revision: Hub branch, tag, or commit to resolve. ``None`` preserves
            the original model reference.

    Returns:
        The original model reference when no revision is provided, otherwise
        the local path of the resolved Hub snapshot.

    Raises:
        ValueError: If a revision is paired with an existing local path.
    """
    _validate_hf_revision_target(hf_model, hf_revision)
    if hf_revision is None:
        return hf_model

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_model, revision=hf_revision)


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


def validate_output_path(path: str, *, source_paths: Iterable[str]) -> Path:
    """Reject a conversion destination that overlaps an existing local source.

    Args:
        path: Destination directory.
        source_paths: Local or remote source references. Nonexistent paths are
            treated as remote identifiers and skipped.

    Returns:
        Destination as a ``Path``.

    Raises:
        ValueError: If the destination equals, contains, or is contained by a
            local source path.
    """
    output_path = Path(path).expanduser()
    resolved_output = output_path.resolve()
    for source in source_paths:
        source_path = Path(source).expanduser()
        if not source_path.exists():
            continue
        resolved_source = source_path.resolve()
        if (
            resolved_output == resolved_source
            or resolved_output in resolved_source.parents
            or resolved_source in resolved_output.parents
        ):
            raise ValueError(f"Destination {output_path} overlaps conversion source {source_path}.")
    return output_path


def prepare_output_directory(path: str, *, overwrite: bool, source_paths: Iterable[str] = ()) -> Path:
    """Validate and optionally clear a conversion destination.

    Args:
        path: Destination directory.
        overwrite: Delete a non-empty destination when true.
        source_paths: Local or remote source references that must not overlap
            the destination.

    Returns:
        Destination as a ``Path``.

    Raises:
        FileExistsError: If the destination is non-empty and overwrite is false.
        ValueError: If the destination overlaps a local source or overwrite
            targets the filesystem root.
    """
    output_path = validate_output_path(path, source_paths=source_paths)
    if not output_path.exists() or not any(output_path.iterdir()):
        return output_path
    if not overwrite:
        raise FileExistsError(f"Destination is not empty: {output_path}. Pass --overwrite to replace it.")
    if output_path.resolve() == Path("/"):
        raise ValueError("Refusing to overwrite the filesystem root.")
    shutil.rmtree(output_path)
    return output_path
