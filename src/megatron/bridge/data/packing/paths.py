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

"""Path detection and resolution for packed Parquet artifacts."""

import glob
import os
from pathlib import Path

from megatron.core.msc_utils import MultiStorageClientFeature


def is_packed_parquet_file(path) -> bool:
    """Check if a path refers to a packed Parquet file or pattern.

    Args:
        path: A Path object or string path.

    Returns:
        True if the path ends with .idx.parquet or .idx.pq, or contains a glob
        pattern that would match such files.
    """
    name = str(path).lower()
    # Matches both direct files and glob patterns (e.g., "data*.idx.parquet")
    # since both end with the extension.
    return name.endswith(".idx.parquet") or name.endswith(".idx.pq")


def is_packed_parquet_spec(spec: str | Path) -> bool:
    """Check if a spec refers to a packed Parquet source (file, directory, or glob).

    This predicate reflects what the dataset loader supports in packed mode:
    - Single .parquet/.idx.parquet/.idx.pq files
    - Glob patterns ending in .parquet/.idx.parquet/.idx.pq
    - Directories containing parquet files

    Args:
        spec: A path specification (file, directory, or glob pattern).

    Returns:
        True if the spec could refer to packed Parquet data.
    """
    spec_str = str(spec).lower()

    # Check for parquet file extensions (including glob patterns)
    if spec_str.endswith(".parquet") or spec_str.endswith(".pq"):
        return True

    # Check for glob patterns containing parquet extension
    if "*" in spec_str or "?" in spec_str:
        # Extract the pattern part after the last glob character
        return ".parquet" in spec_str or ".pq" in spec_str

    # For directories, try to resolve to parquet files
    # This is more robust than is_dir() on distributed filesystems (Lustre, S3, etc.)
    try:
        resolved = _resolve_parquet_paths(str(spec))
        return len(resolved) > 0
    except ValueError:
        pass

    # Fallback: check if it's a directory using filesystem abstraction
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc_path = msc.Path(str(spec))
        return msc_path.is_dir() if hasattr(msc_path, "is_dir") else False
    else:
        return Path(spec).is_dir()


def _is_parquet_file(path: str) -> bool:
    """Check if a path refers to any Parquet file.

    Args:
        path: A string path.

    Returns:
        True if the path ends with .parquet or .pq (case-insensitive).
    """
    name = path.lower()
    return name.endswith(".parquet") or name.endswith(".pq")


def _resolve_parquet_paths(file_path: str) -> list[str]:
    """Resolve a file path specification to a list of actual file paths.

    Supports:
    - Single file: "data.idx.parquet", "shard_0.parquet"
    - Glob pattern: "data*.idx.parquet", "shard_*.parquet"
    - Directory: "/path/to/data/" (globs for *.parquet and *.pq)

    Args:
        file_path: Path specification (file, glob pattern, or directory).

    Returns:
        Sorted list of resolved file paths.

    Raises:
        ValueError: If no matching files are found.
    """
    path_str = str(file_path)

    # Check if it's a glob pattern
    if "*" in path_str or "?" in path_str:
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            # MSC glob support - normalize to strings immediately
            if hasattr(msc, "glob"):
                paths = [str(p) for p in msc.glob(path_str)]
            else:
                # Fallback: try to use msc.Path with glob
                # Use msc.Path to split parent/pattern to handle URIs correctly
                msc_full_path = msc.Path(path_str)
                parent = str(msc_full_path.parent) if hasattr(msc_full_path, "parent") else None
                pattern = msc_full_path.name if hasattr(msc_full_path, "name") else None

                if parent is not None and pattern is not None:
                    msc_parent_path = msc.Path(parent)
                    if hasattr(msc_parent_path, "glob"):
                        paths = [str(p) for p in msc_parent_path.glob(pattern)]
                    else:
                        raise ValueError(f"MSC backend does not support glob operations for pattern: {path_str}")
                else:
                    raise ValueError(f"MSC backend does not support glob operations for pattern: {path_str}")
        else:
            paths = glob.glob(path_str)

        # Filter to only parquet files (accepts both *.parquet and *.idx.parquet)
        paths = [p for p in paths if _is_parquet_file(p)]
        paths = sorted(paths)

        if not paths:
            raise ValueError(
                f"No Parquet files found matching pattern: {path_str}. Files must end with .parquet or .pq"
            )
        return paths

    # Check if it's a directory
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc_path = msc.Path(path_str)
        is_dir = msc_path.is_dir() if hasattr(msc_path, "is_dir") else False
    else:
        is_dir = Path(path_str).is_dir()

    if is_dir:
        # Glob for parquet files in directory (accepts both *.parquet and *.idx.parquet)
        paths = []
        for ext in ["*.parquet", "*.pq"]:
            pattern = os.path.join(path_str, ext)
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                if hasattr(msc, "glob"):
                    # Normalize to strings immediately
                    paths.extend([str(p) for p in msc.glob(pattern)])
                elif hasattr(msc.Path(path_str), "glob"):
                    paths.extend([str(p) for p in msc.Path(path_str).glob(ext)])
            else:
                paths.extend(glob.glob(pattern))

        paths = sorted(set(paths))

        if not paths:
            raise ValueError(f"No Parquet files found in directory: {path_str}. Files must end with .parquet or .pq")
        return paths

    # Single file - verify it has a parquet extension and exists
    if not _is_parquet_file(path_str):
        return []

    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        exists = msc.Path(path_str).exists()
    else:
        exists = Path(path_str).exists()

    if not exists:
        raise ValueError(f"Packed Parquet file not found: {path_str}")

    return [path_str]


def resolve_packed_parquet_paths(spec: str | Path) -> list[str]:
    """Resolve a packed parquet spec to a list of shard file paths.

    Public wrapper around the internal _resolve_parquet_paths function.
    Use this to validate and resolve packed parquet specs before dataset creation.

    Supports:
    - Single file: "data.idx.parquet", "shard_0.parquet"
    - Glob pattern: "data*.idx.parquet", "shard_*.parquet"
    - Directory: "/path/to/data/" (globs for *.parquet and *.pq)

    Args:
        spec: Path specification (file, glob pattern, or directory).

    Returns:
        Sorted list of resolved file paths.

    Raises:
        ValueError: If no matching files are found.
    """
    return _resolve_parquet_paths(str(spec))
