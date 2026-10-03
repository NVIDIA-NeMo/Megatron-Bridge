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

"""Declarative JSONL source read in place through the memmap reader."""

from dataclasses import dataclass


def _is_object_store_url(path: str) -> bool:
    """Whether ``path`` carries a URL scheme (``msc://``, ``s3://``, ...) rather than naming a local file."""
    return "://" in path


@dataclass(kw_only=True)
class JSONLSourceConfig:
    """JSONL rows read in place through the memmap reader.

    The only source type that reads object storage (``msc://`` paths); anything
    HuggingFace ``datasets`` can read goes through ``HFDatasetSourceConfig`` instead.
    """

    paths: list[str]
    index_mapping_dir: str | None = None
    """Where the memmap ``.idx`` sidecars go. Required for object-store ``paths``,
    whose buckets may be read-only."""

    def validate(self) -> None:
        """Validate the paths against the memmap reader's constraints."""
        if not self.paths:
            raise ValueError("JSONLSourceConfig.paths must contain at least one path.")
        bad = [path for path in self.paths if not path.endswith((".jsonl", ".json"))]
        if bad:
            raise ValueError(
                f"JSONLSourceConfig accepts only .jsonl/.json files (the memmap reader's formats); got {bad}. "
                "Other formats must go through HFDatasetSourceConfig, which cannot read msc://."
            )
        if any(_is_object_store_url(path) for path in self.paths) and self.index_mapping_dir is None:
            raise ValueError(
                "index_mapping_dir is required for object-store paths: the memmap index "
                "sidecars cannot be written beside data in a read-only bucket."
            )
