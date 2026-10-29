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
Data path utilities for Megatron-Bridge CI testing.

Resolves cluster-specific paths from a single per-cluster JSON file
(``--base_paths``) for the dataset/tokenizer type selected by ``--type``.

Supported types:

- ``rp2``        — expands per-cluster RedPajama-v2 base into the head/middle
                   shard list (37 paths by default, 13 with ``--head_only``).
- ``tokenizer``  — appends ``tokenizer/tokenizer.model`` to the per-cluster
                   base.
- ``direct``     — emits the per-cluster value unchanged (e.g. webdataset
                   directories used by diffusion recipes).
"""

import argparse
import json
import os


def bool_arg(arg):
    """Convert a string CLI value to a boolean."""
    if arg.lower() in ["true", "1", "t", "yes", "y"]:
        return True
    elif arg.lower() in ["false", "0", "f", "no", "n"]:
        return False
    else:
        raise ValueError(f"Invalid value for boolean argument: {arg}")


def _require_cluster(cluster: str, base_paths: dict[str, str]) -> str:
    """Return ``base_paths[cluster]`` or raise with the list of known clusters."""
    if cluster not in base_paths:
        raise ValueError(f"Unsupported cluster: {cluster}. Supported clusters: {list(base_paths.keys())}")
    return base_paths[cluster]


def get_tokenizer_path(cluster: str, base_paths: dict[str, str]) -> str:
    """Return the per-cluster path to ``tokenizer/tokenizer.model``."""
    return os.path.join(_require_cluster(cluster, base_paths), "tokenizer/tokenizer.model")


def get_direct_path(cluster: str, base_paths: dict[str, str]) -> str:
    """Return the per-cluster path verbatim — no template expansion."""
    return _require_cluster(cluster, base_paths)


def get_rp2_paths(cluster: str, base_paths: dict[str, str], head_only: bool = False) -> list[str]:
    """Expand the per-cluster RedPajama-v2 base into the head/middle shard list."""
    base = _require_cluster(cluster, base_paths)
    paths = [
        os.path.join(
            base,
            f"kenlm_perp_head_gopher_linefilter_decompressed/bin_idx/nemo/head_{i:02d}_text_document",
        )
        for i in range(1, 14)
    ]
    if not head_only:
        paths.extend(
            os.path.join(
                base,
                f"kenlm_perp_middle_gopher_linefilter_decompressed/bin_idx/nemo/middle_{i:02d}_text_document",
            )
            for i in range(1, 26)
        )
    return paths


def _load_base_paths(path: str) -> dict[str, str]:
    """Load a per-cluster base-paths JSON file."""
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["rp2", "tokenizer", "direct"], required=True)
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--base_paths", type=str, required=True, help="Path to per-cluster base-paths JSON.")
    parser.add_argument("--head_only", type=bool_arg, required=False, default=False)
    args = parser.parse_args()

    base_paths = _load_base_paths(args.base_paths)

    if args.type == "rp2":
        print(" ".join(get_rp2_paths(args.cluster, base_paths=base_paths, head_only=args.head_only)))
    elif args.type == "tokenizer":
        print(get_tokenizer_path(args.cluster, base_paths=base_paths))
    elif args.type == "direct":
        print(get_direct_path(args.cluster, base_paths=base_paths))
