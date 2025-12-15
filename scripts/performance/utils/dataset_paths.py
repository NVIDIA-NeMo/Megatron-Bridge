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
Provides cluster-specific paths for datasets and tokenizers.
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


def get_tokenizer_path(cluster: str, base_paths_tokenizer: dict[str, str]) -> str:
    """Get the default tokenizer path for the specified cluster."""
    if cluster not in base_paths_tokenizer:
        raise ValueError(f"Unsupported cluster: {cluster}. Supported clusters: {list(base_paths_tokenizer.keys())}")

    return os.path.join(base_paths_tokenizer[cluster], "tokenizer/tokenizer.model")


def get_dataset_paths(cluster: str, base_paths_rp2: dict[str, str], head_only: bool = False) -> list[str]:
    """Get the default dataset paths for the specified cluster."""
    if cluster not in base_paths_rp2:
        raise ValueError(f"Unsupported cluster: {cluster}. Supported clusters: {list(base_paths_rp2.keys())}")

    paths = []
    for i in range(1, 14):
        paths.append(
            os.path.join(
                base_paths_rp2[cluster],
                f"kenlm_perp_head_gopher_linefilter_decompressed/bin_idx/nemo/head_{i:02d}_text_document",
            )
        )

    if not head_only:
        for i in range(1, 26):
            paths.append(
                os.path.join(
                    base_paths_rp2[cluster],
                    f"kenlm_perp_middle_gopher_linefilter_decompressed/bin_idx/nemo/middle_{i:02d}_text_document",
                )
            )

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["dataset", "tokenizer"])
    parser.add_argument("--cluster", type=str, required=True)
    parser.add_argument("--base_paths_rp2", type=str, required=False)
    parser.add_argument("--base_paths_tokenizer", type=str, required=False)
    parser.add_argument("--head_only", type=bool_arg, required=False, default=False)
    args = parser.parse_args()

    if args.type == "dataset":
        with open(args.base_paths_rp2, "r") as f:
            base_paths_rp2 = json.load(f)
        print(" ".join(get_dataset_paths(args.cluster, base_paths_rp2=base_paths_rp2, head_only=args.head_only)))

    if args.type == "tokenizer":
        with open(args.base_paths_tokenizer, "r") as f:
            base_paths_tokenizer = json.load(f)
        print(get_tokenizer_path(args.cluster, base_paths_tokenizer=base_paths_tokenizer))
