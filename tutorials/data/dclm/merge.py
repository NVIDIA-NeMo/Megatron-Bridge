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

import argparse
import subprocess
import time

import numpy as np


def arguments():
    """Argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_save",
        type=str,
        required=True,
        help="Path where to save merged file.",
    )
    parser.add_argument("--source_dir", type=str, required=True, help="Path to decompressed dataset.")
    parser.add_argument(
        "--remove_small_files",
        action="store_true",
        help="Removes small files after merging.",
    )

    return parser


def merge_data(
    path_to_save: str,
    source_dir: str,
    remove_small_files: bool = True,
) -> None:
    """Merges hundreds of small .jsonl files into single .json file.

    Args:
        path_to_save (str): path where to save merged file.
        source_dir (str): path to decompressed dataset.
        remove_small_files (bool): whether to remove small .jsonl files after merging.
    """
    start_time = time.time()
    print("Merging files...")

    cmd = f"cd {source_dir} && awk '1' *.jsonl > {path_to_save}"
    if remove_small_files:
        cmd += " && rm shard_*"
    subprocess.run(cmd, shell=True, check=True)

    end_time = time.time()
    elapsed_minutes = np.round((end_time - start_time) / 60, 0)
    print(f"Files were successfully merged into {path_to_save} in {elapsed_minutes} minutes.")


if __name__ == "__main__":
    args = arguments().parse_args()

    merge_data(
        path_to_save=args.path_to_save,
        source_dir=args.source_dir,
        remove_small_files=args.remove_small_files,
    )
