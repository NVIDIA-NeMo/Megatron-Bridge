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
import os
import shlex
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
        help="Path where to save decompressed files.",
    )
    parser.add_argument("--source_dir", type=str, required=True, help="Path to downloaded dataset.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used to decompress data.",
    )

    return parser


def decompress_data(path_to_save: str, source_dir: str, num_workers: int = 1) -> None:
    """Decompresses downloaded dataset

    Args:
        path_to_save (str): path where to save downloaded dataset.
        source_dir (str): path to downloaded dataset.
        num_workers (int): number of workers to be used for parallel decompressing.
    """
    start_time = time.time()
    print("Decompressing files...")

    os.makedirs(path_to_save, exist_ok=True)
    cmd = (
        f"mkdir -p {shlex.quote(path_to_save)} && "
        f"cd {shlex.quote(source_dir)} && "
        f'find . -name "*.zst" | '
        f"parallel -j{num_workers} "
        '"zstd -d {} -o ' + shlex.quote(path_to_save) + '/{.}"'
    )
    subprocess.run(cmd, shell=True, check=True)

    end_time = time.time()
    elapsed_minutes = np.round((end_time - start_time) / 60, 0)
    print(f"Files were successfully decompressed in {elapsed_minutes} minutes.")


if __name__ == "__main__":
    args = arguments().parse_args()

    decompress_data(
        path_to_save=args.path_to_save,
        source_dir=args.source_dir,
        num_workers=args.num_workers,
    )
