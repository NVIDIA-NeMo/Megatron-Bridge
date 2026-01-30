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
import glob
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

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


def _decompress_file(zst_file: str, output_dir: str) -> None:
    """Helper function to decompress a single .zst file.

    Args:
        zst_file (str): path to .zst file to decompress.
        output_dir (str): directory where to save decompressed file.
    """
    basename = os.path.basename(zst_file)
    # Remove .zst extension
    output_name = basename[:-4] if basename.endswith(".zst") else basename
    output_file = os.path.join(output_dir, output_name)
    subprocess.run(["zstd", "-d", zst_file, "-o", output_file], check=True)


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

    # Find all .zst files recursively
    zst_files = glob.glob(os.path.join(source_dir, "**", "*.zst"), recursive=True)

    if not zst_files:
        print("No .zst files found")
        return

    # Decompress files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(_decompress_file, zst_files, [path_to_save] * len(zst_files))

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
