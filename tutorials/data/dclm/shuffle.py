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
        help="Path where to save shuffled file.",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        required=True,
        help="Path to .jsonl file to be shuffled.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used to shuffle data.",
    )
    parser.add_argument(
        "--lines_per_split",
        type=int,
        default=1000000,
        help="Number lines per every splitted file.",
    )

    return parser


def shuffle_data(
    path_to_save: str,
    source_file: str,
    num_workers: int = 1,
    lines_per_split: int = 1000000,
) -> None:
    """Shuffles .jsonl file.

    Args:
        path_to_save (str): path where to save shuffled file.
        source_file (str): path to merged file.
        num_workers (int): number of workers to be used for parallel shuffling.
        lines_per_split (int): lines per file to split for parallel shuffling.
    """
    start_time = time.time()
    print("Shuffling file...")

    source_dir = os.path.dirname(source_file)
    chunks_dir = os.path.join(source_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    shuffle_chunks_dir = os.path.join(source_dir, "shuffled_chunks")
    os.makedirs(shuffle_chunks_dir, exist_ok=True)
    cmd = (
        f"split -l {lines_per_split} {source_file} {chunks_dir}/chunk_ && "
        f"ls {chunks_dir}/chunk_* | parallel -j{num_workers} "
        f"'shuf {{}} -o {shuffle_chunks_dir}/$(basename {{}})_shuf' && "
        f"rm -rf {chunks_dir} && "
        f"awk '1' {shuffle_chunks_dir}/chunk_* > {path_to_save} && "
        f"rm -rf {shuffle_chunks_dir}"
    )
    subprocess.run(cmd, shell=True, check=True)

    end_time = time.time()
    elapsed_minutes = np.round((end_time - start_time) / 60, 0)
    print(f"File was successfully shuffled into {path_to_save} in {elapsed_minutes} minutes.")


if __name__ == "__main__":
    args = arguments().parse_args()

    shuffle_data(
        path_to_save=args.path_to_save,
        source_file=args.source_file,
        num_workers=args.num_workers,
        lines_per_split=args.lines_per_split,
    )
