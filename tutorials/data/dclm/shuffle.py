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


def _shuffle_chunk(chunk_file: str, output_dir: str) -> None:
    """Helper function to shuffle a single chunk file.

    Args:
        chunk_file (str): path to chunk file to shuffle.
        output_dir (str): directory where to save shuffled chunk.
    """
    basename = os.path.basename(chunk_file)
    output_file = os.path.join(output_dir, f"{basename}_shuf")
    subprocess.run(["shuf", chunk_file, "-o", output_file], check=True)


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

    try:
        # Split the file into chunks
        subprocess.run(
            ["split", "-l", str(lines_per_split), source_file, os.path.join(chunks_dir, "chunk_")], check=True
        )

        # Shuffle chunks in parallel
        chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            executor.map(_shuffle_chunk, chunk_files, [shuffle_chunks_dir] * len(chunk_files))

        # Merge shuffled chunks
        shuffled_chunks = sorted(glob.glob(os.path.join(shuffle_chunks_dir, "chunk_*")))
        awk_cmd = ["awk", "1"] + shuffled_chunks
        with open(path_to_save, "w") as output_file:
            subprocess.run(awk_cmd, stdout=output_file, check=True)

    finally:
        # Cleanup temporary directories
        for chunk_file in glob.glob(os.path.join(chunks_dir, "chunk_*")):
            os.remove(chunk_file)
        if os.path.exists(chunks_dir):
            os.rmdir(chunks_dir)

        for shuffled_file in glob.glob(os.path.join(shuffle_chunks_dir, "chunk_*")):
            os.remove(shuffled_file)
        if os.path.exists(shuffle_chunks_dir):
            os.rmdir(shuffle_chunks_dir)

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
