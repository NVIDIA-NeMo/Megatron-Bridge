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
import time

import numpy as np
import requests
from huggingface_hub import login, snapshot_download


def arguments():
    """Argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--token", type=str, required=True, help="HF access token.")
    parser.add_argument(
        "--path_to_save",
        type=str,
        required=True,
        help="Path where to save downloaded dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers to be used to download data.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Number of download retries if timeout reached.",
    )
    parser.add_argument(
        "--retry_delay",
        type=int,
        default=10,
        help="Delay (in seconds) between retries.",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_false",
        help="Ignore previously downloaded files and start from scratch.",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default="*.jsonl.zst",
        help="Patterns to download specific subdataset.",
    )

    return parser


def download_dataset(
    path_to_save: str,
    num_workers: int = 1,
    max_retries: int = 5,
    retry_delay: int = 10,
    resume_download: bool = True,
    patterns: str | list = "*.jsonl.zst",
) -> None:
    """Downloads DCLM dataset from HF

    Args:
        path_to_save (str): path where to save downloaded dataset.
        num_workers (int): number of workers to be used for parallel downloading.
        max_retries (int): max number of donwload retries when error has been reached.
        retry_delay (int): delay in seconds between code retries.
        resume_download (bool): whether to resume download from latest saved datafile.
        patterns (str | list): patterns to download specific subdataset.
    """
    start_time = time.time()

    print("Downloading dataset...")
    if isinstance(patterns, str):
        allow_patterns = [patterns]
    else:
        allow_patterns = patterns

    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id="mlfoundations/dclm-baseline-1.0",
                repo_type="dataset",
                local_dir=path_to_save,
                allow_patterns=allow_patterns,
                resume_download=resume_download,
                max_workers=num_workers,  # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise

    end_time = time.time()
    elapsed_minutes = np.round((end_time - start_time) / 60, 0)
    print(f"Dataset was downloaded to {path_to_save} in {elapsed_minutes} minutes.")


if __name__ == "__main__":
    args = arguments().parse_args()

    # login to HF
    login(token=args.token)

    # donwload dataset
    download_dataset(
        path_to_save=args.path_to_save,
        num_workers=args.num_workers,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        patterns=args.patterns,
        resume_download=not args.from_scratch,
    )
