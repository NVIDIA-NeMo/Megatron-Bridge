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
import time
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.filters import RepeatingTopNGramsFilter, UrlsFilter, WordCountFilter
from nemo_curator.stages.text.io.writer import MegatronTokenizerWriter
from nemo_curator.stages.text.modules import ScoreFilter


def main(args: argparse.Namespace) -> None:  # noqa: D103
    # Initialize and start the Ray client
    ray_client = RayClient()
    ray_client.start()

    raw_dir = os.path.join(args.output_dir, "raw")
    tokens_dir = os.path.join(args.output_dir, "tokens")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)

    print("Filtering and tokenizing Common Crawl data for Megatron-LM training")
    print(f"    The raw dataset will be written to '{raw_dir}'")
    print(f"    The filtered and tokenized dataset will be written to '{tokens_dir}'")

    # Create a pipeline with the stages
    pipeline = Pipeline(
        name="commoncrawl-filter-and-tokenize",
        description="Filter and tokenize Common Crawl data for Megatron-LM training.",
    )
    # Download and extract the Common Crawl data
    pipeline.add_stage(
        CommonCrawlDownloadExtractStage(
            start_snapshot=args.start_snapshot,
            end_snapshot=args.end_snapshot,
            download_dir=raw_dir,
            crawl_type="main",
            use_aws_to_download=args.use_aws_to_download,
            verbose=True,
            url_limit=args.url_limit,
        )
    )

    # Filter the data
    if args.filter_data:
        ## Filter short documents
        pipeline.add_stage(ScoreFilter(filter_obj=WordCountFilter(min_words=50)))
        ## Filter documents with too many URLs
        pipeline.add_stage(ScoreFilter(filter_obj=UrlsFilter(max_url_to_text_ratio=0.1)))
        ## Filter documents with too many repeating n-grams
        pipeline.add_stage(ScoreFilter(filter_obj=RepeatingTopNGramsFilter()))

    # Tokenize the data
    pipeline.add_stage(
        MegatronTokenizerWriter(
            path=tokens_dir,
            model_identifier=args.tokenizer_model,
            append_eod=args.append_eod,
        )
    )

    print("Starting the filtering and tokenization pipeline")
    start_time = time.time()
    # Run the pipeline
    results = pipeline.run()
    end_time = time.time()
    execution_time = end_time - start_time
    # Count the total number of records
    print(f"\n\nFiltering and tokenization pipeline finished (took {execution_time} seconds)")
    print(f"The results were written to '{[result.data for result in results]}'")

    # Stop the Ray client
    ray_client.stop()

    # Create --data-args-path file
    data_args_path = os.path.join(args.output_dir, "dataset-prefixes.txt")
    file_prefixes = [str(file)[:-4] for file in Path(tokens_dir).glob("**/*.bin")]
    with open(data_args_path, "w") as f:
        for file_prefix in file_prefixes:
            f.write(file_prefix + "\n")

    print(f"The --data-args-path file was written to '{data_args_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    group.add_argument(
        "--tokenizer-model", type=str, required=True, help="Hugging Face model identifier for the tokenizer"
    )
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of each sample.")
    group.add_argument(
        "--filter-data",
        action="store_true",
        help="Filter short documents, documents with too many URLs, and documents with too many repeating n-grams using NeMo Curator's filters",
    )
    group.add_argument(
        "--use-aws-to-download",
        action="store_true",
        help="Use the s5cmd command to download from the Common Crawl's S3 bucket",
    )
    group.add_argument("--start-snapshot", type=str, default="2025-30", help="Start snapshot to download")
    group.add_argument("--end-snapshot", type=str, default="2025-30", help="End snapshot to download")
    group.add_argument(
        "--url-limit",
        type=int,
        default=2,
        help="Limit the number of URLs/WARC files to download. Each WARC file is ~1GB",
    )
    args = parser.parse_args()
    main(args)
