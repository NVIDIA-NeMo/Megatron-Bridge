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

"""Create tiny chat JSONL files for the Hugging Face text-only tutorial."""

import argparse
import json
from pathlib import Path


EXAMPLES = {
    "training": [
        ("What owns runtime construction?", "The dataset builder."),
        ("What belongs in config?", "Validated declarative data."),
        ("What renders this chat?", "The tokenizer chat template."),
        ("What is in-batch packing?", "Packing examples during collation."),
    ],
    "validation": [("What is JSONL?", "One JSON object per line.")],
    "test": [("What is an adapter?", "A registered dataset normalization function.")],
}


def prepare_example_data(output_dir: Path) -> None:
    """Write deterministic ``messages`` rows for each tutorial split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in EXAMPLES.items():
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as output_file:
            for prompt, answer in examples:
                row = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                }
                output_file.write(json.dumps(row) + "\n")


def main() -> None:
    """Parse arguments and create the example dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_example_data(args.output_dir)


if __name__ == "__main__":
    main()
