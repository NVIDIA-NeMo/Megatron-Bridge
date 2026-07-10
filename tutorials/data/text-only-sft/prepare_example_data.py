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

"""Create a tiny local JSONL dataset for the text-only SFT tutorial."""

import argparse
import json
from pathlib import Path


EXAMPLES = {
    "training": [
        {"input": "What does Megatron Bridge connect?", "output": "Hugging Face and Megatron Core."},
        {"input": "Name the indexed pretraining files.", "output": "The .bin and .idx files."},
        {"input": "What is SFT?", "output": "Supervised fine-tuning."},
        {"input": "What owns runtime dataset creation?", "output": "The dataset builder."},
    ],
    "validation": [
        {"input": "What belongs in a dataset config?", "output": "Validated declarative settings."},
        {"input": "What does PEFT train?", "output": "A parameter-efficient subset such as LoRA adapters."},
    ],
    "test": [
        {"input": "What is JSONL?", "output": "One JSON object per line."},
    ],
}


def prepare_example_data(output_dir: Path) -> None:
    """Write deterministic tutorial splits under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, examples in EXAMPLES.items():
        split_path = output_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as output_file:
            for example in examples:
                output_file.write(json.dumps(example) + "\n")


def main() -> None:
    """Parse arguments and create the example dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare_example_data(args.output_dir)


if __name__ == "__main__":
    main()
