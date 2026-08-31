#!/usr/bin/env python
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

"""Run masked-token ("fill-mask") inference with a Megatron-Bert-family checkpoint.

Unlike the causal-LM examples in `examples/conversion/`, `MegatronBertForMaskedLM`
does not implement `generate()`, so this example uses the Hugging Face `fill-mask`
pipeline directly. This is intentionally HF-only (no Megatron-Core forward pass):
its purpose is to sanity-check that a checkpoint produced by the Megatron-Bridge
BERT bridge (see `conversion.sh`) still makes sensible predictions after a
round trip, not to benchmark Megatron inference.

Run Script Example:
    uv run python examples/models/bert/megatron-bert/fill_mask.py \\
        --hf_model_path /workspace/models/megatron-bert-uncased-345m-hf-export \\
        --text "Paris is the [MASK] of France."
"""

import argparse

from transformers import pipeline


def main() -> None:
    """Parse CLI arguments and print the top masked-token predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_model_path", type=str, required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--text",
        type=str,
        default="Paris is the [MASK] of France.",
        help="Input text containing exactly one `[MASK]` token",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions to print")
    args = parser.parse_args()

    fill_mask = pipeline("fill-mask", model=args.hf_model_path)
    predictions = fill_mask(args.text, top_k=args.top_k)

    print(f"Model: {args.hf_model_path}")
    print(f"Input: {args.text}")
    for prediction in predictions:
        print(f"  {prediction['score']:.4f}  {prediction['sequence']}")


if __name__ == "__main__":
    main()
