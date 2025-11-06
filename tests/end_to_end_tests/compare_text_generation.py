#!/usr/bin/env python3
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

# TODO: add file docstring

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def transformers_generate(hf_model: str, prompt: str, max_new_tokens: int = 20) -> list[str]:
    """
    Generate text from a HuggingFace model using transformers.

    This serves as the baseline for this test.

    Args:
        model_id: HuggingFace model ID or path to model directory
        prompt: Input text for the model
        max_new_tokens: Upper bound on how many tokens to generate.
           May generate fewer tokens than this limit. (default: 20)
    """
    print(f"Loading model and tokenizer: {hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True).cuda()

    print("Generating text using HF weights...")
    in_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_tokens = model.generate(
        **in_tokens,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,  # TODO: not using yet, but should compare this after script is proven for text
    )

    generated_text = tokenizer.decode(out_tokens.sequences[0])

    print("====== HF GENERATED TEXT OUTPUT ======")
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("======================================")
    return generated_text


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Compare text generated through various Megatron-Bridge conversion methods against HuggingFace direct"
    )

    parser.add_argument("--hf-model-id", type=str, required=True, help="Repo or path to the HuggingFace model.")
    parser.add_argument("--prompt", type=str, default="What is a GPU?", help="Input prompt for text generation.")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Maximum number of new tokens to generate.")

    return parser.parse_args()


def main():
    args = parse_cli_args()

    _ = transformers_generate(args.hf_model_id, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()
