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

"""
Baseline inference on the original zai-org/GLM-5 from HF Hub.

Verifies the hardware can load and run the model, establishing the
baseline that the converted model should match.

Requires 8+ nodes with device_map="auto" in BF16, or use --load-in-4bit
to fit on a single 8-GPU node (~350 GB).

Usage:
    uv run python examples/models/glm5/glm5_orig_inference.py
    uv run python examples/models/glm5/glm5_orig_inference.py --load-in-4bit
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, GlmMoeDsaForCausalLM


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROMPTS = [
    "The capital of France is",
    "In machine learning, a transformer is",
    "The largest planet in the solar system is",
]


def main() -> None:
    """Load and run inference on the original GLM-5 model."""
    parser = argparse.ArgumentParser(description="GLM-5 baseline inference")
    parser.add_argument("--model-id", type=str, default="zai-org/GLM-5")
    parser.add_argument("--tokenizer-id", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    tokenizer_id = args.tokenizer_id or args.model_id
    print(f"Loading tokenizer from {tokenizer_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        load_kwargs = dict(quantization_config=quant_config, device_map="auto")
        print("Loading in 4-bit quantization ...")
    else:
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading model from {args.model_id} ...")
    model = GlmMoeDsaForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model.eval()
    print("Model loaded.\n")

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Prompt   : {prompt}")
        print(f"Generated: {text}\n")

    print("INFERENCE_SUCCESS")


if __name__ == "__main__":
    main()
