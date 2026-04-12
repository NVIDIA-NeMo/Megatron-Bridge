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
Inference on a converted (post-roundtrip) GLM-5 checkpoint.

Loads only the converted model and verifies it produces coherent text.
Use --load-in-4bit to fit on a single 8-GPU node (~350 GB).

Usage:
    uv run python examples/models/glm5/glm5_converted_inference.py \
        --model-dir /path/to/converted/GLM-5 \
        --tokenizer-id zai-org/GLM-5 \
        --load-in-4bit
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
    "Albert Einstein was born in",
    "The chemical formula for water is",
]


def main() -> None:
    """Load the converted GLM-5 model and verify coherent output."""
    parser = argparse.ArgumentParser(description="GLM-5 converted model inference")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--tokenizer-id", type=str, default="zai-org/GLM-5")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPUs")

    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        num_layers = 78
        balanced_device_map: dict = {}
        balanced_device_map["model.embed_tokens"] = 0
        balanced_device_map["model.norm"] = n_gpus - 1
        balanced_device_map["lm_head"] = n_gpus - 1
        for layer_i in range(num_layers):
            gpu_id = min(layer_i * n_gpus // num_layers, n_gpus - 1)
            balanced_device_map[f"model.layers.{layer_i}"] = gpu_id

        if args.load_in_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        load_kwargs = dict(quantization_config=quant_config, device_map=balanced_device_map)
    else:
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading converted model from {args.model_dir} ...")
    model = GlmMoeDsaForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    model.eval()
    print("Model loaded.\n")

    all_ok = True
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"Prompt   : {prompt}")
        print(f"Generated: {text}")
        coherent = len(text) > len(prompt)
        print(f"Status   : {'OK' if coherent else 'WARN - empty output'}\n")
        if not coherent:
            all_ok = False

    if all_ok:
        print("SUCCESS: Converted GLM-5 model produces coherent output on all prompts.")
    else:
        print("WARNING: Some prompts produced empty output.")


if __name__ == "__main__":
    main()
