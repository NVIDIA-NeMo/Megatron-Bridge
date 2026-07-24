#!/usr/bin/env python3
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

"""Run greedy, exact-length inference from an exported HF checkpoint."""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any


LOG = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, help="Exported HF model directory.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument("--image-path", help="Optional local path or public HTTP(S) image URL.")
    parser.add_argument("--max-new-tokens", required=True, type=int, help="Exact number of tokens to generate.")
    parser.add_argument("--chat-template", action="store_true", help="Format the prompt as a user chat turn.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model and tokenizer code from the selected Hugging Face repository.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to the tokenizer chat template.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device used for inference.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model loading dtype.",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.disable_thinking and not args.chat_template:
        parser.error("--disable-thinking requires --chat-template")
    return args


def _format_prompt(tokenizer: Any, prompt: str, *, chat_template: bool, disable_thinking: bool) -> str:
    if not chat_template:
        return prompt
    template_options = {"enable_thinking": False} if disable_thinking else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **template_options,
    )


def _read_public_image_url(image_url: str) -> bytes:
    """Read an allowlisted public image URL with Bridge's SSRF protections."""
    from megatron.bridge.utils.safe_url import is_safe_public_http_url, safe_url_open

    is_safe, reason = is_safe_public_http_url(image_url)
    if not is_safe:
        raise ValueError(f"Refusing to fetch image URL ({reason}): {image_url}")
    with safe_url_open(image_url) as response:
        return response.read()


def _load_image(image_path: str) -> Any:
    """Load one RGB image from a local path or safe public URL."""
    from PIL import Image

    if image_path.startswith(("http://", "https://")):
        return Image.open(io.BytesIO(_read_public_image_url(image_path))).convert("RGB")
    return Image.open(Path(image_path)).convert("RGB")


def _prepare_vlm_inputs(processor: Any, prompt: str, image_path: str) -> Any:
    """Build one image-and-text chat batch without relying on model-specific helpers."""
    image = _load_image(image_path)
    if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None) is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    tokenizer = getattr(processor, "tokenizer", processor)
    image_token = getattr(tokenizer, "image_token", "<|image|>")
    return processor(text=[f"{image_token}\n{prompt}"], images=[image], return_tensors="pt")


def _move_inputs(inputs: Any, device: Any, dtype: Any) -> dict[str, Any]:
    """Move tensor inputs to the selected device and cast floating inputs."""
    moved = {}
    for name, value in dict(inputs).items():
        if not hasattr(value, "to"):
            moved[name] = value
        elif getattr(value, "is_floating_point", lambda: False)():
            moved[name] = value.to(device=device, dtype=dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


def main() -> int:
    """Run one exact-length greedy generation and print its completion."""
    args = _parse_args()

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    if args.image_path:
        config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        architectures = getattr(config, "architectures", None) or []
        if len(architectures) != 1 or not hasattr(transformers, architectures[0]):
            raise RuntimeError(
                "Vision-language inference requires one loadable Transformers architecture in config.architectures"
            )
        model_class = getattr(transformers, architectures[0])
        processor = AutoProcessor.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        model_class = AutoModelForCausalLM
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
    model = (
        model_class.from_pretrained(
            args.hf_model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        .to(args.device)
        .eval()
    )
    if processor is not None:
        inputs = _prepare_vlm_inputs(processor, args.prompt, args.image_path)
    else:
        formatted_prompt = _format_prompt(
            tokenizer,
            args.prompt,
            chat_template=args.chat_template,
            disable_thinking=args.disable_thinking,
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = _move_inputs(inputs, model.device, dtype)
    prompt_length = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            min_new_tokens=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=pad_token_id,
        )

    expected_length = prompt_length + args.max_new_tokens
    if output.shape != (1, expected_length):
        observed_length = output.shape[1] - prompt_length
        raise RuntimeError(f"Expected exactly {args.max_new_tokens} generated tokens; observed {observed_length}")

    completion_ids = output[0, prompt_length:].tolist()
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    LOG.info("HF completion (%d tokens): %s", args.max_new_tokens, json.dumps(completion, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
