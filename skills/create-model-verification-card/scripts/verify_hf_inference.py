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

"""Verify deterministic inference from an exported HF checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any
from urllib.parse import urlparse


LOG = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", required=True, help="Exported HF model directory.")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument(
        "--image",
        help="Optional local image path or URL. Uses the model processor and a multimodal chat template.",
    )
    parser.add_argument("--max-new-tokens", required=True, type=int, help="Maximum number of tokens to generate.")
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
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.disable_thinking and not args.chat_template:
        parser.error("--disable-thinking requires --chat-template")
    if args.image and not args.chat_template:
        parser.error("--image requires --chat-template")
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


def _image_content(image: str) -> dict[str, str]:
    """Return one processor-native image content block."""
    location_key = "url" if urlparse(image).scheme in {"http", "https"} else "path"
    return {"type": "image", location_key: image}


def _prepare_inputs(processor: Any, args: argparse.Namespace) -> Any:
    """Prepare text-only or processor-native multimodal model inputs."""
    if args.image:
        template_options = {"enable_thinking": False} if args.disable_thinking else {}
        messages = [
            {
                "role": "user",
                "content": [
                    _image_content(args.image),
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_options,
        )

    formatted_prompt = _format_prompt(
        processor,
        args.prompt,
        chat_template=args.chat_template,
        disable_thinking=args.disable_thinking,
    )
    return processor(formatted_prompt, return_tensors="pt")


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load torch, the selected HF auto-model, and its tokenizer or processor."""
    import torch

    dtype = getattr(torch, args.dtype)
    if args.image:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        model_cls = AutoModelForMultimodalLM
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        processor = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=args.trust_remote_code)
        model_cls = AutoModelForCausalLM
    model = (
        model_cls.from_pretrained(
            args.hf_model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        .to(args.device)
        .eval()
    )
    return torch, model, processor


def _processor_tokenizer(processor: Any) -> Any:
    """Return the tokenizer nested in a processor, or the tokenizer itself."""
    return getattr(processor, "tokenizer", processor)


def main(argv: list[str] | None = None) -> int:
    """Run two greedy generations and print their shared completion."""
    args = _parse_args(argv)
    torch, model, processor = _load_runtime(args)
    tokenizer = _processor_tokenizer(processor)
    inputs = _prepare_inputs(processor, args).to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            outputs.append(
                model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_token_id,
                )
            )

    if not torch.equal(outputs[0], outputs[1]):
        raise RuntimeError("Two greedy HF inference runs produced different token IDs")

    completion_ids = outputs[0][0, prompt_length:].tolist()
    completion = processor.decode(completion_ids, skip_special_tokens=True)
    LOG.info(
        "HF completion (%d generated tokens; maximum %d): %s",
        len(completion_ids),
        args.max_new_tokens,
        json.dumps(completion, ensure_ascii=False),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
