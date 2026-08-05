# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Merge a Hugging Face PEFT adapter into Nemotron 3.5 Lightning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as functional
from peft import PeftModel
from peft_compat import apply_peft_weight_converter_compatibility
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
DEFAULT_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-revision", default=DEFAULT_REVISION)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.995)
    parser.add_argument("--max-relative-l2", type=float, default=0.1)
    return parser.parse_args()


def _last_token_logits(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str) -> torch.Tensor:
    """Return final-token logits on CPU in float32."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    model.eval()
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()


def _greedy_continuation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Return greedily generated continuation token IDs on CPU."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return generated[0, inputs["input_ids"].shape[1] :].cpu()


def main() -> None:
    """Load, verify, merge, and save the adapter."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    apply_peft_weight_converter_compatibility()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, revision=args.hf_revision)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        revision=args.hf_revision,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device_map,
    )
    base_logits = _last_token_logits(base_model, tokenizer, args.prompt)

    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    peft_logits = _last_token_logits(peft_model, tokenizer, args.prompt)
    peft_continuation = _greedy_continuation(
        peft_model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
    )
    adapter_effect = (peft_logits - base_logits).abs().max().item()
    if adapter_effect == 0.0:
        raise RuntimeError("The adapter has no observable effect on the verification prompt.")

    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_logits = _last_token_logits(merged_model, tokenizer, args.prompt)
    merged_continuation = _greedy_continuation(
        merged_model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
    )
    difference = merged_logits - peft_logits
    merge_difference = difference.abs().max().item()
    relative_l2 = (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(peft_logits)).item()
    cosine_similarity = functional.cosine_similarity(merged_logits, peft_logits, dim=0).item()
    merged_effect = (merged_logits - base_logits).abs().max().item()
    peft_top_5 = torch.topk(peft_logits, 5).indices.tolist()
    merged_top_5 = torch.topk(merged_logits, 5).indices.tolist()

    LOGGER.info("Adapter effect max logit difference: %.6e", adapter_effect)
    LOGGER.info("Merged effect max logit difference: %.6e", merged_effect)
    LOGGER.info("Merge max logit difference: %.6e", merge_difference)
    LOGGER.info("Merge relative L2 error: %.6e", relative_l2)
    LOGGER.info("Merge cosine similarity: %.9f", cosine_similarity)
    LOGGER.info("PEFT top-5 token IDs: %s", peft_top_5)
    LOGGER.info("Merged top-5 token IDs: %s", merged_top_5)
    LOGGER.info("PEFT continuation: %s", tokenizer.decode(peft_continuation))
    LOGGER.info("Merged continuation: %s", tokenizer.decode(merged_continuation))

    if merged_effect == 0.0:
        raise RuntimeError("The merged checkpoint has no observable effect on the verification prompt.")
    if peft_top_5[0] != merged_top_5[0]:
        raise RuntimeError("Merged and unmerged PEFT models have different top-1 tokens.")
    if not torch.equal(merged_continuation, peft_continuation):
        raise RuntimeError("Merged and unmerged PEFT models have different greedy continuations.")
    if cosine_similarity < args.min_cosine_similarity:
        raise RuntimeError(
            f"Merge cosine similarity {cosine_similarity:.9f} is below {args.min_cosine_similarity:.9f}."
        )
    if relative_l2 > args.max_relative_l2:
        raise RuntimeError(f"Merge relative L2 error {relative_l2:.6e} exceeds {args.max_relative_l2:.6e}.")

    args.output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    LOGGER.info("Merged model saved to %s", args.output)


if __name__ == "__main__":
    main()
