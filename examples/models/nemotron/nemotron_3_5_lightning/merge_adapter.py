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
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    return parser.parse_args()


def _last_token_logits(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str) -> torch.Tensor:
    """Return final-token logits on CPU in float32."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)
    model.eval()
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()


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
    adapter_effect = (peft_logits - base_logits).abs().max().item()
    if adapter_effect == 0.0:
        raise RuntimeError("The adapter has no observable effect on the verification prompt.")

    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_logits = _last_token_logits(merged_model, tokenizer, args.prompt)
    torch.testing.assert_close(merged_logits, peft_logits, atol=args.atol, rtol=args.rtol)
    if torch.topk(merged_logits, 5).indices.tolist() != torch.topk(peft_logits, 5).indices.tolist():
        raise RuntimeError("Merged and unmerged PEFT models have different top-5 tokens.")

    args.output.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    LOGGER.info("Adapter effect max logit difference: %.6e", adapter_effect)
    LOGGER.info("Merge max logit difference: %.6e", (merged_logits - peft_logits).abs().max().item())
    LOGGER.info("Merged model saved to %s", args.output)


if __name__ == "__main__":
    main()
