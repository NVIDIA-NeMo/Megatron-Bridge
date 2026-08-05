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
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as functional
from peft import PeftModel
from peft_compat import apply_peft_weight_converter_compatibility
from safetensors import safe_open
from safetensors.torch import save_file
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


def _resolve_hf_path(model_name_or_path: str, revision: str) -> Path:
    """Resolve a local directory or pinned Hub model to its snapshot path."""
    local_path = Path(model_name_or_path)
    if local_path.is_dir():
        return local_path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        )
    )


def _merge_training_only_mtp_weights(
    *,
    hf_model: str,
    hf_revision: str,
    adapter_path: Path,
    output: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[int, int]:
    """Append the MTP weights omitted by the HF inference model.

    Transformers does not instantiate Nemotron H's training-only MTP modules,
    so ``merge_and_unload`` cannot include them. This function carries every
    MTP tensor from the base checkpoint and applies each corresponding LoRA
    pair before adding a dedicated shard to the merged checkpoint.

    Returns:
        The number of merged MTP tensors and the number carried unchanged.
    """
    base_path = _resolve_hf_path(hf_model, hf_revision)
    base_index_path = base_path / "model.safetensors.index.json"
    output_index_path = output / "model.safetensors.index.json"
    adapter_weights_path = adapter_path / "adapter_model.safetensors"
    adapter_config_path = adapter_path / "adapter_config.json"
    for required_path in (
        base_index_path,
        output_index_path,
        adapter_weights_path,
        adapter_config_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required merge input does not exist: {required_path}")

    base_index = json.loads(base_index_path.read_text())
    output_index = json.loads(output_index_path.read_text())
    adapter_config = json.loads(adapter_config_path.read_text())
    if adapter_config.get("use_dora") or adapter_config.get("use_rslora"):
        raise ValueError("Training-only MTP merge currently supports standard LoRA only.")
    if adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern"):
        raise ValueError("Training-only MTP merge does not support per-module rank or alpha patterns.")
    scaling = float(adapter_config["lora_alpha"]) / int(adapter_config["r"])

    base_weight_map: dict[str, str] = base_index["weight_map"]
    output_weight_map: dict[str, str] = output_index["weight_map"]
    mtp_base_keys = sorted(key for key in base_weight_map if key.startswith("mtp."))
    if not mtp_base_keys:
        raise RuntimeError("The base checkpoint has no MTP tensors to preserve.")
    duplicate_keys = set(mtp_base_keys).intersection(output_weight_map)
    if duplicate_keys:
        raise RuntimeError(f"The HF merge unexpectedly emitted MTP tensors: {sorted(duplicate_keys)[:5]}")

    adapter_prefix = "base_model.model."
    lora_a_suffix = ".lora_A.weight"
    lora_b_suffix = ".lora_B.weight"
    mtp_shard_name = "model-mtp.safetensors"
    mtp_shard_path = output / mtp_shard_name
    if mtp_shard_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing MTP shard: {mtp_shard_path}")

    merged_state: dict[str, torch.Tensor] = {}
    merged_count = 0
    unchanged_count = 0
    with ExitStack() as stack:
        base_readers = {
            filename: stack.enter_context(safe_open(base_path / filename, framework="pt", device="cpu"))
            for filename in set(base_weight_map.values())
        }
        adapter_reader = stack.enter_context(safe_open(adapter_weights_path, framework="pt", device="cpu"))
        adapter_keys = set(adapter_reader.keys())
        mtp_lora_a_keys = {
            key for key in adapter_keys if key.startswith(f"{adapter_prefix}mtp.") and key.endswith(lora_a_suffix)
        }
        expected_lora_keys: set[str] = set()

        with torch.no_grad():
            for base_key in mtp_base_keys:
                base_tensor = (
                    base_readers[base_weight_map[base_key]].get_tensor(base_key).to(device=device, dtype=dtype)
                )
                module_name = base_key.removesuffix(".weight")
                lora_a_key = f"{adapter_prefix}{module_name}{lora_a_suffix}"
                lora_b_key = f"{adapter_prefix}{module_name}{lora_b_suffix}"
                if lora_a_key in adapter_keys or lora_b_key in adapter_keys:
                    if lora_a_key not in adapter_keys or lora_b_key not in adapter_keys:
                        raise RuntimeError(f"Incomplete MTP LoRA pair for {base_key}.")
                    calculation_dtype = dtype
                    if device.type == "cpu" and dtype in (torch.bfloat16, torch.float16):
                        calculation_dtype = torch.float32
                    weight_a = adapter_reader.get_tensor(lora_a_key).to(
                        device=device,
                        dtype=calculation_dtype,
                    )
                    weight_b = adapter_reader.get_tensor(lora_b_key).to(
                        device=device,
                        dtype=calculation_dtype,
                    )
                    delta = (weight_b @ weight_a) * scaling
                    merged_tensor = base_tensor + delta.to(dtype=dtype)
                    expected_lora_keys.update((lora_a_key, lora_b_key))
                    merged_count += 1
                else:
                    merged_tensor = base_tensor
                    unchanged_count += 1
                merged_state[base_key] = merged_tensor.cpu().contiguous()

        unconsumed_lora_a = mtp_lora_a_keys.difference(expected_lora_keys)
        if unconsumed_lora_a:
            raise RuntimeError(f"MTP LoRA weights do not map to base tensors: {sorted(unconsumed_lora_a)[:5]}")

    save_file(merged_state, mtp_shard_path, metadata={"format": "pt"})
    mtp_size = sum(tensor.numel() * tensor.element_size() for tensor in merged_state.values())
    output_weight_map.update(dict.fromkeys(mtp_base_keys, mtp_shard_name))
    output_index.setdefault("metadata", {})["total_size"] = int(
        output_index.get("metadata", {}).get("total_size", 0)
    ) + int(mtp_size)
    output_index_path.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")
    return merged_count, unchanged_count


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
    merge_device = next(merged_model.parameters()).device
    del base_model, peft_model, merged_model
    if merge_device.type == "cuda":
        torch.cuda.empty_cache()
    merged_mtp_count, unchanged_mtp_count = _merge_training_only_mtp_weights(
        hf_model=args.hf_model,
        hf_revision=args.hf_revision,
        adapter_path=args.adapter_path,
        output=args.output,
        dtype=DTYPES[args.dtype],
        device=merge_device,
    )
    LOGGER.info("Merged %d training-only MTP tensors with LoRA.", merged_mtp_count)
    LOGGER.info("Carried %d non-LoRA MTP tensors unchanged.", unchanged_mtp_count)
    LOGGER.info("Merged model saved to %s", args.output)


if __name__ == "__main__":
    main()
