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

"""Extract a Qwen3.5 MoE HF text checkpoint, retaining pretrained MTP.

Run this on a local, unquantized VL snapshot before ``run_conversion.py import``.
Tensors are copied on CPU, one shard at a time, without constructing a
Transformers model that might omit auxiliary MTP weights from its state dict.
"""

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)


def text_weight_name(name: str) -> str | None:
    """Map a VL tensor to its text-only HF key, or omit a vision tensor."""
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name == "lm_head.weight" or name.startswith("mtp."):
        return name
    if name.startswith("model.visual."):
        return None
    raise ValueError(f"Unexpected checkpoint tensor (not silently discarded): {name}")


def text_model_config(config: dict, names: list[str]) -> dict:
    """Flatten the VL text config and require one pretrained MoE MTP layer."""
    text = dict(config.get("text_config", {}))
    if text.get("model_type") != "qwen3_5_moe_text":
        raise ValueError("Expected a Qwen3.5 MoE VL checkpoint with qwen3_5_moe_text text_config.")
    if config.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Use an unquantized checkpoint; extraction does not dequantize weights.")
    layers = sorted({int(match[1]) for name in names if (match := re.match(r"mtp\.layers\.(\d+)\.", name))})
    if layers != [0]:
        raise ValueError(f"This baseline requires exactly one pretrained MTP layer; found {layers}.")
    required = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "mtp.fc.weight",
        "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
    missing = required.difference(names)
    if missing:
        raise ValueError(f"Missing decoder/MTP tensors: {sorted(missing)}")
    text["architectures"] = ["Qwen3_5MoeForCausalLM"]
    # MTP is top-level in the raw weights. Explicitly enable it for Bridge's
    # provider as well, rather than merely copying tensors that would go unused.
    for field in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "mtp_num_layers"):
        text[field] = 1
    for field in ("bos_token_id", "eos_token_id", "torch_dtype", "dtype"):
        if field not in text and field in config:
            text[field] = config[field]
    # In VL configs the parent owns weight tying, even if text_config disagrees.
    if "tie_word_embeddings" in config:
        text["tie_word_embeddings"] = config["tie_word_embeddings"]
    return text


def _read_weight_map(source: Path) -> dict[str, str]:
    from safetensors import safe_open

    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Expected a nonempty safetensors weight_map.")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise ValueError("Safetensors index must map tensor names to shard filenames.")
        # HF snapshots use symlinks to ../blobs, so allow resolved symlink
        # targets outside the snapshot, but not paths embedded in the index.
        if "/" in shard or "\\" in shard or ":" in shard or not shard.endswith(".safetensors"):
            raise ValueError(f"Expected a plain safetensors shard filename: {shard}")
        if not (source / shard).is_file():
            raise FileNotFoundError(source / shard)
    return weight_map


def _copy_tokenizer_assets(source: Path, output: Path) -> None:
    if not (source / "tokenizer_config.json").is_file():
        raise FileNotFoundError(source / "tokenizer_config.json")
    if not any((source / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise FileNotFoundError(f"No tokenizer vocabulary found in {source}")
    assets = set()
    for pattern in (
        "tokenizer*",
        "chat_template*",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    ):
        assets.update(source.glob(pattern))
    for path in sorted(assets):
        target = output / path.name
        if path.is_dir():
            shutil.copytree(path, target)
        else:
            shutil.copy2(path, target)


def extract_checkpoint(*, source: Path, output: Path) -> None:
    """Copy decoder, LM head, MTP, and tokenizer assets into a new HF directory.

    Args:
        source: Local Qwen3.5 MoE VL safetensors snapshot directory.
        output: New directory outside the source; existing outputs are rejected.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("The output must be outside the input checkpoint directory.")

    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    weight_map = _read_weight_map(source)
    mapping = {name: text_weight_name(name) for name in weight_map}
    names = [name for name in mapping.values() if name is not None]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate tensor names after decoder prefix conversion.")
    text_config = text_model_config(config, names)
    source_shards = sorted(set(weight_map.values()))
    kept_shards = sorted({weight_map[name] for name, target in mapping.items() if target is not None})

    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every shard and metadata file is complete. An interrupted
    # copy must not leave an output directory the launcher could mistake for valid.
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        _copy_tokenizer_assets(source, staging)
        new_weight_map = {}
        total_size = 0
        shard_number = 0
        for shard in source_shards:
            expected = {name for name, filename in weight_map.items() if filename == shard}
            with safe_open(source / shard, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != expected:
                    raise ValueError(f"Safetensors index does not match the tensors in {shard}")
                tensors = {mapping[name]: handle.get_tensor(name) for name in sorted(expected) if mapping[name]}
            if not tensors:
                continue
            shard_number += 1
            filename = f"model-{shard_number:05d}-of-{len(kept_shards):05d}.safetensors"
            logger.info("Extracting %s -> %s (%d tensors)", shard, filename, len(tensors))
            save_file(tensors, staging / filename, metadata={"format": "pt"})
            new_weight_map.update(dict.fromkeys(tensors, filename))
            total_size += sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            del tensors
        index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        manifest = {
            "source": str(source),
            "architecture": text_config["architectures"][0],
            "mtp_num_layers": 1,
            "tensor_count": len(new_weight_map),
            "total_size": total_size,
        }
        for filename, content in (
            ("config.json", text_config),
            ("model.safetensors.index.json", index),
            ("text-extraction.json", manifest),
        ):
            (staging / filename).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    except Exception:
        logger.error("Extraction incomplete; temporary files retained at %s", staging)
        raise
    logger.info("Text checkpoint ready: %s (%d tensors, %d bytes)", output, len(new_weight_map), total_size)


def main() -> None:
    """Run the local extraction step used by the interactive conversion launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local HF Qwen3.5 MoE VL snapshot")
    parser.add_argument("--output", type=Path, required=True, help="New text-only HF checkpoint directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extract_checkpoint(source=args.source, output=args.output)


if __name__ == "__main__":
    main()
