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

"""
Create a toy-sized Kimi K2.5 VL model for compare.py verification.

The full model has 61 layers, 384 routed experts, 27 vision layers, and
INT4/FP8 quantization. This script produces a tiny bf16 model with 2 layers,
4 experts, and 4 vision layers for rapid logits-match testing.

Usage:
    # From an existing local checkpoint (e.g. kimi-k2.5-small config dir):
    uv run python examples/models/vlm/kimi_k25/create_toy_model.py \
        --source-dir /path/to/kimi-k2.5-small \
        --output-dir /tmp/kimi_k25_toy

    # Then run compare.py:
    uv run python examples/conversion/compare_hf_and_megatron/compare.py \
        --hf_model_path /tmp/kimi_k25_toy \
        --prompt "Hello world" \
        --trust_remote_code
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch


_BYTES_TO_UNICODE_FALLBACK = """
try:
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
except ImportError:
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~")+1)) + list(range(0xA1, 0xAC+1)) + list(range(0xAE, 0xFF+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        return dict(zip(bs, (chr(c) for c in cs)))
""".strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a toy Kimi K2.5 VL checkpoint.")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to a Kimi K2.5 config directory (must contain config.json and *.py files).",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save the toy model.")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="LLM decoder layers.")
    parser.add_argument("--num-experts", type=int, default=4, help="Routed MoE experts.")
    parser.add_argument("--num-experts-per-tok", type=int, default=2, help="Experts per token.")
    parser.add_argument("--vt-num-hidden-layers", type=int, default=4, help="Vision tower layers.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _patch_compat(output: Path) -> None:
    """Fix transformers compatibility issues in all copied Python files."""
    for py_file in output.glob("*.py"):
        text = py_file.read_text()
        original = text

        # is_torch_fx_available removed in newer transformers
        text = text.replace(
            "from transformers.utils.import_utils import is_torch_fx_available",
            "is_torch_fx_available = lambda: False",
        )

        # bytes_to_unicode moved/removed in newer transformers
        text = text.replace(
            "from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode",
            _BYTES_TO_UNICODE_FALLBACK,
        )

        # tie_weights() signature changed to accept **kwargs
        text = re.sub(r"def tie_weights\(self\):", "def tie_weights(self, **kwargs):", text)

        # DynamicCache legacy cache API removed in newer transformers
        text = text.replace(
            "past_key_values = DynamicCache.from_legacy_cache(past_key_values)",
            "past_key_values = DynamicCache() if past_key_values is None else past_key_values",
        )
        text = re.sub(
            r"next_decoder_cache\.to_legacy_cache\(\)\s*\n\s*if use_legacy_cache\s*\n\s*else next_decoder_cache",
            "next_decoder_cache",
            text,
        )

        if text != original:
            py_file.write_text(text)
            print(f"  Patched: {py_file.name}")


def _modify_config(config: dict, args: argparse.Namespace) -> dict:
    """Modify config dict in-place to create a toy model."""
    tc = config.get("text_config", config)

    tc.pop("quantization_config", None)

    tc["num_hidden_layers"] = args.num_hidden_layers
    tc["n_routed_experts"] = args.num_experts
    tc["num_experts_per_tok"] = args.num_experts_per_tok
    tc["first_k_dense_replace"] = min(tc.get("first_k_dense_replace", 1), args.num_hidden_layers)
    if "max_window_layers" in tc:
        tc["max_window_layers"] = min(tc["max_window_layers"], args.num_hidden_layers)

    vc = config.get("vision_config", {})
    if isinstance(vc, dict):
        vc["vt_num_hidden_layers"] = args.vt_num_hidden_layers

    config["torch_dtype"] = "bfloat16"
    return config


def main() -> None:
    """Create a toy Kimi K2.5 VL model from a full-size source directory."""
    args = _parse_args()
    source = Path(args.source_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for f in source.iterdir():
        if f.is_file():
            shutil.copy2(f, output / f.name)

    print("Patching files for transformers compatibility...")
    _patch_compat(output)

    config_path = output / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    config = _modify_config(config, args)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved modified config to {config_path}")

    torch.manual_seed(args.seed)

    # Clear cached modules so transformers re-copies our patched files
    import os

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "modules" / "transformers_modules" / output.name
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cached modules: {cache_dir}")

    from transformers import AutoConfig, AutoModelForCausalLM

    loaded_config = AutoConfig.from_pretrained(str(output), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(loaded_config, trust_remote_code=True)
    model = model.bfloat16()

    from safetensors.torch import save_file

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(output / "model.safetensors"))

    weight_map = {k: "model.safetensors" for k in state_dict}
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in state_dict.values())},
        "weight_map": weight_map,
    }
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nToy Kimi K2.5 VL saved to: {output}")
    print(f"  LLM layers: {args.num_hidden_layers}")
    print(f"  Routed experts: {args.num_experts}")
    print(f"  Experts/token: {args.num_experts_per_tok}")
    print(f"  Vision layers: {args.vt_num_hidden_layers}")
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
