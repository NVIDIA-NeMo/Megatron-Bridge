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
Create a patched copy of the Kimi K2.5 VL model directory for newer transformers.

The original Kimi K2.5 custom code targets an older transformers version. This
script creates a thin wrapper directory containing:
  - Patched Python files (compatible with transformers >= 4.50)
  - Symlinks to the original safetensors weights, tokenizer files, etc.

Usage:
    python patch_kimi_k25_compat.py \
        --source /path/to/kimi-k2.5-test-weights_vv1 \
        --output /path/to/kimi-k25-vl-patched
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


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

SYMLINK_EXTENSIONS = {".safetensors", ".bin", ".pt", ".json", ".jinja", ".model", ".txt", ".tiktoken"}
PATCH_EXTENSIONS = {".py"}


def _patch_text(text: str) -> str:
    """Apply all transformers compatibility patches."""
    text = text.replace(
        "from transformers.utils.import_utils import is_torch_fx_available",
        "is_torch_fx_available = lambda: False",
    )
    text = text.replace(
        "from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode",
        _BYTES_TO_UNICODE_FALLBACK,
    )
    text = re.sub(r"def tie_weights\(self\):", "def tie_weights(self, **kwargs):", text)
    text = text.replace(
        "past_key_values = DynamicCache.from_legacy_cache(past_key_values)",
        "past_key_values = DynamicCache() if past_key_values is None else past_key_values",
    )
    text = re.sub(
        r"next_decoder_cache\.to_legacy_cache\(\)\s*\n\s*if use_legacy_cache\s*\n\s*else next_decoder_cache",
        "next_decoder_cache",
        text,
    )
    return text


def _resolve_source(source_str: str) -> Path:
    """Resolve source to a local directory, downloading from HF Hub if needed."""
    source_path = Path(source_str)
    if source_path.is_dir():
        return source_path.resolve()

    # Treat as a HuggingFace Hub model ID (e.g. "moonshotai/Kimi-K2.5")
    print(f"Source is not a local directory, downloading from HuggingFace: {source_str}")
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(source_str, local_dir_use_symlinks=True)
    return Path(local_dir).resolve()


def main() -> None:
    """Patch Kimi K2.5 model files for newer transformers compatibility."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Original Kimi K2.5 model directory or HuggingFace model ID")
    parser.add_argument("--output", required=True, help="Patched output directory")
    args = parser.parse_args()

    source = _resolve_source(args.source)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    patched_count = 0
    linked_count = 0

    for f in source.iterdir():
        if f.name.startswith("."):
            continue
        if not f.is_file():
            continue

        ext = f.suffix.lower()
        dest = output / f.name

        if ext in PATCH_EXTENSIONS:
            text = f.read_text()
            patched = _patch_text(text)
            dest.write_text(patched)
            if patched != text:
                patched_count += 1
                print(f"  Patched: {f.name}")
            else:
                print(f"  Copied (no changes): {f.name}")
        else:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(str(f), str(dest))
            linked_count += 1

    print(f"\nDone: {patched_count} files patched, {linked_count} files symlinked")
    print(f"Patched model at: {output}")

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = Path(hf_home) / "modules" / "transformers_modules" / output.name
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared HF module cache: {cache_dir}")


if __name__ == "__main__":
    main()
