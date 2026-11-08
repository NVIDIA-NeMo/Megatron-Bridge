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
Pre-tokenize text SFT JSONL for the ERNIE 5.5 recipes.

Reads chat rows (``{"messages": [...]}``) or prompt/completion rows (``{"input": ..., "output": ...}``), the formats
of ``tutorials/data/text-only-sft``, and writes ``{"input_ids": [...], "labels": [...]}`` rows for
``dataset.path`` of the ERNIE 5.5 recipes, encoded exactly as the dataset encodes text rows itself when
``dataset.hf_tokenizer`` is set (:mod:`megatron.bridge.models.ernie.ernie55_chat`). Pre-tokenize large corpora once
instead of encoding them on every rank at startup.

Example:

    python examples/models/ernie55/convert_sft_jsonl.py \\
        --input /data/sft/training.jsonl --output /data/sft/training.ernie55.jsonl \\
        --tokenizer /checkpoints/ernie55 --enable-thinking false

A directory ``--input`` converts every ``*.jsonl`` in it into the ``--output`` directory.
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def _load_ernie_module(name: str):
    """A module of ``megatron.bridge.models.ernie``, loaded from this checkout's source file.

    ``ernie55_chat`` and ``configuration_ernie55`` need only the standard library and ``transformers``, but importing
    them through the ``megatron.bridge`` package would import the whole training stack (Megatron-Core, Transformer
    Engine). Loading the files directly lets this script run on a CPU-only machine with just ``transformers``.
    """
    path = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "models" / "ernie" / f"{name}.py"
    if not path.is_file():
        return importlib.import_module(f"megatron.bridge.models.ernie.{name}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Registers model type ``ernie5_5_moe`` with AutoConfig: AutoTokenizer reads the checkpoint's config.json, and an
# unknown model type makes it fall back to a generic config that fails on ERNIE 5.5's rope_scaling.
_load_ernie_module("configuration_ernie55")
_chat = _load_ernie_module("ernie55_chat")
encode_ernie55_sft_row = _chat.encode_ernie55_sft_row
load_ernie55_chat_template = _chat.load_ernie55_chat_template


def convert_file(src: Path, dst: Path, *, tokenizer, template: str, args: argparse.Namespace) -> None:
    """Convert one JSONL file and log its token counts."""
    kwargs = {}
    for name in ("enable_thinking", "preserve_thinking"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value == "true"
    rows = tokens = supervised = empty = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                sample = encode_ernie55_sft_row(
                    tokenizer,
                    json.loads(line),
                    chat_template=template,
                    chat_template_kwargs=kwargs,
                    prompt_column=args.prompt_column,
                    completion_column=args.completion_column,
                    loss=args.loss,
                )
            except Exception as error:
                raise SystemExit(f"{src}:{lineno}: {error}") from error
            n = sum(label != -100 for label in sample["labels"])
            rows, tokens, supervised, empty = (
                rows + 1,
                tokens + len(sample["input_ids"]),
                supervised + n,
                empty + (n == 0),
            )
            fout.write(json.dumps(sample) + "\n")
    logger.info(f"{src} -> {dst}: {rows} rows, {tokens} tokens, {supervised} supervised, {empty} without a target")


def main() -> None:
    """Parse arguments and convert the input file(s)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="a .jsonl file or a directory of them")
    parser.add_argument("--output", type=Path, required=True, help="output .jsonl file, or directory")
    parser.add_argument("--tokenizer", required=True, help="Hugging Face tokenizer (ERNIE 5.5's own for real runs)")
    parser.add_argument("--chat-template", default=None, help="template file (default: the reference's)")
    parser.add_argument("--prompt-column", default="input")
    parser.add_argument("--completion-column", default="output")
    parser.add_argument("--enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--preserve-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--loss", choices=["assistant", "all"], default="assistant")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = load_ernie55_chat_template(args.chat_template)
    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        pairs = [(src, args.output / src.name) for src in sorted(args.input.glob("*.jsonl"))]
    else:
        pairs = [(args.input, args.output)]
    for src, dst in pairs:
        convert_file(src, dst, tokenizer=tokenizer, template=template, args=args)


if __name__ == "__main__":
    main()
