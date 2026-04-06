#!/usr/bin/env python3
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

"""Create a thinker-only Qwen2.5-Omni smoke checkpoint for local single-GPU validation."""

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


ARTIFACTS_TO_COPY = [
    "chat_template.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
]


def keep_weight(
    key: str,
    text_layers: int,
    vision_depth: int,
    audio_layers: int,
) -> bool:
    """Return whether a tensor should be kept in the reduced thinker-only checkpoint."""
    if not key.startswith("thinker."):
        return False

    layer_patterns = [
        (re.compile(r"^thinker\.model\.layers\.(\d+)\."), text_layers),
        (re.compile(r"^thinker\.visual\.blocks\.(\d+)\."), vision_depth),
        (re.compile(r"^thinker\.audio_tower\.layers\.(\d+)\."), audio_layers),
    ]
    for pattern, limit in layer_patterns:
        match = pattern.match(key)
        if match is not None:
            return int(match.group(1)) < limit

    return True


def create_smoke_checkpoint(
    source_model_path: Path,
    output_dir: Path,
    text_layers: int,
    vision_depth: int,
    audio_layers: int,
) -> Path:
    """Materialize a reduced thinker-only checkpoint while preserving hidden dimensions."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    config = json.loads((source_model_path / "config.json").read_text())
    config["enable_audio_output"] = False
    config["thinker_config"]["text_config"]["num_hidden_layers"] = text_layers
    config["thinker_config"]["vision_config"]["depth"] = vision_depth
    config["thinker_config"]["vision_config"]["fullatt_block_indexes"] = list(range(min(vision_depth, 4)))
    ac = config["thinker_config"]["audio_config"]
    ac["encoder_layers"] = audio_layers
    if "num_hidden_layers" in ac:
        ac["num_hidden_layers"] = audio_layers
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    for artifact in ARTIFACTS_TO_COPY:
        source = source_model_path / artifact
        if source.exists():
            shutil.copy2(source, output_dir / artifact)

    index = json.loads((source_model_path / "model.safetensors.index.json").read_text())
    selected_keys = [
        key
        for key in index["weight_map"]
        if keep_weight(
            key,
            text_layers=text_layers,
            vision_depth=vision_depth,
            audio_layers=audio_layers,
        )
    ]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key in selected_keys:
        by_file[index["weight_map"][key]].append(key)

    state_dict = {}
    for filename, keys in sorted(by_file.items()):
        with safe_open(source_model_path / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key] = handle.get_tensor(key)

    save_file(state_dict, str(output_dir / "model.safetensors"))
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for smoke checkpoint creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model-path",
        type=Path,
        required=True,
        help="Path to the full Hugging Face Qwen2.5-Omni checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the reduced thinker-only smoke checkpoint will be written.",
    )
    parser.add_argument("--text-layers", type=int, default=2, help="Number of thinker text layers to keep.")
    parser.add_argument("--vision-depth", type=int, default=2, help="Number of thinker vision blocks to keep.")
    parser.add_argument("--audio-layers", type=int, default=2, help="Number of thinker audio layers to keep.")
    return parser.parse_args()


def main() -> None:
    """Create the reduced checkpoint and print the output location."""
    args = parse_args()
    output_dir = create_smoke_checkpoint(
        source_model_path=args.source_model_path,
        output_dir=args.output_dir,
        text_layers=args.text_layers,
        vision_depth=args.vision_depth,
        audio_layers=args.audio_layers,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
