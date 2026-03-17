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

import io
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3OmniMoeProcessor


SOURCE_MODEL_PATH = Path("/nfs/volume-1615-2/models/Qwen3-Omni-30B-A3B-Instruct")
SOURCE_DATA_PATH = Path("/home/luban/liuwei/omni/ref/projects/Omni_Bench_fix_simple/test.parquet")
SMOKE_MODEL_CACHE_PATH = Path("/nfs/ofs-llab-hdd/users/liuwei/models/qwen3_omni_smoke")
SMOKE_LOCK_DIR = Path("/nfs/ofs-llab-hdd/users/liuwei/models/.hf_config_locks")

SMOKE_TEXT_LAYERS = 2
SMOKE_VISION_DEPTH = 2
SMOKE_AUDIO_LAYERS = 2
SMOKE_DEEPSTACK_INDEXES = [0]

_ARTIFACTS_TO_COPY = [
    "README.md",
    "chat_template.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
]

_LAYER_PATTERNS = [
    (re.compile(r"^thinker\.model\.layers\.(\d+)\."), SMOKE_TEXT_LAYERS),
    (re.compile(r"^thinker\.visual\.blocks\.(\d+)\."), SMOKE_VISION_DEPTH),
    (re.compile(r"^thinker\.audio_tower\.layers\.(\d+)\."), SMOKE_AUDIO_LAYERS),
]
_MERGER_PATTERN = re.compile(r"^thinker\.visual\.merger_list\.(\d+)\.")


def smoke_assets_available() -> bool:
    return SOURCE_MODEL_PATH.exists() and SOURCE_DATA_PATH.exists()


def create_qwen3_omni_smoke_model(output_dir: Path) -> Path:
    """Create a single-GPU Qwen3-Omni smoke model by vertically trimming thinker layers only."""
    if (output_dir / "config.json").exists() and (output_dir / "model.safetensors").exists():
        return output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    config = json.loads((SOURCE_MODEL_PATH / "config.json").read_text())
    config["enable_audio_output"] = False
    config["thinker_config"]["text_config"]["num_hidden_layers"] = SMOKE_TEXT_LAYERS
    config["thinker_config"]["vision_config"]["depth"] = SMOKE_VISION_DEPTH
    config["thinker_config"]["vision_config"]["deepstack_visual_indexes"] = SMOKE_DEEPSTACK_INDEXES
    config["thinker_config"]["audio_config"]["encoder_layers"] = SMOKE_AUDIO_LAYERS
    config["thinker_config"]["audio_config"]["num_hidden_layers"] = SMOKE_AUDIO_LAYERS
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    for artifact in _ARTIFACTS_TO_COPY:
        source = SOURCE_MODEL_PATH / artifact
        if source.exists():
            shutil.copy2(source, output_dir / artifact)

    index = json.loads((SOURCE_MODEL_PATH / "model.safetensors.index.json").read_text())
    selected_keys = [key for key in index["weight_map"] if _keep_smoke_weight(key)]
    by_file: dict[str, list[str]] = defaultdict(list)
    for key in selected_keys:
        by_file[index["weight_map"][key]].append(key)

    state_dict = {}
    for filename, keys in sorted(by_file.items()):
        with safe_open(SOURCE_MODEL_PATH / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                state_dict[key] = handle.get_tensor(key)

    temp_weights = output_dir / "model.safetensors.tmp"
    save_file(state_dict, str(temp_weights))
    temp_weights.replace(output_dir / "model.safetensors")
    return output_dir


def build_real_sample_inputs(model_path: str | Path) -> dict[str, torch.Tensor]:
    """Build one multimodal sample using the local OmniBench parquet and the model processor."""
    row = pd.read_parquet(SOURCE_DATA_PATH).iloc[0]
    image = Image.open(io.BytesIO(row["images"][0]["bytes"])).convert("RGB")
    audio = np.asarray(row["audios"][0], dtype=np.float32)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "audio"},
                {"type": "text", "text": "What is likely to happen next?"},
            ],
        }
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    return processor(text=text, images=[image], audio=[audio], return_tensors="pt")


def move_inputs_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if dtype is not None and key in {"pixel_values", "pixel_values_videos", "input_features"}:
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


def _keep_smoke_weight(key: str) -> bool:
    if not key.startswith("thinker."):
        return False

    merger_match = _MERGER_PATTERN.match(key)
    if merger_match is not None:
        return int(merger_match.group(1)) < len(SMOKE_DEEPSTACK_INDEXES)

    for pattern, limit in _LAYER_PATTERNS:
        match = pattern.match(key)
        if match is not None:
            return int(match.group(1)) < limit

    return True
