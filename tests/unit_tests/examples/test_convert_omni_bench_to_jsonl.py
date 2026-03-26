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

"""Unit tests for Qwen3-Omni Omni Bench parquet conversion."""

from __future__ import annotations

import importlib.util
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "models"
        / "vlm"
        / "qwen3_omni"
        / "convert_omni_bench_to_jsonl.py"
    )
    spec = importlib.util.spec_from_file_location("convert_omni_bench_to_jsonl", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_dataframe_writes_jsonl_and_media(tmp_path):
    module = _load_module()

    image = Image.new("RGB", (8, 8), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    df = pd.DataFrame(
        [
            {
                "data_source": "unit-test",
                "prompt": np.array([{"role": "user", "content": "<image> <audio> What happens next?"}], dtype=object),
                "images": np.array([{"bytes": image_bytes, "path": None}], dtype=object),
                "audios": np.array([np.linspace(-0.5, 0.5, 1600, dtype=np.float32)], dtype=object),
                "ability": "multimodal",
                "reward_model": {"ground_truth": "A person speaks.", "style": "rule"},
                "extra_info": {"answer": "A person speaks.", "index": 7, "question": "<image> <audio> What happens next?"},
            }
        ]
    )

    jsonl_path, num_written = module.convert_dataframe(df, "train", tmp_path, audio_sample_rate=16000)

    assert num_written == 1
    assert jsonl_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["conversation"][0]["role"] == "user"
    assert payload["conversation"][1]["content"][0]["text"] == "A person speaks."

    user_content = payload["conversation"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert user_content[1]["type"] == "audio"
    assert user_content[2]["type"] == "text"
    assert Path(user_content[0]["image"]).exists()
    assert Path(user_content[1]["audio"]).exists()


def test_strip_placeholders_normalizes_spaces():
    module = _load_module()
    assert module.strip_placeholders("<image>  <audio> Hello   world") == "Hello world"
