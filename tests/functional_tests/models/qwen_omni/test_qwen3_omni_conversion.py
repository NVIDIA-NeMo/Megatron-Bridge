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

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from tests.functional_tests.models.qwen_omni.utils import (
    SMOKE_AUDIO_LAYERS,
    SMOKE_MODEL_CACHE_PATH,
    SMOKE_LOCK_DIR,
    SMOKE_TEXT_LAYERS,
    SMOKE_VISION_DEPTH,
    create_qwen3_omni_smoke_model,
    smoke_assets_available,
)


pytestmark = pytest.mark.skipif(not smoke_assets_available(), reason="Qwen3-Omni local smoke assets are unavailable")


class TestQwen3OmniConversion:
    @pytest.fixture(scope="class")
    def qwen3_omni_smoke_model_path(self, tmp_path_factory):
        del tmp_path_factory
        return str(create_qwen3_omni_smoke_model(SMOKE_MODEL_CACHE_PATH))

    def test_smoke_model_creation(self, qwen3_omni_smoke_model_path):
        model_path = Path(qwen3_omni_smoke_model_path)
        assert model_path.exists()
        assert (model_path / "config.json").exists()
        assert (model_path / "model.safetensors").exists()
        assert (model_path / "tokenizer_config.json").exists()
        assert (model_path / "preprocessor_config.json").exists()

        config = json.loads((model_path / "config.json").read_text())
        assert config["architectures"] == ["Qwen3OmniMoeForConditionalGeneration"]
        assert config["enable_audio_output"] is False
        assert config["thinker_config"]["text_config"]["num_hidden_layers"] == SMOKE_TEXT_LAYERS
        assert config["thinker_config"]["vision_config"]["depth"] == SMOKE_VISION_DEPTH
        assert config["thinker_config"]["audio_config"]["encoder_layers"] == SMOKE_AUDIO_LAYERS

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            qwen3_omni_smoke_model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        assert len(model.thinker.model.layers) == SMOKE_TEXT_LAYERS
        assert len(model.thinker.visual.blocks) == SMOKE_VISION_DEPTH
        assert len(model.thinker.audio_tower.layers) == SMOKE_AUDIO_LAYERS
        assert model.has_talker is False

    @pytest.mark.run_only_on("GPU")
    def test_qwen3_omni_conversion_single_gpu(self, qwen3_omni_smoke_model_path, tmp_path):
        output_dir = tmp_path / "qwen3_omni_single_gpu"
        output_dir.mkdir(exist_ok=True)

        repo_root = Path(__file__).parent.parent.parent.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        SMOKE_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        env["MEGATRON_CONFIG_LOCK_DIR"] = str(SMOKE_LOCK_DIR)
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--nnodes=1",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            qwen3_omni_smoke_model_path,
            "--output-dir",
            str(output_dir),
            "--tp",
            "1",
            "--pp",
            "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, env=env)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Qwen3-Omni single-GPU conversion failed with return code {result.returncode}"

        converted_model_dir = output_dir / Path(qwen3_omni_smoke_model_path).name
        assert converted_model_dir.exists()
        assert (converted_model_dir / "config.json").exists()
        assert (converted_model_dir / "model.safetensors").exists() or any(
            converted_model_dir.glob("model-*-of-*.safetensors")
        )
