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
import sys
from pathlib import Path

import pytest


_DEFAULT_HF_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"


class TestNemotronOmniConversion:
    @pytest.mark.run_only_on("GPU")
    def test_nemotron_omni_conversion_roundtrip(self, tmp_path):
        hf_model_id = os.environ.get("NEMOTRON_OMNI_HF_MODEL") or _DEFAULT_HF_ID

        output_dir = tmp_path / "nemotron_omni_roundtrip"
        output_dir.mkdir(exist_ok=True)

        tp = os.environ.get("NEMOTRON_OMNI_CONVERSION_TP", "2")
        pp = os.environ.get("NEMOTRON_OMNI_CONVERSION_PP", "1")
        nproc = os.environ.get("NEMOTRON_OMNI_CONVERSION_GPUS", tp)

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            nproc,
            "--nnodes",
            "1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--hf-model-id",
            hf_model_id,
            "--output-dir",
            str(output_dir),
            "--tp",
            tp,
            "--pp",
            pp,
            "--trust-remote-code",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[5],
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Nemotron Omni conversion failed with return code {result.returncode}"

        model_name = Path(hf_model_id).name
        converted_model_dir = output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"
        assert list(converted_model_dir.glob("model*.safetensors")), (
            f"Model weights file not found in converted model at {converted_model_dir}"
        )

        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config["architectures"][0] == "NemotronH_Nano_Omni_Reasoning_V3"
        assert saved_config["model_type"] == "NemotronH_Nano_Omni_Reasoning_V3"
        assert "llm_config" in saved_config
        assert "vision_config" in saved_config
        assert "sound_config" in saved_config
