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

import json
import subprocess
from pathlib import Path

import pytest


NEMOTRON_VL_HF_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"


class TestNemotronVLConversion:
    """
    Test Nemotron VL model conversion from HuggingFace model with different parallelism configurations.
    """

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (1, 1, "single_gpu"),
            # (2, 1, "TP2"),
            # (1, 2, "PP2"),
        ],
    )
    def test_nemotron_vl_conversion_parallelism(self, tmp_path, tp, pp, test_name):
        """
        Test Nemotron VL model conversion with different parallelism configurations.

        Args:
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"nemotron_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration on the HF model
        cmd = [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--nnodes=1",
            "-m",
            "coverage",
            "run",
            "--data-file=/opt/Megatron-Bridge/.coverage",
            "--source=/opt/Megatron-Bridge/",
            "--parallel-mode",
            "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
            "--not-strict",
            "--hf-model-id",
            NEMOTRON_VL_HF_ID,
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent)
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Nemotron VL {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        model_name = NEMOTRON_VL_HF_ID.split("/")[-1]
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        # Check that essential model files exist
        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        # Check for sharded safetensor weights files
        for shard_idx in range(1, 8):
            shard_file = converted_model_dir / f"model-{shard_idx:05d}-of-00007.safetensors"
            assert shard_file.exists(), f"Missing shard file: {shard_file}"

        # Optionally, verify minimal config keys
        with open(config_file) as f:
            saved_config = json.load(f)
        assert "model_type" in saved_config
