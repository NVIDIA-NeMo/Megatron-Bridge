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
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def _fix_tied_weights_keys(model: nn.Module):
    """Convert _tied_weights_keys from list to dict for transformers 5.x compatibility."""
    for module in model.modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, list):
            module._tied_weights_keys = {k: k for k in tied}


NEMOTRON_VL_HF_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

# Toy model overrides for smaller NemotronH language model
HF_NEMOTRON_VL_TOY_MODEL_OVERRIDES = {
    "attention_head_dim": 48,
    "chunk_size": 48,
    "expand": 2,
    "hidden_size": 768,
    "hybrid_override_pattern": "M*M-",
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": 64,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": 24,
    "max_position_embeddings": 8192,
    "n_groups": 8,
    "num_attention_heads": 16,
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "ssm_state_size": 128,
    "vocab_size": 131072,
    # Vision config overrides
    "vision_config": {
        "hidden_size": 256,
        "image_size": 384,
        "intermediate_size": 1024,
        "model_type": "radio_vision_model",
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "patch_size": 16,
    },
}


class TestNemotronVLConversion:
    """
    Test Nemotron VL model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def nemotron_vl_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Nemotron VL toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotron_vl_toy_model")
        model_dir = temp_dir / "nemotron_vl_toy"

        # Create Nemotron VL toy model config by starting with the full model and applying overrides
        config = AutoConfig.from_pretrained(NEMOTRON_VL_HF_ID, trust_remote_code=True)

        # Apply overrides, handling vision_config specially
        vision_config_overrides = None
        for k, v in HF_NEMOTRON_VL_TOY_MODEL_OVERRIDES.items():
            if k == "vision_config":
                # Save vision_config overrides to apply separately
                vision_config_overrides = v
            else:
                setattr(config, k, v)

        # Apply vision_config overrides to the existing vision_config object
        if vision_config_overrides and hasattr(config, "vision_config"):
            for k, v in vision_config_overrides.items():
                setattr(config.vision_config, k, v)

        # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=NEMOTRON_VL_HF_ID,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id=NEMOTRON_VL_HF_ID,
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Download and save tokenizer from the reference Nemotron VL model
        tokenizer = AutoTokenizer.from_pretrained(NEMOTRON_VL_HF_ID, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        # Save model, config, and modeling code to directory
        _fix_tied_weights_keys(model)
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, nemotron_vl_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            nemotron_vl_toy_model_path: Path to the toy Nemotron VL model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(nemotron_vl_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        assert list(model_path.glob("model*.safetensors")), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert "model_type" in config_data
        assert config_data["hidden_size"] == 768
        assert config_data["intermediate_size"] == 3072
        assert config_data["num_hidden_layers"] == 4
        assert config_data["num_attention_heads"] == 16
        assert config_data["vocab_size"] == 131072
        # Verify vision config exists
        assert "vision_config" in config_data or hasattr(config_data, "vision_config")

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (1, 1, "TP1"),
            (2, 1, "TP2"),
        ],
    )
    def test_nemotron_vl_conversion_parallelism(self, nemotron_vl_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test Nemotron VL model conversion with different parallelism configurations.

        Args:
            nemotron_vl_toy_model_path: Path to the toy Nemotron VL model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"nemotron_vl_{test_name}"
        test_output_dir.mkdir(exist_ok=True)

        # Run hf_megatron_roundtrip_multi_gpu.py with specified parallelism configuration on our toy model
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
            "--hf-model-id",
            nemotron_vl_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--trust-remote-code",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
        )
        print(cmd)

        # Check that the conversion completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            assert False, f"Nemotron VL {test_name} conversion failed with return code {result.returncode}"

        # Verify that the converted model was saved
        # The output directory should be named after the last part of the model path
        model_name = Path(nemotron_vl_toy_model_path).name  # "nemotron_vl_toy"
        converted_model_dir = test_output_dir / model_name
        assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

        # Check that essential model files exist
        config_file = converted_model_dir / "config.json"
        assert config_file.exists(), f"config.json not found in converted model at {config_file}"

        # Check for model weights file (could be either safetensors or pytorch_model.bin)
        assert list(converted_model_dir.glob("model*.safetensors")), (
            f"Model weights file not found in converted model at {converted_model_dir}"
        )

        # Verify the config contains expected parameters
        with open(config_file) as f:
            saved_config = json.load(f)

        assert "model_type" in saved_config, "Model type should be present in config"
        assert saved_config["hidden_size"] == 768, "Hidden size should match toy config"
        assert saved_config["num_attention_heads"] == 16, "Number of attention heads should match toy config"
        # Verify vision config exists
        assert "vision_config" in saved_config or hasattr(saved_config, "vision_config"), (
            "VL model should have vision_config"
        )
