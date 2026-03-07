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
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, dynamic_module_utils


def _fix_tied_weights_keys(model: nn.Module):
    """Convert _tied_weights_keys from list to dict for transformers 5.x compatibility."""
    for module in model.modules():
        tied = getattr(module, "_tied_weights_keys", None)
        if isinstance(tied, list):
            module._tied_weights_keys = {k: k for k in tied}


# Overrides for 8B size
HF_NEMOTRONH_TOY_MODEL_OVERRIDES = {
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
}


class TestNemotronHConversion:
    """
    Test NemotronH model conversion from local HuggingFace model with different parallelism configurations.
    """

    @pytest.fixture(scope="class")
    def nemotronh_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace NemotronH toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotronh_toy_model")
        model_dir = temp_dir / "nemotronh_toy_8b"

        # Create NemotronH toy model config by starting with 8B and applying overrides
        # This avoids attempting import of NemotronHConfig from Transformers
        config = AutoConfig.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        for k, v in HF_NEMOTRONH_TOY_MODEL_OVERRIDES.items():
            setattr(config, k, v)

        # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = dynamic_module_utils.get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path="nvidia/Nemotron-H-8B-Base-8K",
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id="nvidia/Nemotron-H-8B-Base-8K",
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # Download and save tokenizer from a reference NemotronH model
        tokenizer = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        _fix_tied_weights_keys(model)

        # Save model, config, and modeling code to directory
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    def test_toy_model_creation(self, nemotronh_toy_model_path):
        """
        Test that the toy model is created correctly and can be loaded.

        Args:
            nemotronh_toy_model_path: Path to the toy NemotronH model (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(nemotronh_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Check for modeling file
        nemotronh_modeling_file = model_path / "modeling_nemotron_h.py"
        assert nemotronh_modeling_file.exists(), (
            f"modeling_nemotron_h.py must be copied to toy model path. not found at {nemotronh_modeling_file}"
        )

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        assert config_data["model_type"] == "nemotron_h"
        assert config_data["hidden_size"] == 768
        assert config_data["intermediate_size"] == 3072
        assert config_data["num_hidden_layers"] == 4  # Updated to match toy config
        assert config_data["num_attention_heads"] == 16
        assert config_data["vocab_size"] == 131072

        # Try loading the model to verify it's valid
        try:
            model = AutoModelForCausalLM.from_pretrained(
                nemotronh_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
                trust_remote_code=True,
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = AutoTokenizer.from_pretrained(nemotronh_toy_model_path, trust_remote_code=True)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "backbone")
            assert hasattr(model.backbone, "layers")
            assert len(model.backbone.layers) == 4  # num_hidden_layers updated to match toy config

            print(f"SUCCESS: Toy model created and validated at {nemotronh_toy_model_path}")

        except Exception as e:
            assert False, f"Failed to load created toy model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_nemotronh_conversion_parallelism(self, nemotronh_toy_model_path, tmp_path, tp, pp, test_name):
        """
        Test NemotronH model conversion with different parallelism configurations.

        Args:
            nemotronh_toy_model_path: Path to the toy NemotronH model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"nemotronh_{test_name}"
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
            nemotronh_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent.parent.parent
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"NemotronH {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(nemotronh_toy_model_path).name  # "nemotronh_toy"
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            # Verify the config contains NemotronH-specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            assert saved_config["model_type"] == "nemotron_h", "Model type should be nemotron_h"
            assert saved_config["hidden_size"] == 768, "Hidden size should match toy config"
            assert saved_config["intermediate_size"] == 3072, "ffn hidden size should match toy config"
            assert saved_config["num_hidden_layers"] == 4, "Number of hidden layers should match toy config"
            assert saved_config["num_attention_heads"] == 16, "Number of attention heads should match toy config"

            print(f"SUCCESS: NemotronH {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during NemotronH {test_name} conversion test: {e}")
            raise


# Overrides for Nemotron-3-Nano MoE model (30B total, 3B active)
HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES = {
    "num_hidden_layers": 3,
    "hybrid_override_pattern": "M*E",
    "hidden_size": 672,
    "n_routed_experts": 16,
}


class TestNemotron3NanoConversion:
    """
    Test Nemotron-3-Nano MoE model conversion from local HuggingFace model with different parallelism configurations.

    This test class tests the nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 model which is a Mixture-of-Experts (MoE)
    variant of the NemotronH architecture.
    """

    _TOY_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    @pytest.fixture(scope="class")
    def nemotron_3_nano_toy_model_path(self, tmp_path_factory):
        """
        Create and save a HuggingFace Nemotron-3-Nano toy model from config to a temporary directory.

        Args:
            tmp_path_factory: Pytest temporary path factory for class-scoped fixtures

        Returns:
            str: Path to the saved HuggingFace model directory
        """
        # Create a temporary directory for this test class
        temp_dir = tmp_path_factory.mktemp("nemotron_3_nano_toy_model")
        model_dir = temp_dir / "nemotron_3_nano_toy"

        repo_id = self._TOY_MODEL_ID

        # Create Nemotron-3-Nano toy model config by starting with the HF model and applying overrides
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for k, v in HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES.items():
            setattr(config, k, v)

        # # Create model with random weights and convert to bfloat16
        model_class_ref = config.auto_map["AutoModelForCausalLM"]
        model_class = dynamic_module_utils.get_class_from_dynamic_module(
            class_reference=model_class_ref,
            pretrained_model_name_or_path=repo_id,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            proxies=None,
            use_auth_token=None,
            revision=None,
            local_files_only=False,
            repo_id=repo_id,
        )
        model = model_class(config)
        model = model.bfloat16() if hasattr(model, "bfloat16") else model

        # There is a bug in Nemotron Nano HF implementation that
        # TopKRouter weights are not initialized correctly, which leads to NaN values.
        # Reinitialize weights of all NemotronHTopkRouter modules if present
        for module in model.modules():
            if module.__class__.__name__ == "NemotronHTopkRouter":
                torch.nn.init.normal_(module.weight, mean=0.0, std=config.initializer_range)
                torch.nn.init.zeros_(module.e_score_correction_bias)

        for k, v in model.named_buffers():
            if "e_score_correction_bias" in k:
                v.data = v.data.to(torch.float32)

        # Download and save tokenizer from the reference Nemotron-3-Nano model
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        tokenizer.save_pretrained(model_dir)

        _fix_tied_weights_keys(model)

        # Save model, config, and modeling code to directory
        model.save_pretrained(model_dir, safe_serialization=True)
        modeling_filepath = os.path.abspath(sys.modules[model_class.__module__].__file__)
        shutil.copy(modeling_filepath, model_dir)

        # Ensure config.json exists with expected keys
        config_path = model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(model.config.to_dict(), f, indent=2)

        return str(model_dir)

    @pytest.fixture
    def temp_hf_modules(self, tmp_path, monkeypatch):
        """Change transformers.dynamic_module_utils.HF_MODULES_CACHE to a temp path"""
        temp_hf_modules_cache = tmp_path / "hf_modules_cache"
        temp_hf_modules_cache.mkdir(exist_ok=True)
        monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", temp_hf_modules_cache)
        yield temp_hf_modules_cache

    def test_toy_model_creation(self, nemotron_3_nano_toy_model_path, temp_hf_modules):
        """
        Test that the Nemotron-3-Nano toy model is created correctly and can be loaded.

        Args:
            nemotron_3_nano_toy_model_path: Path to the toy Nemotron-3-Nano model (from fixture)
            temp_hf_modules: Temporary HF modules cache path (from fixture)
        """
        # Verify the model directory exists
        model_path = Path(nemotron_3_nano_toy_model_path)
        assert model_path.exists(), f"Model directory not found at {model_path}"

        # Check essential files exist
        config_file = model_path / "config.json"
        assert config_file.exists(), f"config.json not found at {config_file}"

        # Check for model weights (safetensors preferred)
        weights_file = model_path / "model.safetensors"
        if not weights_file.exists():
            weights_file = model_path / "pytorch_model.bin"
        assert weights_file.exists(), f"Model weights file not found in {model_path}"

        # Check for tokenizer files
        tokenizer_config_file = model_path / "tokenizer_config.json"
        assert tokenizer_config_file.exists(), f"tokenizer_config.json not found at {tokenizer_config_file}"

        # Check for modeling file
        nemotronh_modeling_file = model_path / "modeling_nemotron_h.py"
        assert nemotronh_modeling_file.exists(), (
            f"modeling_nemotron_h.py must be copied to toy model path. not found at {nemotronh_modeling_file}"
        )

        # Load and verify config
        with open(config_file) as f:
            config_data = json.load(f)

        # Verify core model configuration
        assert config_data["model_type"] == "nemotron_h"
        for keys in HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES.keys():
            assert config_data[keys] == HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES[keys]

        # Try loading the model to verify it's valid
        try:
            model = AutoModelForCausalLM.from_pretrained(
                nemotron_3_nano_toy_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,  # Ensure full loading
                trust_remote_code=True,
                cache_dir=nemotron_3_nano_toy_model_path,
            )

            # Try loading the tokenizer as well
            try:
                tokenizer = AutoTokenizer.from_pretrained(nemotron_3_nano_toy_model_path, trust_remote_code=True)
                print(f"Tokenizer loaded successfully with vocab_size: {tokenizer.vocab_size}")
            except Exception as e:
                print(f"Warning: Could not load tokenizer (this might be OK for conversion testing): {e}")

            # Verify model structure
            assert hasattr(model, "backbone")
            assert hasattr(model.backbone, "layers")
            assert (
                len(model.backbone.layers) == HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES["num_hidden_layers"]
            )  # num_hidden_layers from toy config

            print(f"SUCCESS: Nemotron-3-Nano toy model created and validated at {nemotron_3_nano_toy_model_path}")

        except Exception as e:
            assert False, f"Failed to load created Nemotron-3-Nano toy model: {e}"

    @pytest.mark.run_only_on("GPU")
    @pytest.mark.parametrize(
        "tp,pp,test_name",
        [
            (2, 1, "TP"),
            (1, 2, "PP"),
        ],
    )
    def test_nemotron_3_nano_conversion_parallelism(
        self, nemotron_3_nano_toy_model_path, tmp_path, tp, pp, test_name, temp_hf_modules
    ):
        """
        Test Nemotron-3-Nano MoE model conversion with different parallelism configurations.

        Args:
            nemotron_3_nano_toy_model_path: Path to the toy Nemotron-3-Nano model (from fixture)
            tmp_path: Pytest temporary path fixture
            tp: Tensor parallelism size
            pp: Pipeline parallelism size
            test_name: Name of the test for identification
            temp_hf_modules: Temporary HF modules cache path (from fixture)
        """

        # Create temporary output directory for conversion results
        test_output_dir = tmp_path / f"nemotron_3_nano_{test_name}"
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
            nemotron_3_nano_toy_model_path,  # Use our local toy model instead of downloading
            "--output-dir",
            str(test_output_dir),
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--trust-remote-code",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent.parent.parent,
                env={**os.environ, "HF_MODULES_CACHE": str(temp_hf_modules)},
            )

            # Check that the conversion completed successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                assert False, f"Nemotron-3-Nano {test_name} conversion failed with return code {result.returncode}"

            # Verify that the converted model was saved
            # The output directory should be named after the last part of the model path
            model_name = Path(nemotron_3_nano_toy_model_path).name
            converted_model_dir = test_output_dir / model_name
            assert converted_model_dir.exists(), f"Converted model directory not found at {converted_model_dir}"

            # Check that essential model files exist
            config_file = converted_model_dir / "config.json"
            assert config_file.exists(), f"config.json not found in converted model at {config_file}"

            # Check for model weights file (could be either safetensors or pytorch_model.bin)
            weights_file_safetensors = converted_model_dir / "model.safetensors"
            weights_file_pytorch = converted_model_dir / "pytorch_model.bin"
            assert weights_file_safetensors.exists() or weights_file_pytorch.exists(), (
                f"Model weights file not found in converted model at {converted_model_dir}"
            )

            # Verify the config contains Nemotron-3-Nano specific parameters
            with open(config_file) as f:
                saved_config = json.load(f)

            # Verify core model configuration
            assert saved_config["model_type"] == "nemotron_h", "Model type should be nemotron_h"
            for keys in HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES.keys():
                assert saved_config[keys] == HF_NEMOTRON_3_NANO_TOY_MODEL_OVERRIDES[keys], (
                    f"{keys} should match toy config"
                )

            print(f"SUCCESS: Nemotron-3-Nano {test_name} conversion test completed successfully")
            print(f"Converted model saved at: {converted_model_dir}")

        except Exception as e:
            print(f"Error during Nemotron-3-Nano {test_name} conversion test: {e}")
            raise
