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

"""
Unit tests for PreTrainedAdapters generic adapter loader with lazy loading.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from peft import LoraConfig

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.state import StateDict
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters


class TestPreTrainedAdapters:
    """Test cases for PreTrainedAdapters class."""

    @pytest.fixture
    def lora_config_dict(self):
        """Create a sample LoRA configuration."""
        return {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    def create_mock_adapter_directory(self, config_dict, save_dir):
        """Create a mock adapter directory with proper files."""
        save_path = Path(save_dir)

        # Create adapter_config.json
        config_path = save_path / "adapter_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create adapter_model.safetensors with dummy weights
        weights_path = save_path / "adapter_model.safetensors"
        dummy_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4096, 8),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(8, 4096),
            "base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(1024, 8),
        }
        import safetensors.torch

        safetensors.torch.save_file(dummy_weights, weights_path)

    def test_inheritance_from_base(self):
        """Test that PreTrainedAdapters properly inherits from PreTrainedBase."""
        assert issubclass(PreTrainedAdapters, PreTrainedBase)

        # Test artifacts configuration
        assert "config" in PreTrainedAdapters.ARTIFACTS
        assert isinstance(PreTrainedAdapters.OPTIONAL_ARTIFACTS, list)

    def test_initialization_basic(self):
        """Test basic initialization."""
        adapters = PreTrainedAdapters(
            model_name_or_path="username/test-adapters", trust_remote_code=False, strict=True
        )

        assert adapters.model_name_or_path == "username/test-adapters"
        assert adapters.trust_remote_code == False
        assert adapters.strict == True

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        adapters = PreTrainedAdapters()

        assert adapters.model_name_or_path is None
        assert adapters.trust_remote_code == False
        assert adapters.strict == True

    def test_from_pretrained_basic(self, lora_config_dict):
        """Test from_pretrained class method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters.from_pretrained(temp_dir)

            assert isinstance(adapters, PreTrainedAdapters)
            assert adapters.model_name_or_path == temp_dir
            assert adapters.trust_remote_code == False
            assert adapters.strict == True

    def test_from_pretrained_with_kwargs(self, lora_config_dict):
        """Test from_pretrained with custom kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters.from_pretrained(temp_dir, trust_remote_code=True, strict=False)

            assert adapters.trust_remote_code == True
            assert adapters.strict == False

    def test_load_config_lazy(self, lora_config_dict):
        """Test lazy loading of adapter configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters(model_name_or_path=temp_dir)

            # Config should not be loaded yet
            assert not hasattr(adapters, "_config")

            # Access config - should trigger loading
            config = adapters.config
            assert isinstance(config, LoraConfig)
            assert config.peft_type.value == "LORA"
            assert config.r == lora_config_dict["r"]

    def test_load_config_missing_file(self):
        """Test config loading with missing adapter_config.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create any files
            adapters = PreTrainedAdapters(model_name_or_path=temp_dir)

            with pytest.raises(FileNotFoundError, match="No adapter_config.json found"):
                _ = adapters.config

    def test_load_config_missing_peft_type(self):
        """Test config loading with missing peft_type field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config without peft_type
            invalid_config = {"r": 8}
            config_path = Path(temp_dir) / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(invalid_config, f)

            adapters = PreTrainedAdapters(model_name_or_path=temp_dir)

            with pytest.raises(ValueError, match="must contain 'peft_type' field"):
                _ = adapters.config

    def test_load_state_lazy(self, lora_config_dict):
        """Test lazy loading of adapter state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters(model_name_or_path=temp_dir)

            # State should not be loaded yet
            assert not hasattr(adapters, "_state_dict_accessor")

            # Access state - should trigger loading
            state = adapters.state
            assert isinstance(state, StateDict)

    def test_load_state_strict_validation(self, lora_config_dict):
        """Test state loading with strict validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            # Test with strict=True (default)
            adapters = PreTrainedAdapters(model_name_or_path=temp_dir, strict=True)
            state = adapters.state  # Should work with valid adapter weights
            assert isinstance(state, StateDict)

    def test_load_state_no_adapter_weights(self):
        """Test state loading with no recognizable adapter weights."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config but no proper adapter weights
            config_dict = {"peft_type": "LORA", "r": 8, "lora_alpha": 16}
            config_path = Path(temp_dir) / "adapter_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            # Create safetensors with non-adapter weights
            weights_path = Path(temp_dir) / "adapter_model.safetensors"
            dummy_weights = {"model.embed_tokens.weight": torch.randn(1000, 512)}
            import safetensors.torch

            safetensors.torch.save_file(dummy_weights, weights_path)

            adapters = PreTrainedAdapters(model_name_or_path=temp_dir, strict=True)

            with pytest.raises(ValueError, match="No recognizable PEFT adapter keys found"):
                _ = adapters.state

    def test_get_helper_methods(self, lora_config_dict):
        """Test convenience getter methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters.from_pretrained(temp_dir)

            # Test getter methods
            assert adapters.get_peft_type() == "LORA"
            assert adapters.get_rank() == lora_config_dict["r"]
            assert adapters.get_alpha() == lora_config_dict["lora_alpha"]
            assert adapters.get_dropout() == lora_config_dict["lora_dropout"]
            assert adapters.get_target_modules() == lora_config_dict["target_modules"]

    def test_supports_layout(self, lora_config_dict):
        """Test layout support detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters.from_pretrained(temp_dir)

            # Should support canonical layout (has q_proj, k_proj, etc.)
            assert adapters.supports_layout("canonical") == True
            assert adapters.supports_layout("fused") == False
            assert adapters.supports_layout("unknown") == False

    def test_resolve_path_local(self):
        """Test path resolution for local directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            resolved = PreTrainedAdapters._resolve_path(path)
            assert resolved == path

    def test_resolve_path_hub_download(self):
        """Test path resolution with HuggingFace Hub download."""
        with patch("megatron.bridge.peft.conversion.pretrained_adapters.snapshot_download") as mock_download:
            mock_download.return_value = "/cache/path/to/adapters"

            resolved = PreTrainedAdapters._resolve_path("username/repo-name")
            assert resolved == Path("/cache/path/to/adapters")
            mock_download.assert_called_once()

    def test_resolve_path_hub_download_failure(self):
        """Test path resolution when Hub download fails."""
        with patch("megatron.bridge.peft.conversion.pretrained_adapters.snapshot_download") as mock_download:
            mock_download.side_effect = Exception("Download failed")

            with pytest.raises(ValueError, match="Could not resolve path"):
                PreTrainedAdapters._resolve_path("invalid/repo")

    def test_repr(self, lora_config_dict):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_mock_adapter_directory(lora_config_dict, temp_dir)

            adapters = PreTrainedAdapters.from_pretrained(temp_dir)
            repr_str = repr(adapters)

            assert "PreTrainedAdapters(" in repr_str
            assert "(config):" in repr_str
            assert "(state):" in repr_str
            assert "(path):" in repr_str

    def test_repr_not_loaded(self):
        """Test repr when components are not loaded."""
        adapters = PreTrainedAdapters()
        repr_str = repr(adapters)

        assert "PreTrainedAdapters(" in repr_str
        assert "not loaded" in repr_str
