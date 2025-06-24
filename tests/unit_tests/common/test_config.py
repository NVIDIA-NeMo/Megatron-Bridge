# Copyright (c) 2025, NVIDIA CORPORATION.
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

"""Tests for megatron.hub.common.config module."""

import json
from dataclasses import dataclass

import pytest
import yaml

from megatron.hub.common.config import (
    ConfigProtocol,
    from_pretrained,
    save_pretrained,
)


@dataclass
class MockConfig:
    """Mock configuration class for testing."""

    model_type: str = "mock"
    hidden_size: int = 768
    num_layers: int = 12

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Mock from_pretrained implementation."""
        return from_pretrained(cls, pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """Mock save_pretrained implementation."""
        save_pretrained(self, save_directory, **kwargs)


class TestConfigProtocol:
    """Tests for ConfigProtocol."""

    def test_protocol_implementation(self):
        """Test that MockConfig implements ConfigProtocol."""
        assert isinstance(MockConfig, ConfigProtocol)
        assert hasattr(MockConfig, "from_pretrained")
        assert hasattr(MockConfig, "save_pretrained")


class TestFromPretrained:
    """Tests for from_pretrained function."""

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {"model_type": "test", "hidden_size": 1024, "num_layers": 24}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_pretrained(MockConfig, str(config_file))
        assert config.model_type == "test"
        assert config.hidden_size == 1024
        assert config.num_layers == 24

    def test_load_from_json_file(self, tmp_path):
        """Test loading configuration from JSON file."""
        config_data = {"model_type": "test_json", "hidden_size": 512, "num_layers": 6}

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        config = from_pretrained(MockConfig, str(config_file))
        assert config.model_type == "test_json"
        assert config.hidden_size == 512
        assert config.num_layers == 6

    def test_load_from_directory(self, tmp_path):
        """Test loading configuration from directory containing config file."""
        config_data = {"model_type": "dir_test", "hidden_size": 256, "num_layers": 4}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_pretrained(MockConfig, str(tmp_path))
        assert config.model_type == "dir_test"
        assert config.hidden_size == 256
        assert config.num_layers == 4

    def test_kwargs_override(self, tmp_path):
        """Test that kwargs override loaded configuration."""
        config_data = {"model_type": "original", "hidden_size": 768, "num_layers": 12}

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_pretrained(MockConfig, str(config_file), hidden_size=2048, num_layers=48)
        assert config.model_type == "original"  # Not overridden
        assert config.hidden_size == 2048  # Overridden
        assert config.num_layers == 48  # Overridden

    def test_custom_config_name(self, tmp_path):
        """Test loading with custom config name."""
        config_data = {"model_type": "custom", "hidden_size": 1024}

        config_file = tmp_path / "my_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = from_pretrained(MockConfig, str(tmp_path), config_name="my_config")
        assert config.model_type == "custom"
        assert config.hidden_size == 1024

    def test_file_not_found(self, tmp_path):
        """Test error when config file not found."""
        with pytest.raises(FileNotFoundError):
            from_pretrained(MockConfig, str(tmp_path / "nonexistent"))


class TestSavePretrained:
    """Tests for save_pretrained function."""

    def test_save_to_yaml(self, tmp_path):
        """Test saving configuration to YAML format."""
        config = MockConfig(model_type="save_test", hidden_size=2048, num_layers=32)

        save_pretrained(config, str(tmp_path), config_format="yaml")

        # Verify saved file
        saved_file = tmp_path / "config.yaml"
        assert saved_file.exists()

        with open(saved_file) as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data["model_type"] == "save_test"
        assert loaded_data["hidden_size"] == 2048
        assert loaded_data["num_layers"] == 32

    def test_save_to_json(self, tmp_path):
        """Test saving configuration to JSON format."""
        config = MockConfig(model_type="json_save", hidden_size=1024)

        save_pretrained(config, str(tmp_path), config_format="json")

        saved_file = tmp_path / "config.json"
        assert saved_file.exists()

        with open(saved_file) as f:
            loaded_data = json.load(f)

        assert loaded_data["model_type"] == "json_save"
        assert loaded_data["hidden_size"] == 1024

    def test_save_with_custom_name(self, tmp_path):
        """Test saving with custom config name."""
        config = MockConfig()

        save_pretrained(config, str(tmp_path), config_name="my_model_config")

        saved_file = tmp_path / "my_model_config.json"
        assert saved_file.exists()

    def test_create_directory(self, tmp_path):
        """Test that save_pretrained creates directory if it doesn't exist."""
        config = MockConfig()
        save_dir = tmp_path / "new_dir"

        save_pretrained(config, str(save_dir))

        assert save_dir.exists()
        assert (save_dir / "config.json").exists()
