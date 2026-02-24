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
Unit tests for AutoBridge integration/validation for Sarvam architectures.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from transformers import PretrainedConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestAutoBridgeSarvamConfigValidation:
    def test_autobridge_supports_sarvam_architectures(self):
        cfg = Mock()
        cfg.architectures = ["SarvamMoEForCausalLM"]
        assert AutoBridge.supports(cfg) is True

        cfg.architectures = ["SarvamMLAForCausalLM"]
        assert AutoBridge.supports(cfg) is True

    def test_autobridge_from_hf_config_accepts_registered_sarvam_arch(self):
        # from_hf_config validates only architecture + implementation registration.
        config = PretrainedConfig(
            architectures=["SarvamMoEForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_sarvam.SarvamMoEForCausalLM"},
        )
        bridge = AutoBridge.from_hf_config(config)
        assert isinstance(bridge, AutoBridge)

    def _write_minimal_model_dir(self, config_dict, save_dir: str) -> None:
        config_path = Path(save_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # minimal safetensors index to resemble HF layout (not used directly in these unit tests)
        index_path = Path(save_dir) / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": {}}, f, indent=2)

    @patch(
        "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
    )
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_with_temp_dir(
        self, mock_safe_load_cfg, mock_from_pretrained
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_dict = {
                "architectures": ["SarvamMoEForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_sarvam.SarvamMoEForCausalLM"
                },
            }
            self._write_minimal_model_dir(cfg_dict, temp_dir)

            config = PretrainedConfig(**cfg_dict)
            mock_safe_load_cfg.return_value = config

            # Must be a real PreTrainedCausalLM instance to satisfy AutoBridge's isinstance checks.
            # Note: we cannot call PreTrainedCausalLM.from_pretrained here because it is patched in this test.
            hf_model = PreTrainedCausalLM(model_name_or_path=temp_dir)
            hf_model.config = config
            mock_from_pretrained.return_value = hf_model

            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == hf_model

            mock_safe_load_cfg.assert_called_once_with(
                temp_dir, trust_remote_code=False
            )
            mock_from_pretrained.assert_called_once_with(temp_dir)

    @patch(
        "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
    )
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_with_temp_dir_mla(
        self, mock_safe_load_cfg, mock_from_pretrained
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_dict = {
                "architectures": ["SarvamMLAForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_sarvam.SarvamMLAForCausalLM"
                },
            }
            self._write_minimal_model_dir(cfg_dict, temp_dir)

            config = PretrainedConfig(**cfg_dict)
            mock_safe_load_cfg.return_value = config

            hf_model = PreTrainedCausalLM(model_name_or_path=temp_dir)
            hf_model.config = config
            mock_from_pretrained.return_value = hf_model

            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == hf_model

            mock_safe_load_cfg.assert_called_once_with(
                temp_dir, trust_remote_code=False
            )
            mock_from_pretrained.assert_called_once_with(temp_dir)

    @patch(
        "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
    )
    @patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry")
    def test_from_pretrained_to_megatron_provider_calls_bridge(
        self, mock_safe_load_cfg, mock_from_pretrained
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_dict = {
                "architectures": ["SarvamMoEForCausalLM"],
                "auto_map": {
                    "AutoModelForCausalLM": "modeling_sarvam.SarvamMoEForCausalLM"
                },
            }
            self._write_minimal_model_dir(cfg_dict, temp_dir)

            config = PretrainedConfig(**cfg_dict)
            mock_safe_load_cfg.return_value = config

            hf_model = PreTrainedCausalLM(model_name_or_path=temp_dir)
            hf_model.config = config
            mock_from_pretrained.return_value = hf_model

            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            with patch(
                "megatron.bridge.models.conversion.auto_bridge.model_bridge.get_model_bridge"
            ) as mock_get_bridge:
                mock_bridge = Mock()
                mock_provider = Mock()
                mock_bridge.provider_bridge.return_value = mock_provider
                mock_get_bridge.return_value = mock_bridge

                provider = bridge.to_megatron_provider(load_weights=False)
                assert provider == mock_provider
                mock_bridge.provider_bridge.assert_called_once_with(hf_model)
