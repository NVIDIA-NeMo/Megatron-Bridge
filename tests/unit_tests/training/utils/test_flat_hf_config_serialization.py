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

from dataclasses import dataclass

import pytest
import yaml
from transformers import PreTrainedConfig

from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
from megatron.bridge.utils.instantiate_utils import InstantiationMode, register_allowed_target_prefix


pytestmark = pytest.mark.unit

register_allowed_target_prefix(f"{__name__}.")


class FlatRemoteHFConfig(PreTrainedConfig):
    """Minimal stand-in for a trust-remote-code config with dynamic fields."""

    def __init__(
        self,
        hidden_size=768,
        vision_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vision_config = vision_config or {"hidden_size": 3584, "attn_sep": False}


@dataclass
class FlatRemoteHFConfigContainer(_ConfigContainerBase):
    hf_config: FlatRemoteHFConfig


def test_flat_remote_hf_config_yaml_roundtrip_preserves_dynamic_fields(tmp_path):
    config = FlatRemoteHFConfigContainer(
        hf_config=FlatRemoteHFConfig(
            hidden_size=2560,
            vision_config={"hidden_size": 1280, "attn_sep": True},
        )
    )
    config.hf_config.text_config = config.hf_config
    yaml_path = tmp_path / "run_config.yaml"
    config.to_yaml(str(yaml_path))

    serialized = yaml.safe_load(yaml_path.read_text())
    assert serialized["hf_config"]["hidden_size"] == 2560
    assert serialized["hf_config"]["vision_config"] == {"hidden_size": 1280, "attn_sep": True}
    assert "text_config" not in serialized["hf_config"]

    loaded = FlatRemoteHFConfigContainer.from_yaml(str(yaml_path), mode=InstantiationMode.STRICT)
    assert isinstance(loaded.hf_config, FlatRemoteHFConfig)
    assert loaded.hf_config.hidden_size == 2560
    assert loaded.hf_config.vision_config == {"hidden_size": 1280, "attn_sep": True}

    loaded.hf_config.text_config = loaded.hf_config
    assert loaded.hf_config.text_config.hidden_size == 2560
