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


from unittest.mock import patch

import pytest
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.utils import get_causal_lm_class_via_auto_map


class DummyConfig(PretrainedConfig):
    def __init__(self, auto_map=None, name_or_path=None, architectures=None):
        super().__init__()
        if auto_map is not None:
            self.auto_map = auto_map
        if name_or_path is not None:
            setattr(self, "_name_or_path", name_or_path)
        if architectures is not None:
            self.architectures = architectures


def test_returns_none_when_auto_map_absent():
    config = DummyConfig(auto_map=None)
    result = get_causal_lm_class_via_auto_map(model_name_or_path=None, config=config)
    assert result is None


def test_raises_when_auto_map_present_but_missing_repo_id():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "some.module:Class"}, name_or_path=None)
    with pytest.raises(ValueError, match="no repository identifier"):
        get_causal_lm_class_via_auto_map(model_name_or_path=None, config=config)


def test_bubbles_error_when_dynamic_loading_fails():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "some.module:Class"}, name_or_path="repo/id")
    # Force the dynamic loader to raise
    with patch(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            get_causal_lm_class_via_auto_map("repo/id", config)


def test_dynamic_loading_success():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "some.module:Class"}, name_or_path="repo/id")

    class DummyModel:
        pass

    with patch(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module",
        return_value=DummyModel,
    ):
        cls = get_causal_lm_class_via_auto_map("repo/id", config)
        assert cls is DummyModel
