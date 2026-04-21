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


from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.utils import get_causal_lm_class_name_via_auto_map


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
    result = get_causal_lm_class_name_via_auto_map(config=config)
    assert result is None


def test_returns_class_name_when_auto_map_present():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "some.module.Class"}, name_or_path=None)
    result = get_causal_lm_class_name_via_auto_map(config=config)
    assert result == "Class"


def test_splits_on_last_dot():
    config = DummyConfig(auto_map={"AutoModelForCausalLM": "pkg.subpkg.module.DeepClass"}, name_or_path="repo/id")
    result = get_causal_lm_class_name_via_auto_map(config)
    assert result == "DeepClass"


def test_returns_none_when_key_missing():
    config = DummyConfig(auto_map={"AutoModel": "some.module.Class"}, name_or_path="repo/id")
    result = get_causal_lm_class_name_via_auto_map(config)
    assert result is None
