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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.bridge import AutoBridge
from megatron.bridge.training import checkpointing


pytestmark = pytest.mark.unit


def test_checkpoint_bridge_cache_separates_full_text_and_revisions(monkeypatch):
    cache = {}
    monkeypatch.setattr(checkpointing, "_AUTO_BRIDGE_CACHE", cache)
    factory = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", factory)
    cfg = SimpleNamespace(
        model=SimpleNamespace(hf_model_text_only=False, hf_model_revision="first"),
        checkpoint=SimpleNamespace(hf_trust_remote_code=True),
    )
    full = checkpointing._build_auto_bridge_for_save(cfg, hf_source="org/vl")
    factory.assert_called_once_with("org/vl", trust_remote_code=True)
    cfg.model.hf_model_text_only = True
    text = checkpointing._build_auto_bridge_for_save(cfg, hf_source="org/vl")
    factory.assert_called_with("org/vl", trust_remote_code=True, text_only=True, revision="first")
    assert text is not full
    assert checkpointing._build_auto_bridge_for_save(cfg, hf_source="org/vl") is text
    cfg.model.hf_model_revision = "second"
    assert checkpointing._build_auto_bridge_for_save(cfg, hf_source="org/vl") is not text
    factory.assert_called_with("org/vl", trust_remote_code=True, text_only=True, revision="second")
    assert factory.call_count == 3
