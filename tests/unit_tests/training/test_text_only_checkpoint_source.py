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
        model=SimpleNamespace(hf_model_id="org/vl", hf_model_text_only=False, hf_model_revision="first"),
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


@pytest.mark.parametrize("use_template_override", [False, True])
def test_checkpoint_replacement_source_does_not_inherit_revision(monkeypatch, use_template_override):
    monkeypatch.setattr(checkpointing, "_AUTO_BRIDGE_CACHE", {})
    factory = Mock(return_value=object())
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", factory)
    cfg = SimpleNamespace(
        model=SimpleNamespace(hf_model_id="org/vl", hf_model_text_only=True, hf_model_revision="original"),
        checkpoint=SimpleNamespace(
            hf_trust_remote_code=True,
            hf_source_path="org/text-export" if use_template_override else None,
        ),
    )
    kwargs = {} if use_template_override else {"hf_source": "org/text-export"}
    first = checkpointing._build_auto_bridge_for_save(cfg, **kwargs)
    factory.assert_called_once_with("org/text-export", trust_remote_code=True, text_only=True, revision=None)
    cfg.model.hf_model_revision = "another-original-revision"
    assert checkpointing._build_auto_bridge_for_save(cfg, **kwargs) is first
    assert factory.call_count == 1


def test_peft_sidecar_uses_standalone_text_source_override(monkeypatch, tmp_path):
    monkeypatch.setattr(checkpointing, "_AUTO_BRIDGE_CACHE", {})
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(checkpointing, "get_rank_safe", lambda: 0)
    bridge = Mock(text_only=False)
    factory = Mock(return_value=bridge)
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", factory)
    cfg = SimpleNamespace(
        model=SimpleNamespace(hf_model_id="org/vl", hf_model_text_only=True, hf_model_revision="original"),
        checkpoint=SimpleNamespace(hf_source_path="org/text-base", hf_trust_remote_code=False),
        peft=Mock(),
    )
    checkpointing._save_hf_adapter_weights(SimpleNamespace(cfg=cfg), [], str(tmp_path))
    factory.assert_called_once_with("org/text-base", trust_remote_code=False, text_only=True, revision=None)
    bridge.save_hf_adapter.assert_called_once_with(
        [], str(tmp_path), peft_config=cfg.peft, base_model_name_or_path="org/text-base", show_progress=False
    )
