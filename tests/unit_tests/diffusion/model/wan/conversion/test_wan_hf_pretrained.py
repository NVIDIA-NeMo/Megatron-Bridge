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

import pytest

from megatron.bridge.diffusion.models.wan.conversion import wan_hf_pretrained as wan_hf_module


pytestmark = [pytest.mark.unit]


def test_load_config_uses_transformer_subfolder(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def __init__(self, cfg):
            self.config = cfg

    class FakeWAN:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            calls.append((str(path), subfolder))
            return FakeModel(cfg={"ok": True})

    monkeypatch.setattr(wan_hf_module, "WanTransformer3DModel", FakeWAN)

    src = tmp_path / "hf"
    src.mkdir(parents=True, exist_ok=True)
    hf = wan_hf_module.PreTrainedWAN(str(src))

    # Accessing .config should trigger _load_config
    cfg = hf.config
    assert cfg == {"ok": True}
    # Ensure we called with transformer subfolder
    assert calls and calls[-1][1] == "transformer"


def test_state_uses_transformer_subfolder_and_caches(monkeypatch, tmp_path):
    captured = {"source_path": None, "constructed": 0}

    class FakeSource:
        def __init__(self, path):
            captured["source_path"] = str(path)

    class FakeStateDict:
        def __init__(self, source):
            self.source = source
            captured["constructed"] += 1

    monkeypatch.setattr(wan_hf_module, "WanSafeTensorsStateSource", FakeSource)
    monkeypatch.setattr(wan_hf_module, "StateDict", FakeStateDict)

    src = tmp_path / "hf_model"
    (src / "transformer").mkdir(parents=True, exist_ok=True)
    hf = wan_hf_module.PreTrainedWAN(str(src))

    s1 = hf.state
    s2 = hf.state  # Cached
    assert s1 is s2
    # Correct subfolder used
    assert captured["source_path"] == str(src / "transformer")
    # StateDict constructed only once due to caching
    assert captured["constructed"] == 1


def test_save_artifacts_copies_existing_files(tmp_path):
    # Prepare source with transformer/config.json and index
    src = tmp_path / "src"
    tdir = src / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)
    config_src = tdir / "config.json"
    index_src = tdir / "diffusion_pytorch_model.safetensors.index.json"
    config_data = {"a": 1}
    index_data = {"weight_map": {}}
    config_src.write_text(json.dumps(config_data))
    index_src.write_text(json.dumps(index_data))

    # Destination directory
    dest = tmp_path / "dest"

    hf = wan_hf_module.PreTrainedWAN(str(src))
    hf.save_artifacts(str(dest))

    # Validate files copied
    dest_tdir = dest / "transformer"
    assert dest_tdir.is_dir()
    assert json.loads((dest_tdir / "config.json").read_text()) == config_data
    assert json.loads((dest_tdir / "diffusion_pytorch_model.safetensors.index.json").read_text()) == index_data


def test_save_artifacts_exports_config_when_missing(monkeypatch, tmp_path):
    class FakeCfg:
        def to_dict(self):
            return {"from_model": True}

    class FakeModel:
        def __init__(self):
            self.config = FakeCfg()

    class FakeWAN:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            # Ensure it targets the transformer subfolder
            assert subfolder == "transformer"
            return FakeModel()

    monkeypatch.setattr(wan_hf_module, "WanTransformer3DModel", FakeWAN)

    src = tmp_path / "empty_src"
    src.mkdir(parents=True, exist_ok=True)

    dest = tmp_path / "out"
    hf = wan_hf_module.PreTrainedWAN(str(src))
    hf.save_artifacts(dest)

    # Should create transformer/config.json with exported contents
    dest_cfg = dest / "transformer" / "config.json"
    assert dest_cfg.is_file()
    assert json.loads(dest_cfg.read_text()) == {"from_model": True}


def test_save_artifacts_handles_export_failure(monkeypatch, tmp_path):
    class FailingWAN:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            raise RuntimeError("fail")

    monkeypatch.setattr(wan_hf_module, "WanTransformer3DModel", FailingWAN)

    src = tmp_path / "src2"
    src.mkdir(parents=True, exist_ok=True)
    dest = tmp_path / "dest2"

    hf = wan_hf_module.PreTrainedWAN(str(src))
    # Should not raise
    hf.save_artifacts(dest)

    # Transformer folder created but no config.json written
    dest_tdir = dest / "transformer"
    assert dest_tdir.is_dir()
    assert not (dest_tdir / "config.json").exists()
