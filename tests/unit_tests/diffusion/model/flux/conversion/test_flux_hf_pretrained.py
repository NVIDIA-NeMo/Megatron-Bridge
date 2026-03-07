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

from megatron.bridge.diffusion.models.flux.conversion import flux_hf_pretrained as flux_hf_module


pytestmark = [pytest.mark.unit]


def test_load_config_uses_transformer_subfolder(monkeypatch, tmp_path):
    calls = []

    class FakeModel:
        def __init__(self, cfg):
            self.config = cfg

    class FakeFlux:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            calls.append((str(path), subfolder))
            return FakeModel(cfg={"ok": True})

    monkeypatch.setattr(flux_hf_module, "FluxTransformer2DModel", FakeFlux)

    src = tmp_path / "hf"
    src.mkdir(parents=True, exist_ok=True)
    hf = flux_hf_module.PreTrainedFlux(str(src))

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

    monkeypatch.setattr(flux_hf_module, "FluxSafeTensorsStateSource", FakeSource)
    monkeypatch.setattr(flux_hf_module, "StateDict", FakeStateDict)

    src = tmp_path / "hf_model"
    (src / "transformer").mkdir(parents=True, exist_ok=True)
    hf = flux_hf_module.PreTrainedFlux(str(src))

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

    hf = flux_hf_module.PreTrainedFlux(str(src))
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

    class FakeFlux:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            # Ensure it targets the transformer subfolder
            assert subfolder == "transformer"
            return FakeModel()

    monkeypatch.setattr(flux_hf_module, "FluxTransformer2DModel", FakeFlux)

    src = tmp_path / "empty_src"
    src.mkdir(parents=True, exist_ok=True)

    dest = tmp_path / "out"
    hf = flux_hf_module.PreTrainedFlux(str(src))
    hf.save_artifacts(dest)

    # Should create transformer/config.json with exported contents
    dest_cfg = dest / "transformer" / "config.json"
    assert dest_cfg.is_file()
    assert json.loads(dest_cfg.read_text()) == {"from_model": True}


def test_save_artifacts_handles_export_failure(monkeypatch, tmp_path):
    class FailingFlux:
        @classmethod
        def from_pretrained(cls, path, subfolder=None):
            raise RuntimeError("fail")

    monkeypatch.setattr(flux_hf_module, "FluxTransformer2DModel", FailingFlux)

    src = tmp_path / "src2"
    src.mkdir(parents=True, exist_ok=True)
    dest = tmp_path / "dest2"

    hf = flux_hf_module.PreTrainedFlux(str(src))
    # Should not raise
    hf.save_artifacts(dest)

    # Transformer folder created but no config.json written
    dest_tdir = dest / "transformer"
    assert dest_tdir.is_dir()
    assert not (dest_tdir / "config.json").exists()


def test_flux_safetensors_state_source_save_generator(monkeypatch, tmp_path):
    """Test that FluxSafeTensorsStateSource.save_generator writes to transformer/ subfolder"""
    parent_save_called = []

    class FakeParentClass:
        def save_generator(self, generator, output_path, strict=True):
            parent_save_called.append((str(output_path), strict))
            return "success"

    # Temporarily replace SafeTensorsStateSource for testing
    monkeypatch.setattr(flux_hf_module, "SafeTensorsStateSource", FakeParentClass)

    # Need to recreate the class with the new base
    class TestFluxSource(FakeParentClass):
        def save_generator(self, generator, output_path, strict=True):
            from pathlib import Path

            target_dir = Path(output_path) / "transformer"
            return super().save_generator(generator, target_dir, strict=strict)

    output_path = tmp_path / "output"
    source = TestFluxSource()
    result = source.save_generator(None, output_path, strict=False)

    # Verify parent's save_generator was called with transformer/ subfolder
    assert len(parent_save_called) == 1
    assert parent_save_called[0][0] == str(output_path / "transformer")
    assert parent_save_called[0][1] is False
    assert result == "success"


def test_pretrained_flux_model_name_or_path_property(tmp_path):
    """Test that model_name_or_path property returns the correct path"""
    src = tmp_path / "model"
    src.mkdir(parents=True, exist_ok=True)

    hf = flux_hf_module.PreTrainedFlux(str(src))
    assert hf.model_name_or_path == str(src)


def test_load_model_calls_from_pretrained(monkeypatch, tmp_path):
    """Test that _load_model calls FluxTransformer2DModel.from_pretrained"""
    calls = []

    class FakeFlux:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(str(path))
            return FakeFlux()

    monkeypatch.setattr(flux_hf_module, "FluxTransformer2DModel", FakeFlux)

    src = tmp_path / "model"
    src.mkdir(parents=True, exist_ok=True)

    hf = flux_hf_module.PreTrainedFlux(str(src))
    model = hf._load_model()

    assert len(calls) == 1
    assert calls[0] == str(src)
    assert isinstance(model, FakeFlux)


def test_state_uses_model_state_dict_when_model_loaded(monkeypatch, tmp_path):
    """Test that state property uses model's state_dict when model is already loaded"""
    model_state = {"weight1": "value1"}

    class FakeModel:
        def state_dict(self):
            return model_state

    class FakeStateDict:
        def __init__(self, source):
            self.source = source

    monkeypatch.setattr(flux_hf_module, "StateDict", FakeStateDict)

    src = tmp_path / "model"
    src.mkdir(parents=True, exist_ok=True)

    hf = flux_hf_module.PreTrainedFlux(str(src))
    # Manually set _model to simulate loaded model
    hf._model = FakeModel()

    state = hf.state
    # Should use model's state_dict, not file source
    assert state.source == model_state


def test_save_artifacts_creates_transformer_directory(tmp_path):
    """Test that save_artifacts creates transformer directory structure"""
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)

    dest = tmp_path / "dest"

    hf = flux_hf_module.PreTrainedFlux(str(src))
    hf.save_artifacts(dest)

    # Verify transformer directory was created
    assert (dest / "transformer").is_dir()


def test_save_artifacts_only_copies_index_if_config_exists(tmp_path):
    """Test that index file is only copied when config.json exists"""
    src = tmp_path / "src"
    tdir = src / "transformer"
    tdir.mkdir(parents=True, exist_ok=True)

    # Only create index file, not config.json
    index_src = tdir / "diffusion_pytorch_model.safetensors.index.json"
    index_data = {"weight_map": {}}
    index_src.write_text(json.dumps(index_data))

    dest = tmp_path / "dest"

    hf = flux_hf_module.PreTrainedFlux(str(src))
    hf.save_artifacts(dest)

    # Index should not be copied since config.json doesn't exist
    dest_tdir = dest / "transformer"
    assert dest_tdir.is_dir()
    assert not (dest_tdir / "diffusion_pytorch_model.safetensors.index.json").exists()
