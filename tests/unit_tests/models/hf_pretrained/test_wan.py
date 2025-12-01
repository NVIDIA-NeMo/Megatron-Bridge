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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from megatron.bridge.models.hf_pretrained.state import DictStateSource
from megatron.bridge.models.hf_pretrained.wan import PreTrainedWAN, WanSafeTensorsStateSource


class TestPreTrainedWANLoad:
    @patch("megatron.bridge.models.hf_pretrained.wan.WanTransformer3DModel.from_pretrained")
    def test_load_model_calls_from_pretrained(self, mock_from_pretrained):
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        wan = PreTrainedWAN(model_name_or_path="wan-model")
        model = wan.model

        assert model is mock_model
        mock_from_pretrained.assert_called_once_with("wan-model")

    @patch("megatron.bridge.models.hf_pretrained.wan.WanTransformer3DModel.from_pretrained")
    def test_load_config_uses_transformer_subfolder(self, mock_from_pretrained):
        mock_config = object()
        mock_from_pretrained.return_value = SimpleNamespace(config=mock_config)

        wan = PreTrainedWAN(model_name_or_path="wan-model")
        cfg = wan.config

        assert cfg is mock_config
        mock_from_pretrained.assert_called_once_with("wan-model", subfolder="transformer")


class TestPreTrainedWANState:
    def test_state_uses_transformer_subfolder_when_model_not_loaded_and_caches(self, tmp_path: Path):
        model_dir = tmp_path / "wan"
        (model_dir / "transformer").mkdir(parents=True)

        wan = PreTrainedWAN(model_name_or_path=str(model_dir))

        state1 = wan.state
        state2 = wan.state

        # Same cached object
        assert state1 is state2

        # Source should be WAN-specific safetensors source pointing to transformer/
        assert isinstance(state1.source, WanSafeTensorsStateSource)
        expected = model_dir / "transformer"
        # Avoid touching .path (which may try to resolve); check raw attribute
        assert Path(state1.source.model_name_or_path) == expected

    def test_state_uses_in_memory_when_model_loaded(self):
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"a": 1, "b": 2}

        wan = PreTrainedWAN(model_name_or_path="unused")
        wan.model = mock_model

        state = wan.state

        # When model is loaded, source should be DictStateSource
        assert isinstance(state.source, DictStateSource)
        # Keys should come from mock dict
        assert set(state.keys()) == {"a", "b"}


class TestWanSafeTensorsStateSourceSaveGenerator:
    def test_redirects_output_under_transformer_subdir(self, tmp_path: Path):
        src = WanSafeTensorsStateSource(path="dummy")
        gen = [("x", MagicMock())]

        with patch("megatron.bridge.models.hf_pretrained.state.SafeTensorsStateSource.save_generator") as parent_save:
            src.save_generator(gen, tmp_path, strict=True)

            assert parent_save.call_count == 1
            args, kwargs = parent_save.call_args
            # args: (generator, output_path, strict)
            assert args[0] is gen
            assert Path(args[1]) == tmp_path / "transformer"
            # strict may be passed as a keyword
            assert kwargs.get("strict", True) is True


class TestPreTrainedWANSaveArtifacts:
    def test_copies_existing_transformer_files(self, tmp_path: Path):
        # Prepare source repo with transformer/config.json and optional index
        src_dir = tmp_path / "src"
        t_src = src_dir / "transformer"
        t_src.mkdir(parents=True)
        cfg_content = {"num_layers": 2}
        (t_src / "config.json").write_text(json.dumps(cfg_content))
        (t_src / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))

        dest_dir = tmp_path / "dest"
        wan = PreTrainedWAN(model_name_or_path=str(src_dir))

        wan.save_artifacts(dest_dir)

        t_dest = dest_dir / "transformer"
        assert (t_dest / "config.json").exists()
        assert json.loads((t_dest / "config.json").read_text()) == cfg_content
        # Optional index should also be copied if present
        assert (t_dest / "diffusion_pytorch_model.safetensors.index.json").exists()

    @patch("megatron.bridge.models.hf_pretrained.wan.WanTransformer3DModel.from_pretrained")
    def test_exports_config_when_missing_in_source(self, mock_from_pretrained, tmp_path: Path):
        # No config in source, so export path should be taken
        src_dir = tmp_path / "src_no_cfg"
        (src_dir / "transformer").mkdir(parents=True)

        mock_cfg_dict = {"num_layers": 12, "eps": 1e-6}

        class _Cfg:
            def to_dict(self):
                return mock_cfg_dict

        mock_from_pretrained.return_value = SimpleNamespace(config=_Cfg())

        dest_dir = tmp_path / "dest"
        wan = PreTrainedWAN(model_name_or_path=str(src_dir))
        wan.save_artifacts(dest_dir)

        t_dest = dest_dir / "transformer"
        assert (t_dest / "config.json").exists()
        assert json.loads((t_dest / "config.json").read_text()) == mock_cfg_dict
