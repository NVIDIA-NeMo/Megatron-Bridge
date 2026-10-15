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
Unit tests for StateDict class.
"""

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.bridge.models.hf_pretrained.base import PreTrainedBase
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict, StateSource


@pytest.fixture
def mock_model():
    """Create a mock model with a state dict."""
    model = MagicMock()
    model.state_dict.return_value = {
        "transformer.wte.weight": torch.randn(50257, 768),
        "transformer.h.0.attn.qkv.weight": torch.randn(2304, 768),
        "transformer.h.0.attn.out.weight": torch.randn(768, 768),
        "transformer.h.0.attn.out.bias": torch.randn(768),
        "transformer.ln_f.weight": torch.randn(768),
        "transformer.ln_f.bias": torch.randn(768),
        "lm_head.weight": torch.randn(50257, 768),
    }
    return model


@pytest.fixture
def temp_safetensors_dir():
    """Create a temporary directory with mock safetensors files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create index file
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "transformer.wte.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.qkv.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.out.weight": "model-00001-of-00002.safetensors",
                "transformer.h.0.attn.out.bias": "model-00001-of-00002.safetensors",
                "transformer.ln_f.weight": "model-00002-of-00002.safetensors",
                "transformer.ln_f.bias": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }

        index_path = Path(temp_dir) / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index_data, f)

        # Create mock safetensors files (empty for testing)
        (Path(temp_dir) / "model-00001-of-00002.safetensors").touch()
        (Path(temp_dir) / "model-00002-of-00002.safetensors").touch()

        yield temp_dir


class MockPreTrainedBase(PreTrainedBase):
    """Mock implementation of PreTrainedBase for testing."""

    def __init__(self, model_name_or_path=None, model=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self._model = model
        self._state_dict_accessor = None

    @property
    def model(self):
        return self._model

    def _load_model(self):
        """Abstract method implementation."""
        return self._model


class TestStateDictInitialization:
    """Test StateDict initialization and basic setup."""

    def test_init(self):
        """Test StateDict initialization."""
        source = {"test.weight": torch.randn(2, 2)}
        state_dict = StateDict(source)

        assert state_dict.source == source
        assert hasattr(state_dict, "source")

    def test_repr(self, mock_model):
        """Test string representation."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        repr_str = repr(state_dict)

        assert "StateDict" in repr_str
        assert "7 entries" in repr_str

    def test_repr_no_access(self):
        """Test string representation with empty dict."""
        state_dict = StateDict({})

        repr_str = repr(state_dict)

        assert "StateDict" in repr_str
        assert "0 entries" in repr_str


class TestStateDictKeyRetrieval:
    """Test methods for retrieving and managing keys."""

    def test_get_all_keys_from_loaded_model(self, mock_model):
        """Test getting keys from a loaded model."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = list(state_dict.keys())

        assert len(keys) > 0
        assert "transformer.wte.weight" in keys
        assert "lm_head.weight" in keys
        assert keys == sorted(keys)  # Should be sorted

    def test_get_all_keys_from_safetensors(self, temp_safetensors_dir):
        """Test getting keys from safetensors files."""
        # Mock SafeTensorsStateSource to avoid actual file operations
        with patch("megatron.bridge.models.hf_pretrained.state.SafeTensorsStateSource") as mock_source_cls:
            mock_source = MagicMock()
            mock_source.keys.return_value = [
                "transformer.wte.weight",
                "transformer.h.0.attn.qkv.weight",
                "transformer.h.0.attn.out.weight",
                "transformer.h.0.attn.out.bias",
                "transformer.ln_f.weight",
                "transformer.ln_f.bias",
                "lm_head.weight",
            ]
            mock_source_cls.return_value = mock_source

            source = SafeTensorsStateSource(temp_safetensors_dir)
            state_dict = StateDict(source)

            keys = list(state_dict.keys())

            assert len(keys) == 7
            assert "transformer.wte.weight" in keys
            assert "lm_head.weight" in keys

    def test_get_all_keys_no_model_or_path(self):
        """Test getting keys with empty dict works fine."""
        state_dict = StateDict({})
        keys = list(state_dict.keys())
        assert keys == []

    def test_keys_method(self, mock_model):
        """Test keys method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = list(state_dict.keys())

        assert isinstance(keys, list)
        assert len(keys) == 7
        assert keys == sorted(keys)

    def test_inherited_mapping_items(self, mock_model):
        """Test items behavior inherited from Mapping."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        assert "items" not in StateDict.__dict__
        items = list(state_dict.items())

        assert isinstance(items, list)
        assert len(items) == 7
        assert all(isinstance(item, tuple) and len(item) == 2 for item in items)

    def test_contains_method(self, mock_model):
        """Test contains method."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        assert "transformer.wte.weight" in state_dict
        assert "missing.key" not in state_dict


class TestStateDictTensorLoading:
    """Test tensor loading functionality."""

    def test_load_tensors_from_model(self, mock_model):
        """Test loading tensors from a loaded model."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Access tensors directly
        tensor1 = state_dict["transformer.wte.weight"]
        tensor2 = state_dict["lm_head.weight"]

        assert tensor1.shape == (50257, 768)
        assert isinstance(tensor1, torch.Tensor)
        assert isinstance(tensor2, torch.Tensor)

    def test_load_tensors_from_safetensors(self, temp_safetensors_dir):
        """Test loading tensors from safetensors files."""

        # Create a mock StateSource
        class MockStateSource(StateSource):
            def get_all_keys(self):
                return ["transformer.wte.weight", "lm_head.weight"]

            def load_tensors(self, keys):
                result = {}
                for key in keys:
                    if key in self.get_all_keys():
                        result[key] = torch.randn(10, 10)
                    else:
                        raise KeyError(f"Key not found: {key}")
                return result

            def keys(self):
                return self.get_all_keys()

        mock_source = MockStateSource()
        state_dict = StateDict(mock_source)

        keys = ["transformer.wte.weight", "lm_head.weight"]
        tensors = state_dict[keys]

        assert len(tensors) == 2
        assert "transformer.wte.weight" in tensors
        assert "lm_head.weight" in tensors

    def test_load_tensors_missing_keys(self, mock_model):
        """Test loading tensors with missing keys raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = ["transformer.wte.weight", "missing.key"]

        with pytest.raises(KeyError, match="Keys not found: \\['missing.key'\\]"):
            _ = state_dict[keys]

    def test_empty_keys_list(self, mock_model):
        """Test loading empty keys list."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict[[]]

        assert tensors == {}

    def test_call_method(self, mock_model):
        """Test calling state dict returns full dict."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        full_dict = state_dict()

        assert isinstance(full_dict, dict)
        assert len(full_dict) == 7
        assert "transformer.wte.weight" in full_dict
        assert "lm_head.weight" in full_dict


class TestStateDictPatternMatching:
    """Test pattern matching functionality."""

    def test_match_keys_exact(self, mock_model):
        """Test matching keys with exact pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        matched = state_dict["transformer.wte.weight"]

        assert isinstance(matched, torch.Tensor)
        assert matched.shape == (50257, 768)

    def test_match_keys_glob(self, mock_model):
        """Test matching keys with glob pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Match all attention weights
        matched = state_dict["*.attn.*.weight"]

        assert len(matched) == 2
        assert all("attn" in key and "weight" in key for key in matched)

    def test_match_keys_regex(self, mock_model):
        """Test matching keys with regex pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        # Match all bias tensors
        pattern = re.compile(r".*\.bias$")
        matched = state_dict[pattern]

        assert len(matched) == 2
        assert all(key.endswith(".bias") for key in matched)

    def test_no_matching_glob_pattern(self, mock_model):
        """Test glob pattern with no matches."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(KeyError, match="No keys match pattern"):
            _ = state_dict["*.nonexistent.*"]

    def test_no_matching_regex_pattern(self, mock_model):
        """Test regex pattern with no matches."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        pattern = re.compile(r"nonexistent\..*")
        with pytest.raises(KeyError, match="No keys match regex pattern"):
            _ = state_dict[pattern]


class TestStateDictIndexing:
    """Test __getitem__ indexing functionality."""

    def test_getitem_single_key(self, mock_model):
        """Test accessing single key."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensor = state_dict["transformer.wte.weight"]

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (50257, 768)

    def test_getitem_missing_key(self, mock_model):
        """Test accessing missing key raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(KeyError, match="Key not found: missing.key"):
            _ = state_dict["missing.key"]

    def test_getitem_multiple_keys(self, mock_model):
        """Test accessing multiple keys."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        keys = ["transformer.wte.weight", "lm_head.weight"]
        tensors = state_dict[keys]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all(key in tensors for key in keys)

    def test_getitem_glob_pattern(self, mock_model):
        """Test accessing with glob pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        tensors = state_dict["*.bias"]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all(key.endswith(".bias") for key in tensors)

    def test_getitem_regex_pattern(self, mock_model):
        """Test accessing with regex pattern."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        pattern = re.compile(r"transformer\.h\.\d+\.attn\..*\.weight")
        tensors = state_dict[pattern]

        assert isinstance(tensors, dict)
        assert len(tensors) == 2
        assert all("attn" in key and "weight" in key for key in tensors)

    def test_getitem_invalid_type(self, mock_model):
        """Test accessing with invalid type raises error."""
        state_dict_source = mock_model.state_dict()
        state_dict = StateDict(state_dict_source)

        with pytest.raises(TypeError, match="Key must be str, list of str, or compiled regex"):
            _ = state_dict[123]


class TestSafeTensorsStateSourceSave:
    """Test streaming safetensors save behavior."""

    def test_save_generator_ignores_source_key_prefixes(self, tmp_path, capsys):
        from safetensors.torch import save_file

        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        save_file(
            {
                "model.weight": torch.ones(1),
                "mtp.0.weight": torch.ones(1),
            },
            source_dir / "model-00001-of-00001.safetensors",
        )
        with open(source_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.weight": "model-00001-of-00001.safetensors",
                        "mtp.0.weight": "model-00001-of-00001.safetensors",
                    },
                },
                f,
            )

        source = SafeTensorsStateSource(source_dir)
        source.save_generator(
            iter([("model.weight", torch.zeros(1))]),
            output_dir,
            ignored_source_key_prefixes=("mtp.",),
        )

        output = capsys.readouterr().out
        assert "Error:" not in output
        assert "Success: All tensors from the original checkpoint were written." in output

        with open(output_dir / "model.safetensors.index.json") as f:
            output_index = json.load(f)
        assert output_index["weight_map"] == {"model.weight": "model-00001-of-00001.safetensors"}

    def test_filter_source_keys_filters_expected_map(self):
        source_map = {
            "model.weight": "model.safetensors",
            "model.weight_scale_inv": "model.safetensors",
            "mtp.0.weight": "model.safetensors",
            "mtp_extra.weight": "model.safetensors",
        }

        assert SafeTensorsStateSource._filter_source_keys(None, ("mtp.",), ("_scale_inv",)) == {}
        assert SafeTensorsStateSource._filter_source_keys(source_map, None, None) == source_map
        assert SafeTensorsStateSource._filter_source_keys(source_map, ("mtp.",), ("_scale_inv",)) == {
            "model.weight": "model.safetensors",
            "mtp_extra.weight": "model.safetensors",
        }

    @pytest.mark.parametrize("distributed_save", [False, True])
    def test_save_generator_ignores_source_key_suffixes(self, tmp_path, distributed_save):
        from safetensors import safe_open
        from safetensors.torch import save_file

        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        shard = "model-00001-of-00001.safetensors"
        save_file(
            {
                "model.weight": torch.ones(1),
                "model.weight_scale_inv": torch.ones(1),
            },
            source_dir / shard,
        )
        with open(source_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.weight": shard,
                        "model.weight_scale_inv": shard,
                    },
                },
                f,
            )

        source = SafeTensorsStateSource(source_dir)
        source.save_generator(
            iter([("model.weight", torch.zeros(1))]),
            output_dir,
            distributed_save=distributed_save,
            ignored_source_key_suffixes=("_scale_inv",),
        )

        with safe_open(output_dir / shard, framework="pt") as output:
            assert output.keys() == ["model.weight"]
        with open(output_dir / "model.safetensors.index.json") as f:
            output_index = json.load(f)
        assert output_index["weight_map"] == {"model.weight": shard}

    def test_distributed_save_generator_ignores_source_key_prefixes_without_initialized_dist(self, tmp_path):
        from safetensors.torch import save_file

        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        save_file(
            {
                "model.weight": torch.ones(1),
                "mtp.0.weight": torch.ones(1),
            },
            source_dir / "model-00001-of-00001.safetensors",
        )
        with open(source_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {},
                    "weight_map": {
                        "model.weight": "model-00001-of-00001.safetensors",
                        "mtp.0.weight": "model-00001-of-00001.safetensors",
                    },
                },
                f,
            )

        source = SafeTensorsStateSource(source_dir)
        source.save_generator(
            iter([("model.weight", torch.zeros(1))]),
            output_dir,
            distributed_save=True,
            ignored_source_key_prefixes=("mtp.",),
        )

        with open(output_dir / "model.safetensors.index.json") as f:
            output_index = json.load(f)
        assert output_index["weight_map"] == {"model.weight": "model-00001-of-00001.safetensors"}


class TestSafeTensorsStateSourceResolve:
    """Test SafeTensorsStateSource._resolve_path with revision and offline mode."""

    def test_resolve_path_with_revision(self, tmp_path):
        revision = "b5968e9190ef"  # pragma: allowlist secret
        source = SafeTensorsStateSource("some/model-repo", revision=revision)
        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.return_value = str(tmp_path)
            resolved = source.path
            mock_download.assert_called_once()
            _, kwargs = mock_download.call_args
            assert kwargs.get("repo_id") == "some/model-repo"
            assert kwargs.get("revision") == revision
            assert resolved == tmp_path

    def test_resolve_path_offline_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
        source = SafeTensorsStateSource("some/model-repo", revision="my-tag")
        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.return_value = str(tmp_path)
            resolved = source.path
            mock_download.assert_called_once()
            _, kwargs = mock_download.call_args
            assert kwargs.get("revision") == "my-tag"
            assert kwargs.get("local_files_only") is True
            assert resolved == tmp_path

    def test_pretrained_base_passes_revision(self):
        class DummyPreTrained(PreTrainedBase):
            def __init__(self, model_name_or_path, revision=None):
                super().__init__(revision=revision)
                self.model_name_or_path = model_name_or_path

            def _load_config(self):
                pass

            def _load_model(self):
                pass

        obj = DummyPreTrained("deepseek-ai/DeepSeek-V4-Pro", revision="abc1234")
        with patch.object(SafeTensorsStateSource, "_resolve_path", return_value=Path("/tmp")):
            accessor = obj.state
            source = accessor.source
            assert isinstance(source, SafeTensorsStateSource)
            assert source.revision == "abc1234"


class TestSafeTensorsStateSourceCache:
    """Small-tensor cache and safe_open handle reuse in SafeTensorsStateSource.load_tensors."""

    @staticmethod
    def _write_checkpoint(root: Path, tensors: dict, filename: str = "model-00001-of-00001.safetensors") -> None:
        from safetensors.torch import save_file

        root.mkdir(parents=True, exist_ok=True)
        save_file(tensors, root / filename)
        index = {"metadata": {}, "weight_map": {k: filename for k in tensors}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))

    @staticmethod
    def _count_safe_open(monkeypatch):
        import safetensors

        calls = {"n": 0}
        real = safetensors.safe_open

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(safetensors, "safe_open", counting)
        return calls

    def test_small_cache_returns_independent_tensors(self, tmp_path):
        self._write_checkpoint(tmp_path, {"w.scale": torch.tensor([1.0]), "w.weight": torch.ones(4, 4)})
        source = SafeTensorsStateSource(tmp_path)

        first = source.load_tensors(["w.scale"])["w.scale"]
        first.zero_()
        second = source.load_tensors(["w.scale"])["w.scale"]
        assert second.item() == 1.0
        second.fill_(7.0)
        assert source.load_tensors(["w.scale"])["w.scale"].item() == 1.0

    def test_retry_refreshes_incomplete_shard(self, tmp_path, monkeypatch):
        from safetensors.torch import save_file

        shard = "model-00001-of-00001.safetensors"
        self._write_checkpoint(tmp_path, {"text.other": torch.zeros(2)}, shard)
        index = {"metadata": {}, "weight_map": {"text.other": shard, "text.weight": shard}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

        def replace_shard(_seconds):
            complete = tmp_path / "complete.safetensors"
            save_file({"text.other": torch.zeros(2), "text.weight": torch.ones(2)}, complete)
            complete.replace(tmp_path / shard)

        monkeypatch.setattr("time.sleep", replace_shard)
        source = SafeTensorsStateSource(tmp_path, key_prefix="text.")
        loaded = source.load_tensors(["weight"])
        assert torch.equal(loaded["weight"], torch.ones(2))

    @pytest.mark.parametrize("cache_small,reuse_handles", [("1", "1"), ("1", "0"), ("0", "1"), ("0", "0")])
    def test_cache_and_handle_switches(self, tmp_path, monkeypatch, cache_small, reuse_handles):
        self._write_checkpoint(tmp_path, {"w.scale": torch.tensor([2.0]), "w.weight": torch.ones(8, 8)})
        monkeypatch.setenv("MBRIDGE_STATE_CACHE_SMALL_TENSORS", cache_small)
        monkeypatch.setenv("MBRIDGE_STATE_REUSE_HANDLES", reuse_handles)
        calls = self._count_safe_open(monkeypatch)
        source = SafeTensorsStateSource(tmp_path)

        for _ in range(3):
            assert source.load_tensors(["w.scale"])["w.scale"].item() == 2.0
        opens_after_scales = calls["n"]
        if cache_small == "1":
            assert opens_after_scales == 1  # first read opens the shard, the two cache hits do not
        elif reuse_handles == "1":
            assert opens_after_scales == 1  # one handle, reused for all three reads
        else:
            assert opens_after_scales == 3  # baseline behaviour: one safe_open per load_tensors

        # w.weight (64 elements) is also below the default small-tensor threshold, so with the
        # cache on its second read is a hit; with the cache off every read goes to the shard.
        for _ in range(2):
            assert source.load_tensors(["w.weight"]).keys() == {"w.weight"}
        extra_opens = calls["n"] - opens_after_scales
        expected = {("1", "1"): 0, ("1", "0"): 1, ("0", "1"): 0, ("0", "0"): 2}
        assert extra_opens == expected[(cache_small, reuse_handles)]

    def test_cache_threshold_and_instance_isolation(self, tmp_path, monkeypatch):
        self._write_checkpoint(tmp_path, {"small": torch.ones(4), "big": torch.ones(16)})
        monkeypatch.setenv("MBRIDGE_STATE_SMALL_TENSOR_NUMEL", "4")
        monkeypatch.setenv("MBRIDGE_STATE_REUSE_HANDLES", "0")
        calls = self._count_safe_open(monkeypatch)

        source = SafeTensorsStateSource(tmp_path)
        source.load_tensors(["small", "big"])
        assert set(source._small_tensor_cache) == {"small"}  # inclusive threshold: 4 <= 4 cached, 16 is not
        source.load_tensors(["small"])
        assert calls["n"] == 1  # served from the cache
        source.load_tensors(["big"])
        assert calls["n"] == 2  # re-read from the shard

        other = SafeTensorsStateSource(tmp_path)
        assert other._small_tensor_cache == {}
        other.load_tensors(["small"])
        assert calls["n"] == 3  # a new instance starts cold
