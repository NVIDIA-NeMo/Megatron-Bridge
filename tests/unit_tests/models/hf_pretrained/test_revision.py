# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Revision selection against real offline Hub caches, without model downloads."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import huggingface_hub
import pytest
import torch
from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError
from safetensors.torch import save_file

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.hf_pretrained.token_classification import PreTrainedTokenClassification


pytestmark = pytest.mark.unit
_REPO_ID = "test-org/revision-test"
_REVISION = "a" * 40
_OTHER_REVISION = "b" * 40
_WRAPPERS = [PreTrainedCausalLM, PreTrainedMaskedLM, PreTrainedTokenClassification]


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def unexpected_hub_request(*args, **kwargs):
        pytest.fail("Resolving a cached offline checkpoint must not request Hub metadata")

    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", unexpected_hub_request)
    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_tree", unexpected_hub_request)
    return tmp_path / "models--test-org--revision-test"


def _write_snapshot(cache: Path, revision: str, *, value: float = 1.0, sharded: bool = False) -> Path:
    """Create a per-file Hub cache: no refs/main or snapshot tree metadata."""
    snapshot = cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if sharded:
        filenames = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for index, filename in enumerate(filenames):
            save_file({f"weight{index}": torch.tensor([value + index])}, str(snapshot / filename))
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"weight{index}": filename for index, filename in enumerate(filenames)}})
        )
    else:
        save_file({"weight0": torch.tensor([value])}, str(snapshot / "model.safetensors"))
    return snapshot


def _write_ref(cache: Path, name: str, revision: str) -> None:
    ref = cache / "refs" / name
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)


@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize("sharded", [False, True])
def test_pinned_weights_from_commit_only_offline_cache(offline_cache, wrapper, sharded):
    snapshot = _write_snapshot(offline_cache, _REVISION, sharded=sharded)
    pretrained = wrapper.from_pretrained(_REPO_ID, revision=_REVISION)

    assert pretrained.state.source.path == snapshot
    torch.testing.assert_close(pretrained.state["weight0"], torch.tensor([1.0]))
    if sharded:
        torch.testing.assert_close(pretrained.state["weight1"], torch.tensor([2.0]))
    assert not (offline_cache / "refs" / "main").exists()


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_pinned_weights_do_not_use_different_cached_main(offline_cache, wrapper):
    snapshot = _write_snapshot(offline_cache, _REVISION)
    _write_snapshot(offline_cache, _OTHER_REVISION, value=99.0)
    _write_ref(offline_cache, "main", _OTHER_REVISION)

    pretrained = wrapper.from_pretrained(_REPO_ID, revision=_REVISION)

    assert pretrained.state.source.path == snapshot
    torch.testing.assert_close(pretrained.state["weight0"], torch.tensor([1.0]))


@pytest.mark.parametrize("revision", ["release-tag", "refs/pr/123", None])
def test_cached_named_revision_and_default(offline_cache, revision):
    snapshot = _write_snapshot(offline_cache, _REVISION)
    _write_ref(offline_cache, revision or "main", _REVISION)

    pretrained = PreTrainedCausalLM.from_pretrained(_REPO_ID, revision=revision)

    assert pretrained.state.source.path == snapshot
    torch.testing.assert_close(pretrained.state["weight0"], torch.tensor([1.0]))


def test_missing_pinned_snapshot_does_not_fall_back_to_main(offline_cache):
    _write_snapshot(offline_cache, _OTHER_REVISION, value=99.0)
    _write_ref(offline_cache, "main", _OTHER_REVISION)
    pretrained = PreTrainedCausalLM.from_pretrained(_REPO_ID, revision=_REVISION)

    with pytest.raises(LocalEntryNotFoundError):
        _ = pretrained.state["weight0"]


def test_incomplete_pinned_snapshot_does_not_fall_back_to_main(offline_cache):
    snapshot = _write_snapshot(offline_cache, _REVISION, sharded=True)
    (snapshot / "model-00002-of-00002.safetensors").unlink()
    _write_snapshot(offline_cache, _OTHER_REVISION, value=99.0, sharded=True)
    _write_ref(offline_cache, "main", _OTHER_REVISION)
    pretrained = PreTrainedCausalLM.from_pretrained(_REPO_ID, revision=_REVISION)

    with pytest.raises(KeyError, match="weight1"):
        _ = pretrained.state["weight1"]


@pytest.mark.parametrize("distributed_save", [False, True])
def test_export_uses_pinned_shard_map(offline_cache, tmp_path, distributed_save):
    _write_snapshot(offline_cache, _REVISION, sharded=True)
    _write_snapshot(offline_cache, _OTHER_REVISION, value=99.0)
    _write_ref(offline_cache, "main", _OTHER_REVISION)
    pretrained = PreTrainedCausalLM.from_pretrained(_REPO_ID, revision=_REVISION)
    output = tmp_path / "export"

    pretrained.state.source.save_generator(
        iter([("weight0", torch.tensor([3.0])), ("weight1", torch.tensor([4.0]))]),
        output,
        distributed_save=distributed_save,
    )

    assert (output / "model-00001-of-00002.safetensors").is_file()
    assert (output / "model-00002-of-00002.safetensors").is_file()
    tensors = SafeTensorsStateSource(output).load_tensors(["weight0", "weight1"])
    torch.testing.assert_close(tensors["weight0"], torch.tensor([3.0]))
    torch.testing.assert_close(tensors["weight1"], torch.tensor([4.0]))


@pytest.mark.parametrize("wrapper", _WRAPPERS)
def test_loaded_model_takes_precedence_over_pinned_hub_source(wrapper):
    pretrained = wrapper.from_pretrained(_REPO_ID, revision=_REVISION)
    pretrained.model = Mock()
    pretrained.model.state_dict.return_value = {"weight0": torch.tensor([7.0])}

    with patch("huggingface_hub.snapshot_download") as download:
        torch.testing.assert_close(pretrained.state["weight0"], torch.tensor([7.0]))

    download.assert_not_called()
