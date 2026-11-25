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

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.unit

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts" / "conversion"
spec = importlib.util.spec_from_file_location("conversion_utils_under_test", SCRIPT_DIR / "utils.py")
assert spec is not None and spec.loader is not None
conversion_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion_utils)
prepare_output_directory = conversion_utils.prepare_output_directory
resolve_hf_commit_revision = conversion_utils.resolve_hf_commit_revision
resolve_hf_model_revision = conversion_utils.resolve_hf_model_revision
_configure_model_config = conversion_utils._configure_model_config
_configure_model_provider = conversion_utils._configure_model_provider


def test_resolve_hf_model_revision_downloads_exact_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or "/cache/models--hf--model/snapshots/0123456789abcdef",
    )

    resolved = resolve_hf_model_revision("hf/model", "0123456789abcdef")

    assert resolved == "/cache/models--hf--model/snapshots/0123456789abcdef"
    assert calls == [{"repo_id": "hf/model", "revision": "0123456789abcdef"}]


def test_config_only_revision_never_downloads_weights(monkeypatch):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs) or "/cache/config")
    assert resolve_hf_model_revision("hf/model", "pinned", config_only=True) == "/cache/config"
    assert calls == [{"repo_id": "hf/model", "revision": "pinned", "allow_patterns": ["config.json", "*.py"]}]


def test_resolve_hf_commit_revision_resolves_named_ref_without_downloading(monkeypatch):
    calls = []

    def fake_model_info(_self, *, repo_id, revision):
        calls.append({"repo_id": repo_id, "revision": revision})
        return SimpleNamespace(sha="0123456789abcdef0123456789abcdef01234567")  # pragma: allowlist secret

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", fake_model_info)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: pytest.fail("import revision resolution must not download a snapshot"),
    )

    resolved = resolve_hf_commit_revision("hf/model", "release-tag")

    assert resolved == "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    assert calls == [{"repo_id": "hf/model", "revision": "release-tag"}]


def test_resolve_hf_commit_revision_preserves_immutable_sha_without_network(monkeypatch):
    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info",
        lambda *_args, **_kwargs: pytest.fail("an immutable commit SHA must not require Hub metadata"),
    )

    assert resolve_hf_commit_revision("hf/model", revision) == revision


@pytest.mark.parametrize("resolver", [resolve_hf_commit_revision, resolve_hf_model_revision])
def test_hf_revision_resolvers_reject_local_path(tmp_path, resolver):
    with pytest.raises(ValueError, match="only to Hugging Face Hub model IDs"):
        resolver(str(tmp_path), "0123456789abcdef")


@pytest.mark.parametrize("relationship", ["equal", "destination-parent", "destination-child"])
def test_prepare_output_directory_rejects_source_overlap(tmp_path, relationship):
    source = tmp_path / "source"
    source.mkdir()
    source_marker = source / "checkpoint"
    source_marker.write_text("checkpoint")

    if relationship == "equal":
        destination = source
    elif relationship == "destination-parent":
        destination = tmp_path
    else:
        destination = source / "export"
        destination.mkdir()
        (destination / "partial-output").write_text("partial")

    with pytest.raises(ValueError, match="overlaps conversion source"):
        prepare_output_directory(str(destination), overwrite=True, source_paths=[str(source)])

    assert source_marker.read_text() == "checkpoint"


@pytest.mark.parametrize(
    ("vocab_size", "tp", "padded"),
    [(127, 2, True), (151936, 3, True), (128, 2, False), (32000, 8, False), (50257, 1, False)],
)
def test_configure_model_provider_pads_only_a_vocabulary_tp_does_not_divide(vocab_size, tp, padded):
    provider = SimpleNamespace(vocab_size=vocab_size, should_pad_vocab=False)

    _configure_model_provider(provider, tp=tp, pp=1, ep=1, etp=1, dtype=torch.bfloat16)

    assert provider.should_pad_vocab is padded


def test_configure_model_provider_keeps_requested_padding():
    provider = SimpleNamespace(vocab_size=128, should_pad_vocab=True)

    _configure_model_provider(provider, tp=2, pp=1, ep=1, etp=1, dtype=torch.bfloat16)

    assert provider.should_pad_vocab is True


def test_configure_model_config_pads_the_builder_vocabulary():
    model_config = SimpleNamespace(vocab_size=127, should_pad_vocab=False, transformer=SimpleNamespace())

    _configure_model_config(model_config, tp=2, pp=1, ep=1, etp=1, dtype=torch.bfloat16)

    assert model_config.should_pad_vocab is True
    assert model_config.transformer.tensor_model_parallel_size == 2
