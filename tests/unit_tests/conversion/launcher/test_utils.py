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


pytestmark = pytest.mark.unit

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts" / "conversion"
spec = importlib.util.spec_from_file_location("conversion_utils_under_test", SCRIPT_DIR / "utils.py")
assert spec is not None and spec.loader is not None
conversion_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversion_utils)
prepare_output_directory = conversion_utils.prepare_output_directory
resolve_hf_commit_revision = conversion_utils.resolve_hf_commit_revision
resolve_hf_model_revision = conversion_utils.resolve_hf_model_revision


def test_resolve_hf_model_revision_downloads_exact_snapshot(monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs) or "/cache/models--hf--model/snapshots/0123456789abcdef",
    )

    resolved = resolve_hf_model_revision("hf/model", "0123456789abcdef")

    assert resolved == "/cache/models--hf--model/snapshots/0123456789abcdef"
    assert calls == [
        {
            "repo_id": "hf/model",
            "revision": "0123456789abcdef",
            "local_files_only": False,
            "allow_patterns": None,
        }
    ]


def test_config_only_revision_never_downloads_weights(monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kwargs: calls.append(kwargs) or "/cache/config")
    assert resolve_hf_model_revision("hf/model", "pinned", config_only=True) == "/cache/config"
    assert calls == [
        {
            "repo_id": "hf/model",
            "revision": "pinned",
            "allow_patterns": ["config.json", "*.py"],
            "local_files_only": False,
        }
    ]


@pytest.mark.parametrize("config_only", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_offline_revision_uses_only_cached_snapshot(tmp_path, monkeypatch, cached, config_only):
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import LocalEntryNotFoundError

    revision = "a" * 40
    snapshot = tmp_path / "models--example--model" / "snapshots" / revision
    if cached:
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)

    def reject_network(*args, **kwargs):
        pytest.fail("Offline conversion attempted a Hub API request")

    monkeypatch.setattr(HfApi, "repo_info", reject_network)
    monkeypatch.setattr(HfApi, "list_repo_tree", reject_network)
    if cached:
        assert resolve_hf_model_revision("example/model", revision, config_only=config_only) == str(snapshot)
    else:
        with pytest.raises(LocalEntryNotFoundError):
            resolve_hf_model_revision("example/model", revision, config_only=config_only)


def test_unpinned_reference_is_not_resolved(monkeypatch):
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **_kwargs: pytest.fail("An unpinned reference must remain unchanged"),
    )
    assert resolve_hf_model_revision("example/model", None) == "example/model"


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
