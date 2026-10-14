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

"""Tests for scripts/performance/utils/nvcre_bridge.py — running the launcher's checkout in NVCRE pods."""

import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from utils import nvcre_bridge  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git checkout with an origin remote, like a real Megatron-Bridge clone."""
    repo = tmp_path / "bridge"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", "https://github.com/example/Megatron-Bridge.git")
    (repo / "README").write_text("x")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.mark.parametrize(
    "remote, expected",
    [
        ("https://github.com/NVIDIA-NeMo/Megatron-Bridge.git", "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"),
        ("git@github.com:NVIDIA-NeMo/Megatron-Bridge.git", "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"),
        ("ssh://git@github.com/NVIDIA-NeMo/Megatron-Bridge.git", "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"),
        # the port of an ssh:// URL is the SSH port; it must not leak into the HTTPS URL or its path
        ("ssh://git@gitlab.example.com:2222/o/r.git", "https://gitlab.example.com/o/r.git"),
        ("git@gitlab.example.com:o/sub/r.git", "https://gitlab.example.com/o/sub/r.git"),
        # credentials must never end up in the WorkloadRun spec
        ("https://user:secret-token@github.com/o/r.git", "https://github.com/o/r.git"),
        ("https://gitlab.example.com:8443/o/r.git", "https://gitlab.example.com:8443/o/r.git"),
    ],
)
def test_pod_clone_url(remote: str, expected: str) -> None:
    assert nvcre_bridge.pod_clone_url(remote) == expected


@pytest.mark.parametrize("remote", ["/srv/git/bridge.git", "file:///srv/git/bridge.git", "../bridge"])
def test_pod_clone_url_rejects_remotes_a_pod_cannot_reach(remote: str) -> None:
    with pytest.raises(ValueError, match="clone URL"):
        nvcre_bridge.pod_clone_url(remote)


def test_local_bridge_source_reads_origin_and_head(checkout: Path) -> None:
    source = nvcre_bridge.local_bridge_source(checkout)

    assert source.url == "https://github.com/example/Megatron-Bridge.git"
    assert source.commit == _git(checkout, "rev-parse", "HEAD")


def test_local_bridge_source_requires_a_git_checkout_with_origin(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="origin"):
        nvcre_bridge.local_bridge_source(tmp_path)


def test_local_bridge_source_rejects_uncommitted_changes(checkout: Path) -> None:
    (checkout / "README").write_text("changed")

    with pytest.raises(RuntimeError, match="uncommitted"):
        nvcre_bridge.local_bridge_source(checkout)


def test_local_bridge_source_rejects_staged_changes(checkout: Path) -> None:
    (checkout / "README").write_text("changed")
    _git(checkout, "add", "README")

    with pytest.raises(RuntimeError, match="uncommitted"):
        nvcre_bridge.local_bridge_source(checkout)


def test_local_bridge_source_ignores_untracked_files(checkout: Path) -> None:
    (checkout / "scratch.txt").write_text("not tracked")

    assert nvcre_bridge.local_bridge_source(checkout).commit == _git(checkout, "rev-parse", "HEAD")


def test_local_bridge_source_ignores_a_drifted_submodule(checkout: Path, tmp_path: Path) -> None:
    """Only this repository's commit is fetched, so a submodule checked out at another commit is not a dirty tree."""
    sub_repo = tmp_path / "sub_repo"
    sub_repo.mkdir()
    _git(sub_repo, "init", "-q")
    _git(sub_repo, "config", "user.email", "t@example.com")
    _git(sub_repo, "config", "user.name", "t")
    (sub_repo / "f").write_text("1")
    _git(sub_repo, "add", "f")
    _git(sub_repo, "commit", "-q", "-m", "one")
    recorded = _git(sub_repo, "rev-parse", "HEAD")
    _git(checkout, "update-index", "--add", "--cacheinfo", f"160000,{recorded},3rdparty/sub")
    _git(checkout, "commit", "-q", "-m", "add submodule")
    _git(checkout, "clone", "-q", str(sub_repo), "3rdparty/sub")
    (checkout / "3rdparty" / "sub" / "f").write_text("2")
    _git(
        checkout / "3rdparty" / "sub",
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "-qam",
        "two",
    )
    assert "3rdparty/sub" in _git(checkout, "status", "--porcelain", "--untracked-files=no")  # git sees drift

    assert nvcre_bridge.local_bridge_source(checkout).commit == _git(checkout, "rev-parse", "HEAD")


def test_bridge_init_container_fetches_the_exact_commit() -> None:
    source = nvcre_bridge.LocalBridge("https://github.com/o/r.git", "a" * 40)

    container = nvcre_bridge.bridge_init_container(source, "nvcr.io/nvidia/nemo:26.08.01")

    assert container["name"] == "megatron-bridge-clone"
    assert container["image"] == "nvcr.io/nvidia/nemo:26.08.01"  # the training image, which has git
    assert container["volumeMounts"] == [nvcre_bridge.WORKSPACE_MOUNT]
    script = container["args"][0]
    assert f"git fetch -q --depth 1 origin {'a' * 40}" in script
    assert "git remote add origin https://github.com/o/r.git" in script
    assert "submodule" not in script  # only the Megatron-Bridge commit is fetched
    assert nvcre_bridge.BRIDGE_POD_ROOT in script


def test_bridge_init_container_quotes_the_url() -> None:
    source = nvcre_bridge.LocalBridge("https://github.com/o/r.git; rm -rf /", "a" * 40)

    assert "'https://github.com/o/r.git; rm -rf /'" in nvcre_bridge.bridge_init_container(source, "img")["args"][0]


def test_workspace_volume_and_mount_agree() -> None:
    assert nvcre_bridge.WORKSPACE_VOLUME["name"] == nvcre_bridge.WORKSPACE_MOUNT["name"]
    assert nvcre_bridge.BRIDGE_POD_ROOT.startswith(nvcre_bridge.WORKSPACE_MOUNT["mountPath"] + "/")
