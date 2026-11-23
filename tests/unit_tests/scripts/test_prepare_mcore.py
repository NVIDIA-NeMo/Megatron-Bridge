"""Exercise MCore preparation with real Git checkouts and local changes."""

import os
import subprocess
from pathlib import Path

import pytest


G_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/prepare_mcore.sh"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def mcore(tmp_path: Path) -> Path:
    repo = tmp_path / "MCore"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    for name in ("megatron/core/__init__.py", "megatron/training/__init__.py", "megatron/bridge/__init__.py"):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n")
    (repo / ".gitignore").write_text("/build/\n/*.egg-info/\n__pycache__/\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Fixture")
    return repo


def prepare(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(G_SCRIPT), str(repo)], text=True, capture_output=True)


@pytest.mark.unit
def test_prepare_preserves_pin_sources_and_clean_status(mcore: Path) -> None:
    head = git(mcore, "rev-parse", "HEAD")
    for _ in range(2):
        result = prepare(mcore)
        assert result.returncode == 0, result.stderr
        assert git(mcore, "rev-parse", "HEAD") == head
        assert git(mcore, "status", "--porcelain") == ""
        assert not (mcore / "megatron/bridge").exists()
        assert (mcore / "megatron/core/__init__.py").read_text() == "# source\n"
        assert (mcore / "megatron/training/__init__.py").is_file()


@pytest.mark.unit
def test_prepare_cleans_stale_package_output(mcore: Path) -> None:
    for name in ("build/lib/megatron/bridge/__init__.py", "megatron_core.egg-info/SOURCES.txt"):
        path = mcore / name
        path.parent.mkdir(parents=True)
        path.write_text("stale upstream Bridge\n")
    result = prepare(mcore)
    assert result.returncode == 0, result.stderr
    assert not (mcore / "build").exists()
    assert not (mcore / "megatron_core.egg-info").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "name", ["megatron/bridge/__init__.py", "megatron/bridge/local.py", "megatron/bridge/__pycache__/local.pyc"]
)
def test_prepare_preserves_local_source(mcore: Path, name: str) -> None:
    path = mcore / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("local work\n")
    result = prepare(mcore)
    assert result.returncode != 0
    assert path.read_text() == "local work\n"
    assert (mcore / "megatron/bridge/__init__.py").exists()


@pytest.mark.unit
def test_prepare_preserves_unexpected_build_files(mcore: Path) -> None:
    (mcore / ".gitignore").write_text("")
    git(mcore, "commit", "-qam", "Do not ignore build")
    path = mcore / "build/local.txt"
    path.parent.mkdir()
    path.write_text("local work\n")
    result = prepare(mcore)
    assert result.returncode != 0
    assert path.read_text() == "local work\n"


@pytest.mark.unit
def test_prepare_preserves_build_symlink_target(mcore: Path, tmp_path: Path) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("local work\n")
    (mcore / "build").symlink_to(target)
    assert prepare(mcore).returncode != 0
    assert target.read_text() == "local work\n"


@pytest.mark.unit
def test_submodule_updates_keep_exclusion(mcore: Path, tmp_path: Path) -> None:
    first = git(mcore, "rev-parse", "HEAD")
    (mcore / "megatron/bridge/new.py").write_text("# new upstream source\n")
    git(mcore, "add", ".")
    git(mcore, "commit", "-qm", "Next pin")
    second = git(mcore, "rev-parse", "HEAD")
    parent = tmp_path / "parent"
    parent.mkdir()
    git(parent, "init", "-q")
    git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(mcore), "mcore")
    child = parent / "mcore"
    assert prepare(child).returncode == 0
    for commit in (first, second):
        git(parent, "update-index", "--cacheinfo", f"160000,{commit},mcore")
        git(parent, "submodule", "update", "--init", "--no-fetch")
        assert git(child, "rev-parse", "HEAD") == commit
        assert not (child / "megatron/bridge").exists()
        assert (child / "megatron/core/__init__.py").is_file()


@pytest.mark.unit
def test_preparing_linked_worktree_does_not_filter_other_checkout(mcore: Path, tmp_path: Path) -> None:
    linked = tmp_path / "linked"
    git(mcore, "worktree", "add", "--detach", str(linked))
    # Older Git may otherwise inherit a globally selected template directory.
    assert os.path.exists(linked / ".git")
    result = prepare(linked)
    assert result.returncode == 0, result.stderr
    assert not (linked / "megatron/bridge").exists()
    assert (mcore / "megatron/bridge/__init__.py").is_file()
    assert git(mcore, "status", "--porcelain") == ""
