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

"""Candidate-tree link checks need no model runtime or network access."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/docs/check_repository_links.py"
SPEC = importlib.util.spec_from_file_location("check_repository_links", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)
REPOSITORY = "NVIDIA-NeMo/Megatron-Bridge"
PREFIX = f"https://github.com/{REPOSITORY}"
pytestmark = pytest.mark.unit


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--no-gpg-sign", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-b", "main")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/retired.py").write_text("# Retired model\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/historical.md").write_text(f"[Model]({PREFIX}/blob/main/src/retired.py)\n")
    commit(tmp_path, "base")
    git(tmp_path, "tag", "v0.4.2")
    return tmp_path


def check(repo: Path, links: list[str], revision: str = "HEAD") -> tuple[list[str], list[str]]:
    return checker.check_links(links, tree=checker.git_tree(repo, revision), repository=REPOSITORY, branch="main")


def test_deletion_breaks_unchanged_historical_referer(repo: Path) -> None:
    links = [f"{PREFIX}/blob/main/src/retired.py", f"{PREFIX}/tree/main/src"]
    base = git(repo, "rev-parse", "HEAD")
    assert check(repo, links) == ([], [])
    (repo / "src/retired.py").unlink()
    commit(repo, "remove model, leave historical doc unchanged")
    assert git(repo, "diff", base, "--", "docs") == ""
    external, errors = check(repo, links)
    assert external == []
    assert len(errors) == 2  # File and now-empty directory are gone from Git.
    assert all("missing" in error for error in errors)
    assert check(repo, links, base) == ([], [])
    # Removing the referer or pinning its URL avoids candidate-tree failures.
    assert check(repo, []) == ([], [])
    pinned = [f"{PREFIX}/blob/v0.4.2/src/retired.py"]
    assert check(repo, pinned) == (pinned, [])
    assert checker.git_tree(repo, "v0.4.2")["src/retired.py"] == "blob"


def test_added_target_passes_before_merge_and_missing_target_fails(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    link = f"{PREFIX}/blob/main/src/new.py"
    (repo / "src/new.py").write_text("# New model\n")
    # An untracked file must not make the checked Git tree pass.
    assert len(check(repo, [link])[1]) == 1
    commit(repo, "new model")
    assert check(repo, [link]) == ([], [])
    assert len(check(repo, [link], base)[1]) == 1
    assert len(check(repo, [f"{PREFIX}/blob/main/src/missing.py"])[1]) == 1


@pytest.mark.parametrize(
    "url",
    [
        f"{PREFIX}/blob/main/src/retired.py#L1",
        f"{PREFIX}/blob/main/src/retired.py/",
        f"{PREFIX}/tree/main/src/retired.py?plain=1#L1",
        f"{PREFIX}/tree/main/src/",
        f"{PREFIX}/blob/main/src/",
        f"{PREFIX}/tree/main",
        f"https://raw.githubusercontent.com/{REPOSITORY}/main/src/retired.py",
        f"https://raw.githubusercontent.com/{REPOSITORY}/refs/heads/main/src/retired.py",
        f"http://github.com/{REPOSITORY.lower()}/blob/main/src/retired.py",
        f"https://github.com:443/{REPOSITORY}/blob/main/src/retired.py",
        f"http://github.com:80/{REPOSITORY}/blob/main/src/retired.py",
    ],
)
def test_supported_github_and_raw_urls(repo: Path, url: str) -> None:
    assert check(repo, [url]) == ([], [])


def test_raw_url_requires_file(repo: Path) -> None:
    assert len(check(repo, [f"https://raw.githubusercontent.com/{REPOSITORY}/main/src"])[1]) == 1


@pytest.mark.parametrize("suffix", ["/", "%2F"])
def test_raw_file_trailing_slash_fails(repo: Path, suffix: str) -> None:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/main/src/retired.py{suffix}"
    external, errors = check(repo, [url])
    assert external == []
    assert len(errors) == 1
    assert "missing" in errors[0]


@pytest.mark.parametrize(
    "url",
    [
        f"{PREFIX}/blob/v0.4.2/src/retired.py",
        f"{PREFIX}/blob/abcdef0123456789/src/retired.py",
        f"{PREFIX}/blob/release/src/retired.py",
        f"{PREFIX}/blob/main-other/src/retired.py",
        f"{PREFIX}/blob/main%2Fother/src/retired.py",
        f"https://raw.githubusercontent.com/{REPOSITORY}/main%2fother/src/retired.py",
        f"{PREFIX}/issues/1",
        f"{PREFIX}/stargazers",
        "https://github.com/another/repo/blob/main/missing.py",
        f"https://example.com/{REPOSITORY}/blob/main/missing.py",
        f"https://github.com.test.example/{REPOSITORY}/blob/main/missing.py",
        f"https://github.com:8443/{REPOSITORY}/blob/main/src/retired.py",
        f"https://raw.githubusercontent.com:8443/{REPOSITORY}/main/src/retired.py",
        f"https://github.com:80/{REPOSITORY}/blob/main/src/retired.py",
        f"http://github.com:443/{REPOSITORY}/blob/main/src/retired.py",
        f"https://user@github.com/{REPOSITORY}/blob/main/src/retired.py",
    ],
)
def test_explicit_versions_and_external_urls_remain_online(repo: Path, url: str) -> None:
    assert check(repo, [url]) == ([url], [])


def test_encoded_paths_and_branch_names() -> None:
    target = checker.repository_target(
        f"{PREFIX}/blob/release%2Fnext/docs/a%20b%23c.md#heading", repository=REPOSITORY, branch="release/next"
    )
    assert target == ("object", "docs/a b#c.md")
    assert checker.repository_target(f"{PREFIX}/blob/main/docs/a%2520b.md", repository=REPOSITORY, branch="main") == (
        "object",
        "docs/a%20b.md",
    )


def test_shallow_clone_and_fork_remote_do_not_affect_resolution(repo: Path, tmp_path: Path) -> None:
    # Two commits prove that the clone really lacks parents, not just has depth=1 set.
    (repo / "src/new.py").write_text("# New model\n")
    commit(repo, "add model")
    clone = tmp_path / "fork"
    subprocess.run(["git", "clone", "--depth=1", repo.as_uri(), str(clone)], check=True, capture_output=True)
    git(clone, "remote", "set-url", "origin", "https://github.com/contributor/fork.git")
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"
    assert check(clone, [f"{PREFIX}/blob/main/src/new.py"]) == ([], [])
    assert len(check(clone, [f"{PREFIX}/blob/main/src/missing.py"])[1]) == 1


def test_cli_writes_only_external_urls_and_fails_empty_scan(repo: Path, tmp_path: Path) -> None:
    import sys

    links = tmp_path / "links.txt"
    external = tmp_path / "external.txt"
    links.write_text(f"{PREFIX}/blob/main/src/retired.py\nhttps://example.com/\n")
    command = [
        sys.executable,
        str(SCRIPT),
        "--repo-root",
        str(repo),
        "--repository",
        REPOSITORY,
        "--branch",
        "main",
        "--links-file",
        str(links),
        "--external-links-file",
        str(external),
    ]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert external.read_text() == "https://example.com/\n"
    links.write_text(f"{PREFIX}/blob/main/src/missing.py\n")
    assert subprocess.run(command, capture_output=True).returncode == 1
    links.write_text("")
    assert subprocess.run(command, capture_output=True).returncode == 1


def test_invalid_revision_fails_instead_of_falling_back(repo: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        checker.git_tree(repo, "unavailable-revision")
