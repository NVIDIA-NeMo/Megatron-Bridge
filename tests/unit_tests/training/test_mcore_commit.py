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

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_commit_file(name: str) -> str:
    path = REPO_ROOT / name
    assert path.exists(), f"{name} not found at {path}"
    return path.read_text().strip()


def _get_submodule_commit() -> str:
    result = subprocess.run(
        ["git", "ls-tree", "HEAD", "3rdparty/Megatron-LM"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"git ls-tree failed: {result.stderr}"
    # output format: "<mode> commit <sha>\t<path>"
    return result.stdout.split()[2]


class TestMCoreCommitFiles:
    def test_main_commit_file_exists(self):
        assert (REPO_ROOT / ".main.commit").is_file()

    def test_dev_commit_file_exists(self):
        assert (REPO_ROOT / ".dev.commit").is_file()

    def test_main_commit_is_valid_sha(self):
        sha = _read_commit_file(".main.commit")
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    def test_dev_commit_is_valid_sha(self):
        sha = _read_commit_file(".dev.commit")
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)

    def test_main_commit_matches_submodule(self):
        """The .main.commit file must stay in sync with the pinned 3rdparty/Megatron-LM submodule."""
        main_commit = _read_commit_file(".main.commit")
        submodule_commit = _get_submodule_commit()
        assert main_commit == submodule_commit, (
            f".main.commit ({main_commit[:12]}) does not match "
            f"3rdparty/Megatron-LM submodule ({submodule_commit[:12]}). "
            f"Run: echo {submodule_commit} > .main.commit"
        )
