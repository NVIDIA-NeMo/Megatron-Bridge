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

from types import SimpleNamespace
from unittest.mock import MagicMock
from zipfile import ZipFile

from tests.functional_tests import fixture_utils
from tests.functional_tests.test_groups.data import download_unit_tests_dataset


def test_get_oldest_release_prefers_staged_assets(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged"
    output_root = tmp_path / "output"
    for relative_path in download_unit_tests_dataset.STAGED_RELEASE_ASSETS:
        staged_asset = staged_root / relative_path
        staged_asset.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(staged_asset, "w") as archive:
            archive.writestr(f"{staged_asset.stem}/fixture.txt", staged_asset.name)

    monkeypatch.setenv(fixture_utils.TEST_DATA_ROOT_ENV, str(staged_root))
    github = MagicMock(side_effect=AssertionError("GitHub fallback should not be used"))
    monkeypatch.setattr(download_unit_tests_dataset, "Github", github)

    download_unit_tests_dataset.get_oldest_release_and_assets(assets_dir=str(output_root))

    assert (output_root / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_root / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    github.assert_not_called()


def test_get_oldest_release_falls_back_without_github_token(monkeypatch, tmp_path):
    release = SimpleNamespace(
        tag_name="v2.5",
        title="Megatron-LM v2.5",
        created_at=0,
        published_at=0,
        draft=False,
        prerelease=False,
        html_url="https://github.com/NVIDIA/Megatron-LM/releases/tag/v2.5",
        body="",
        get_assets=lambda: [],
    )
    repo = SimpleNamespace(
        full_name="NVIDIA/Megatron-LM",
        description="Megatron-LM",
        html_url="https://github.com/NVIDIA/Megatron-LM",
        get_releases=lambda: [release],
    )
    github = MagicMock()
    github.return_value.get_repo.return_value = repo

    monkeypatch.setenv(fixture_utils.TEST_DATA_ROOT_ENV, str(tmp_path / "missing"))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(download_unit_tests_dataset, "Github", github)

    download_unit_tests_dataset.get_oldest_release_and_assets(assets_dir=str(tmp_path / "output"))

    github.assert_called_once_with()
