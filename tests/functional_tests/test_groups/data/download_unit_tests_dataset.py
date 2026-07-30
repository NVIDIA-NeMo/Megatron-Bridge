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

#!/usr/bin/env python3
"""
Script to fetch the oldest release of NVIDIA/Megatron-LM on GitHub and list its assets.
Uses the PyGithub SDK to interact with the GitHub API.
"""

import os
import sys
import tarfile
import zipfile
from pathlib import Path

import click
import requests
from github import Github

from tests.functional_tests.fixture_utils import get_test_data_root


DEFAULT_REPO_NAME = "NVIDIA/Megatron-LM"
STAGED_RELEASE_ASSETS = (
    Path("megatron-lm/release-assets/v2.5/datasets.zip"),
    Path("megatron-lm/release-assets/v2.5/tokenizers.zip"),
)


def extract_asset(asset_path: Path, assets_dir: Path) -> bool:
    """Extract a release asset into the writable test data directory."""
    try:
        print(f"  Extracting {asset_path.name} to {assets_dir}...")

        if asset_path.name.endswith(".zip"):
            with zipfile.ZipFile(asset_path, "r") as zip_ref:
                zip_ref.extractall(assets_dir)
        elif asset_path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(asset_path, "r:gz") as tar_ref:
                tar_ref.extractall(assets_dir)
        elif asset_path.name.endswith(".tar"):
            with tarfile.open(asset_path, "r") as tar_ref:
                tar_ref.extractall(assets_dir)
        else:
            print(f"  Warning: Unknown file type for {asset_path.name}, skipping extraction")
            return False

        print(f"  Successfully extracted to {assets_dir}")
        return True

    except Exception as e:
        print(f"  Error extracting {asset_path.name}: {e}")
        return False


def extract_staged_release_assets(repo_name: str, assets_dir: Path) -> bool:
    """Extract staged Megatron-LM v2.5 assets when all of them are available."""
    if repo_name != DEFAULT_REPO_NAME:
        return False

    staged_assets = tuple(get_test_data_root() / relative_path for relative_path in STAGED_RELEASE_ASSETS)
    if not all(asset_path.is_file() for asset_path in staged_assets):
        return False

    print(f"Using staged release assets from {staged_assets[0].parent}")
    return all(extract_asset(asset_path, assets_dir) for asset_path in staged_assets)


def download_and_extract_asset(asset_url: str, asset_name: str, assets_dir: Path) -> bool:
    """
    Download and extract an asset to the assets directory.

    Args:
        asset_url: URL to download the asset from
        asset_name: Name of the asset file
        assets_dir: Directory to extract the asset to

    Returns:
        bool: True if successful, False otherwise
    """
    temp_file = assets_dir / asset_name
    try:
        # Download the asset
        print(f"  Downloading {asset_name}...")
        response = requests.get(asset_url, stream=True, timeout=60)
        response.raise_for_status()

        # Save to temporary file
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return extract_asset(temp_file, assets_dir)

    except Exception as e:
        print(f"  Error downloading/extracting {asset_name}: {e}")
        return False
    finally:
        if temp_file.is_file():
            temp_file.unlink()


def get_oldest_release_and_assets(repo_name: str = DEFAULT_REPO_NAME, assets_dir: str = "assets") -> None:
    """
    Fetch the oldest release of a GitHub repository and list its assets.

    Args:
        repo_name: The repository name in format "owner/repo"
        assets_dir: Directory to extract assets to
    """
    try:
        assets_path = Path(assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)

        if extract_staged_release_assets(repo_name, assets_path):
            return

        # Initialize an authenticated GitHub client when a token is available.
        token = os.getenv("GH_TOKEN", None)
        g = Github(login_or_token=token) if token else Github()

        # Get the repository
        repo = g.get_repo(repo_name)
        print(f"Repository: {repo.full_name}")
        print(f"Description: {repo.description}")
        print(f"URL: {repo.html_url}")
        print("-" * 80)

        # Get all releases
        releases = list(repo.get_releases())

        if not releases:
            print("No releases found for this repository.")
            return

        # Sort releases by creation date to find the oldest
        releases.sort(key=lambda x: x.created_at)
        oldest_release = releases[0]

        print("Oldest Release:")
        print(f"  Tag: {oldest_release.tag_name}")
        print(f"  Title: {oldest_release.title}")
        print(f"  Created: {oldest_release.created_at}")
        print(f"  Published: {oldest_release.published_at}")
        print(f"  Draft: {oldest_release.draft}")
        print(f"  Prerelease: {oldest_release.prerelease}")
        print(f"  URL: {oldest_release.html_url}")

        if oldest_release.body:
            print(f"  Description: {oldest_release.body[:200]}...")

        print("-" * 80)

        # List assets
        assets = list(oldest_release.get_assets())

        if not assets:
            print("No assets found for this release.")
            return

        print(f"Assets ({len(assets)} total):")
        print("-" * 80)

        for i, asset in enumerate(assets, 1):
            print(f"{i}. {asset.name}")
            print(f"   Size: {asset.size} bytes ({asset.size / 1024 / 1024:.2f} MB)")
            print(f"   Downloads: {asset.download_count}")
            print(f"   Content Type: {asset.content_type}")
            print(f"   URL: {asset.browser_download_url}")
            print(f"   Created: {asset.created_at}")
            print(f"   Updated: {asset.updated_at}")
            print()

        # Summary
        total_size = sum(asset.size for asset in assets)
        total_downloads = sum(asset.download_count for asset in assets)

        print("Summary:")
        print(f"  Total assets: {len(assets)}")
        print(f"  Total size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")
        print(f"  Total downloads: {total_downloads}")

        # Download and extract assets if requested
        if assets:
            print("-" * 80)
            print("Downloading and extracting assets...")

            print(f"Created assets directory: {assets_path.absolute()}")

            successful_downloads = 0
            for asset in assets:
                print(f"\nProcessing asset: {asset.name}")
                if download_and_extract_asset(asset.browser_download_url, asset.name, assets_path):
                    successful_downloads += 1

            print("\nDownload Summary:")
            print(f"  Successfully downloaded and extracted: {successful_downloads}/{len(assets)} assets")
            print(f"  Assets directory: {assets_path.absolute()}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.option("--repo", default=DEFAULT_REPO_NAME, help="GitHub repository name (format: owner/repo)")
@click.option("--assets-dir", default="assets", help="Directory to extract assets to")
def main(repo, assets_dir):
    """Fetch the oldest release of a GitHub repository and download its assets."""
    print(f"Fetching oldest release of {repo}...")
    print("=" * 80)

    get_oldest_release_and_assets(repo_name=repo, assets_dir=assets_dir)


if __name__ == "__main__":
    main()
