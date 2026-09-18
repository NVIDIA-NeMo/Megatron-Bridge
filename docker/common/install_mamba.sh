#!/usr/bin/env bash
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

# uv sync installs Mamba's dependencies but skips Mamba itself. Build the locked
# sdist here so its C++ and NVCC flags inherit PyTorch's required C++ standard.
set -euo pipefail

lock_file="${1:?Usage: install_mamba.sh UV_LOCK MAMBA_PATCH}"
patch_file="${2:?Usage: install_mamba.sh UV_LOCK MAMBA_PATCH}"

# Read the resolved lock after sync, including when dispatched MCore updates it.
# Fail on a new version/source rather than silently installing a different pin.
sdist_info="$(uv run --no-project --no-sync python - "$lock_file" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as lock_file:
    packages = tomllib.load(lock_file)["package"]
matches = [package for package in packages if package["name"] == "mamba-ssm"]
if len(matches) != 1:
    raise RuntimeError("Expected exactly one locked mamba-ssm package")
package = matches[0]
if package["version"] != "2.3.1" or package["source"] != {"registry": "https://pypi.org/simple"}:
    raise RuntimeError("Mamba source/version changed; update docker/patches/mamba.patch before building")
sdist = package["sdist"]
algorithm, digest = sdist["hash"].split(":", 1)
if algorithm != "sha256" or len(digest) != 64:
    raise RuntimeError("Expected a SHA-256 digest for the Mamba sdist")
sys.stdout.write(f"{sdist['url']} {digest}\n")
PY
)"
read -r sdist_url sdist_sha256 <<< "$sdist_info"

mamba_build_dir="$(mktemp -d)"
trap 'rm -rf -- "${mamba_build_dir:?}"' EXIT
curl --fail --location --retry 3 "$sdist_url" --output "$mamba_build_dir/mamba.tar.gz"
echo "$sdist_sha256  $mamba_build_dir/mamba.tar.gz" | sha256sum -c -
tar -xzf "$mamba_build_dir/mamba.tar.gz" -C "$mamba_build_dir"
patch --batch --forward --fuzz=0 -d "$mamba_build_dir/mamba_ssm-2.3.1" -p1 < "$patch_file"

# --no-deps preserves the environment resolved by uv sync, especially base-image
# torch. FORCE_BUILD prevents Mamba's setup.py from downloading an unpatched wheel.
MAMBA_FORCE_BUILD=TRUE uv pip install --no-build-isolation --no-deps --reinstall \
    "$mamba_build_dir/mamba_ssm-2.3.1"
