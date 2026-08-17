#!/usr/bin/env bash
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

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DATACOMP_ROOT" >&2
    exit 2
fi

datacomp_root=$1
datacomp_env="${datacomp_root}/downloader-env"
datacomp_upstream="${datacomp_root}/datacomp-upstream"
datacomp_revision=4a8df1992566ef8334773f7152e1855b1f716162
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${datacomp_root}"
if [[ ! -d "${datacomp_upstream}/.git" ]]; then
    git clone https://github.com/mlfoundations/datacomp "${datacomp_upstream}"
fi
git -C "${datacomp_upstream}" fetch origin "${datacomp_revision}"
git -C "${datacomp_upstream}" checkout --detach "${datacomp_revision}"
test "$(git -C "${datacomp_upstream}" rev-parse HEAD)" = "${datacomp_revision}"

uv venv --python 3.10 "${datacomp_env}"
uv pip install \
    --python "${datacomp_env}/bin/python" \
    --requirements "${script_dir}/download_requirements.txt"

# DataComp pins both OpenCV distributions. Keep the exact headless wheel and
# remove the GUI package so download workers do not require libGL.
uv pip uninstall --python "${datacomp_env}/bin/python" opencv-python
uv pip install \
    --python "${datacomp_env}/bin/python" \
    --reinstall-package opencv-python-headless \
    opencv-python-headless==4.6.0.66

uv run --no-project --python "${datacomp_env}/bin/python" python - <<'PY'
from importlib.metadata import version

import cv2

assert cv2.__version__ == "4.6.0"
assert version("img2dataset") == "1.40.0"
PY

uv pip freeze --python "${datacomp_env}/bin/python" > "${datacomp_root}/downloader-environment.txt"
