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
datacomp_metadata="${datacomp_root}/raw/metadata"
datacomp_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

test -x "${datacomp_env}/bin/python"
test "$(git -C "${datacomp_upstream}" rev-parse HEAD)" = 4a8df1992566ef8334773f7152e1855b1f716162

uv run --no-project --python "${datacomp_env}/bin/python" \
    python "${datacomp_script_dir}/download_metadata.py" \
    --output-dir "${datacomp_metadata}"

uv run --no-project --python "${datacomp_env}/bin/python" \
    python "${datacomp_upstream}/download_upstream.py" \
    --scale datacomp_1b \
    --data_dir "${datacomp_root}/raw" \
    --metadata_dir "${datacomp_metadata}" \
    --processes_count 16 \
    --thread_count 128 \
    --image_size 512 \
    --resize_mode keep_ratio_largest \
    --encode_format jpg \
    --output_format webdataset \
    --retries 2
