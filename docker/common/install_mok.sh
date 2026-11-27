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

set -euxo pipefail

MOK_REVISION="${1:?Usage: install_mok.sh <git commit> <wheel directory>}"
WHEEL_DIR=$(realpath -m "${2:?Usage: install_mok.sh <git commit> <wheel directory>}")
PYTHON="${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python"

if [[ ! "${MOK_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "MOK_COMMIT must be a full Git commit" >&2
    exit 1
fi

# Each native CPU wheel contains Blackwell and Rubin kernels by default.
export MOK_ARCH="${MOK_ARCH:-ALL}"
case "${MOK_ARCH}" in
    ALL) MOK_CUBINS=(sm_100a sm_103a sm_107a) ;;
    SM100) MOK_CUBINS=(sm_100a) ;;
    SM103) MOK_CUBINS=(sm_103a) ;;
    SM107) MOK_CUBINS=(sm_107a) ;;
    *)
        echo "Unsupported MOK_ARCH '${MOK_ARCH}'; expected ALL, SM100, SM103 or SM107" >&2
        exit 1
        ;;
esac

export NVCC="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
export MOK_NVCC="${NVCC}"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

# Some toolkits list only base names (sm_100), even when sm_100a is supported.
# Compile the exact gencode pairs used by MoK instead of trusting that listing.
PROBE_DIR="${WORK_DIR}/compiler-probe"
mkdir "${PROBE_DIR}"
cat > "${PROBE_DIR}/arch_probe.cu" <<'CUDA'
extern "C" __global__ void mok_arch_probe() {}
CUDA
for MOK_CUBIN in "${MOK_CUBINS[@]}"; do
    if ! "${NVCC}" -cubin -gencode "arch=compute_${MOK_CUBIN#sm_},code=${MOK_CUBIN}" \
        "${PROBE_DIR}/arch_probe.cu" -o "${PROBE_DIR}/${MOK_CUBIN}.cubin" \
        || [[ ! -s "${PROBE_DIR}/${MOK_CUBIN}.cubin" ]]; then
        echo "${NVCC} cannot build ${MOK_CUBIN}; use a base toolkit supporting every requested MoK target" >&2
        exit 1
    fi
done
rm -rf "${PROBE_DIR}"

git init "${WORK_DIR}"
git -C "${WORK_DIR}" remote add origin https://github.com/QiZhangNV/mixture-of-kittens.git
git -C "${WORK_DIR}" fetch --depth 1 origin "${MOK_REVISION}"
git -C "${WORK_DIR}" checkout --detach "${MOK_REVISION}"
bash "${WORK_DIR}/scripts/prepare_thunderkittens.sh"

# Docker builds have no GPU driver. Stubs are link inputs, never runtime libraries.
export LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
uv build --python "${PYTHON}" --wheel --no-build-isolation --no-cache --out-dir "${WHEEL_DIR}" "${WORK_DIR}"
uv pip install --python "${PYTHON}" --no-deps --no-cache "${WHEEL_DIR}"/*.whl

# Inspect the installed wheel without loading libcuda on GPU-less image builders.
MOK_EXTENSION=$(uv run --no-project --no-sync --python "${PYTHON}" python - <<'PY'
import sys
from importlib.metadata import distribution
from sysconfig import get_config_var

extension = distribution("mixture-of-kittens").locate_file("mok/_C" + get_config_var("EXT_SUFFIX"))
assert extension.is_file(), f"MoK native extension missing: {extension}"
sys.stdout.write(str(extension))
PY
)
"${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump" --list-elf "${MOK_EXTENSION}" > "${WORK_DIR}/cubins.txt"
cat "${WORK_DIR}/cubins.txt"
for MOK_CUBIN in "${MOK_CUBINS[@]}"; do
    grep -Fq ".${MOK_CUBIN}.cubin" "${WORK_DIR}/cubins.txt"
done
