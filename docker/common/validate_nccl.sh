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

#!/bin/bash

set -euo pipefail

EXPECTED_VERSION="${1:?expected NCCL package version is required}"
RUNTIME_VERSION=$(dpkg-query -W -f='${Version}' libnccl2)
DEVELOPMENT_VERSION=$(dpkg-query -W -f='${Version}' libnccl-dev)

if [[ "$RUNTIME_VERSION" != "$EXPECTED_VERSION" || "$DEVELOPMENT_VERSION" != "$EXPECTED_VERSION" ]]; then
    echo "NCCL package mismatch: expected $EXPECTED_VERSION, runtime=$RUNTIME_VERSION, development=$DEVELOPMENT_VERSION" >&2
    exit 1
fi

CUDA_INCLUDE_DIR=
for cuda_root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /usr/local/cuda-*; do
    if [[ -n "$cuda_root" && -f "$cuda_root/include/cuda_runtime.h" ]]; then
        CUDA_INCLUDE_DIR="$cuda_root/include"
        break
    fi
done

if [[ -z "$CUDA_INCLUDE_DIR" ]]; then
    echo "CUDA development headers are unavailable: cuda_runtime.h was not found via CUDA_HOME, CUDA_PATH, or /usr/local/cuda*" >&2
    exit 1
fi

if ! g++ -std=c++17 -I"$CUDA_INCLUDE_DIR" -x c++ -fsyntax-only - <<'C'
#include <nccl.h>
#include <nccl_device.h>

static void* const NCCL_EP_APIS[] = {
    (void*)ncclWinGetUserPtr,
    (void*)ncclGetPeerDevicePointer,
    (void*)ncclCommQueryProperties,
};
C
then
    echo "NCCL development headers do not expose the required NCCL-EP APIs" >&2
    exit 1
fi

IFS=. read -r NCCL_MAJOR NCCL_MINOR NCCL_PATCH <<< "${EXPECTED_VERSION%%-*}"
EXPECTED_VERSION_CODE=$((NCCL_MAJOR * 10000 + NCCL_MINOR * 100 + NCCL_PATCH))
LOADED_RUNTIME=$(python3 - <<'PY'
import ctypes
import ctypes.util
import os


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


nccl = ctypes.CDLL("libnccl.so.2")
version = ctypes.c_int()
if nccl.ncclGetVersion(ctypes.byref(version)) != 0:
    raise RuntimeError("ncclGetVersion failed")

libdl = ctypes.CDLL(ctypes.util.find_library("dl"))
info = DlInfo()
if libdl.dladdr(ctypes.cast(nccl.ncclGetVersion, ctypes.c_void_p), ctypes.byref(info)) == 0:
    raise RuntimeError("dladdr failed for ncclGetVersion")

print(f"{os.path.realpath(info.dli_fname.decode())}|{version.value}")
PY
)
LOADED_PATH=${LOADED_RUNTIME%|*}
LOADED_VERSION_CODE=${LOADED_RUNTIME##*|}
PACKAGED_RUNTIME=false

while IFS= read -r packaged_path; do
    if [[ -e "$packaged_path" && "$(readlink -f "$packaged_path")" == "$LOADED_PATH" ]]; then
        PACKAGED_RUNTIME=true
        break
    fi
done < <(dpkg-query -L libnccl2)

if [[ "$PACKAGED_RUNTIME" != true ]]; then
    echo "The dynamic loader selected $LOADED_PATH, which is not owned by libnccl2 $EXPECTED_VERSION" >&2
    exit 1
fi

if [[ "$LOADED_VERSION_CODE" != "$EXPECTED_VERSION_CODE" ]]; then
    echo "Loaded NCCL reports version code $LOADED_VERSION_CODE, expected $EXPECTED_VERSION_CODE" >&2
    exit 1
fi

printf 'Validated NCCL %s from %s\n' "$EXPECTED_VERSION" "$LOADED_PATH"
