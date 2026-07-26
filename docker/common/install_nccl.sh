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

NCCL_VER="2.30.7-1+cuda13.3"

for arg in "$@"; do
    case "$arg" in
        --NCCL_VER=?*) NCCL_VER="${arg#*=}";;
        *) ;;
    esac
done

ARCH=$(uname -m)
if [[ "$ARCH" == "amd64" ]]; then ARCH="x86_64"; fi
if [[ "$ARCH" == "aarch64" ]]; then ARCH="sbsa"; fi

curl -fsSLO "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb"
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update

if dpkg-query -W 'libnccl*' >/dev/null 2>&1; then
    apt-get remove --purge -y --allow-change-held-packages 'libnccl*'
fi

apt-get install -y --no-install-recommends \
    "libnccl2=${NCCL_VER}" \
    "libnccl-dev=${NCCL_VER}"
ldconfig

runtime_version=$(dpkg-query -W -f='${Version}' libnccl2)
devel_version=$(dpkg-query -W -f='${Version}' libnccl-dev)
if [[ "$runtime_version" != "$NCCL_VER" || "$devel_version" != "$NCCL_VER" ]]; then
    echo "NCCL package mismatch: requested=${NCCL_VER} runtime=${runtime_version} devel=${devel_version}" >&2
    exit 1
fi

NCCL_HEADER=/usr/include/nccl.h
NCCL_DEVICE_HEADER=/usr/include/nccl_device.h
[[ -f "$NCCL_HEADER" ]] || { echo "Missing $NCCL_HEADER" >&2; exit 1; }
[[ -f "$NCCL_DEVICE_HEADER" ]] || { echo "Missing $NCCL_DEVICE_HEADER" >&2; exit 1; }
grep -Fq 'ncclWinGetUserPtr' "$NCCL_HEADER" || { echo "Missing ncclWinGetUserPtr declaration" >&2; exit 1; }
device_header_root=/usr/include/nccl_device
[[ -d "$device_header_root" ]] || device_header_root=/usr/include
for declaration in ncclGetPeerDevicePointer ncclCommQueryProperties; do
    grep -R -Fq "$declaration" "$device_header_root" || {
        echo "Missing $declaration declaration" >&2
        exit 1
    }
done

IFS='|' read -r runtime_path runtime_api_version < <(python - <<'PY'
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


libnccl = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_LOCAL)
version = ctypes.c_int()
if libnccl.ncclGetVersion(ctypes.byref(version)) != 0:
    raise RuntimeError("ncclGetVersion failed")
libdl = ctypes.CDLL(ctypes.util.find_library("dl") or "libdl.so.2")
libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
info = DlInfo()
if libdl.dladdr(ctypes.cast(libnccl.ncclGetVersion, ctypes.c_void_p), ctypes.byref(info)) == 0:
    raise RuntimeError("dladdr could not resolve libnccl.so.2")
print(f"{os.path.realpath(info.dli_fname.decode())}|{version.value}")
PY
)
runtime_owner=$(dpkg-query -S "$runtime_path" | cut -d: -f1)
[[ "$runtime_owner" == "libnccl2" ]] || {
    echo "Dynamic loader resolves $runtime_path from $runtime_owner instead of libnccl2" >&2
    exit 1
}

version_without_release=${NCCL_VER%%-*}
IFS=. read -r major minor patch <<< "$version_without_release"
expected_api_version=$((major * 10000 + minor * 100 + patch))
if [[ "$runtime_api_version" != "$expected_api_version" ]]; then
    echo "NCCL runtime API mismatch: requested=${expected_api_version} loaded=${runtime_api_version}" >&2
    exit 1
fi

printf 'Verified NCCL runtime=%s api=%s devel=%s prefix=/usr library=%s\n' \
    "$runtime_version" "$runtime_api_version" "$devel_version" "$runtime_path"

apt-get clean
rm -rf /var/lib/apt/lists/*
