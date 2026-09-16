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

set -ex

NSYS_VERSION="${NSIGHT_SYSTEMS_VERSION:-}"

for i in "$@"; do
    case $i in
        --NSYS_VERSION=?*) NSYS_VERSION="${i#*=}";;
        *) ;;
    esac
    shift
done

if [ -z "$NSYS_VERSION" ]; then
    echo "Error: NSYS_VERSION is required (via --NSYS_VERSION= or NSIGHT_SYSTEMS_VERSION env var)"
    exit 1
fi

ARCH=$(dpkg --print-architecture)

# Extract year.major.patch for the devtools package name.
NSYS_RELEASE=$(echo "$NSYS_VERSION" | cut -d. -f1-3)
NSYS_PKG="nsight-systems-${NSYS_RELEASE}"

NSYS_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu2404/${ARCH}"
curl -fsSL "${NSYS_REPO}/nvidia.pub" -o /usr/share/keyrings/nvidia-devtools.asc
echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.asc] ${NSYS_REPO}/ /" \
    > /etc/apt/sources.list.d/nvidia-devtools.list

apt-get update

apt-get remove --purge -y --allow-change-held-packages 'nsight-systems*' || true

apt-get install -y --no-install-recommends "${NSYS_PKG}=${NSYS_VERSION}"

# CUDA can precede /usr/local/bin on PATH in the base image.
if [ -L /usr/local/cuda/bin/nsys ]; then
    ln -sf /usr/local/bin/nsys /usr/local/cuda/bin/nsys
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
