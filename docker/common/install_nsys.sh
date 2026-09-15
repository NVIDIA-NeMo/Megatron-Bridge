#!/bin/bash
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
case "$ARCH" in
    amd64) NSYS_TARGET=target-linux-x64 ;;
    arm64) NSYS_TARGET=target-linux-sbsa-armv8 ;;
    *) echo "Error: unsupported Nsight Systems architecture: $ARCH" >&2; exit 1 ;;
esac

# Devtools CLI packages include the patch version in their name. Their Debian
# revision is publisher-specific, rather than always being "-1".
NSYS_RELEASE=$(cut -d. -f1-3 <<< "$NSYS_VERSION")
NSYS_PKG="nsight-systems-cli-${NSYS_RELEASE}"
NSYS_BIN="/opt/nvidia/nsight-systems-cli/${NSYS_RELEASE}/${NSYS_TARGET}/nsys"
NSYS_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu2404/${ARCH}"
curl -fsSL "${NSYS_REPO}/nvidia.pub" | gpg --dearmor --yes --output /usr/share/keyrings/nvidia-devtools.gpg
echo "deb [arch=${ARCH} signed-by=/usr/share/keyrings/nvidia-devtools.gpg] ${NSYS_REPO}/ /" \
    > /etc/apt/sources.list.d/nvidia-devtools.list

apt-get update

NSYS_APT_VERSION=$(apt-cache madison "$NSYS_PKG" | awk -v version="$NSYS_VERSION" \
    '$3 == version || index($3, version "-") == 1 {print $3; exit}')
if [ -z "$NSYS_APT_VERSION" ]; then
    echo "Error: Nsight Systems $NSYS_VERSION is unavailable for $ARCH" >&2
    exit 1
fi

# Keep an existing installation of the requested package when MB is built on
# fw-base, while purging older CLI and GUI packages inherited from the base.
mapfile -t OLD_NSYS_PACKAGES < <(
    dpkg-query -W -f='${Package} ${db:Status-Status}\n' 'nsight-systems*' 2>/dev/null \
        | awk -v keep="$NSYS_PKG" '$2 == "installed" && $1 != keep {print $1}'
)
if [ "${#OLD_NSYS_PACKAGES[@]}" -gt 0 ]; then
    apt-get remove --purge -y --allow-change-held-packages "${OLD_NSYS_PACKAGES[@]}"
fi

# A base image may retain the dpkg record while moving the payload into CUDA.
INSTALL_ARGS=()
if [ ! -x "$NSYS_BIN" ]; then
    INSTALL_ARGS+=(--reinstall)
fi
apt-get install -y --no-install-recommends --allow-change-held-packages --allow-downgrades "${INSTALL_ARGS[@]}" \
    "${NSYS_PKG}=${NSYS_APT_VERSION}"
test -x "$NSYS_BIN"
update-alternatives --set nsys "$NSYS_BIN"

# NGC images also bundle copies outside dpkg's /opt/nvidia installation. Remove
# those copies and point CUDA's PATH entry at the newly registered alternative.
for CUDA_DIR in /usr/local/cuda /usr/local/cuda-*; do
    [ -d "$CUDA_DIR" ] || continue
    rm -rf "${CUDA_DIR}"/NsightSystems-cli-*
    if [ -d "${CUDA_DIR}/bin" ]; then
        ln -sfn /usr/local/bin/nsys "${CUDA_DIR}/bin/nsys"
    fi
done

nsys --version

apt-get clean
rm -rf /var/lib/apt/lists/*
