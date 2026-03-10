#!/bin/bash

set -ex

CUDNN_VERSION=""
DEVEL="0"

for i in "$@"; do
    case $i in
        --CUDNN_VERSION=?*) CUDNN_VERSION="${i#*=}";;
        --DEVEL) DEVEL="1";;
        *) ;;
    esac
    shift
done

if [ -z "$CUDNN_VERSION" ]; then
    echo "Error: CUDNN_VERSION is required (via --CUDNN_VERSION=)"
    exit 1
fi

ARCH=$(uname -m)
if [ "$ARCH" = "amd64" ]; then ARCH="x86_64"; fi
if [ "$ARCH" = "aarch64" ]; then ARCH="sbsa"; fi

# Extract major version for package name suffix (e.g., "9.18.1.3-1" -> "9")
CUDNN_MAJOR=$(echo "$CUDNN_VERSION" | cut -d. -f1)

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update

apt-get remove --purge -y --allow-change-held-packages "libcudnn${CUDNN_MAJOR}*" || true

apt-get install -y --no-install-recommends \
    "libcudnn${CUDNN_MAJOR}-cuda-13=${CUDNN_VERSION}"

if [ "$DEVEL" = "1" ]; then
    apt-get install -y --no-install-recommends \
        "libcudnn${CUDNN_MAJOR}-dev-cuda-13=${CUDNN_VERSION}"
fi

apt-get clean
rm -rf /var/lib/apt/lists/*
