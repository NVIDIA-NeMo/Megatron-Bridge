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

ARCH=$(uname -m)
if [ "$ARCH" = "amd64" ]; then ARCH="x86_64"; fi
if [ "$ARCH" = "aarch64" ]; then ARCH="sbsa"; fi

# Extract year.major for package name (e.g., "2026.1.0.1085" -> "nsight-systems-2026.1")
NSYS_YEAR_MAJOR=$(echo "$NSYS_VERSION" | cut -d. -f1,2)
NSYS_PKG="nsight-systems-${NSYS_YEAR_MAJOR}"

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update

apt-get remove --purge -y --allow-change-held-packages 'nsight-systems*' || true

apt-get install -y --no-install-recommends "${NSYS_PKG}=${NSYS_VERSION}-1"

apt-get clean
rm -rf /var/lib/apt/lists/*
