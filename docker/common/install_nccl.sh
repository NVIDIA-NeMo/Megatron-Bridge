#!/bin/bash

set -ex

NCCL_VER="2.27.3-1+cuda12.9"

for i in "$@"; do
    case $i in
        --NCCL_VER=?*) NCCL_VER="${i#*=}";;
        *) ;;
    esac
    shift
done

ARCH=$(uname -m)
if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${ARCH}/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

apt-get update

if [[ $(apt list --installed | grep libnccl) ]]; then
  apt-get remove --purge -y --allow-change-held-packages libnccl*
fi

apt-get install -y --no-install-recommends \
    libnccl2=${NCCL_VER} \
    libnccl-dev=${NCCL_VER} \

apt-get clean
rm -rf /var/lib/apt/lists/*
