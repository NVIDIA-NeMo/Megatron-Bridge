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

#!/bin/sh

# GCP detection: try multiple independent signals because newer machine types
# do not expose SMBIOS to the guest, so the DMI path alone misses them.
_is_gcp_host() {
    # gVNIC driver bound to at least one PCI device.
    for _d in /sys/bus/pci/drivers/gve/*:*; do
        [ -e "$_d" ] && return 0
    done
    # Any PCI device with Google's vendor ID.
    for _v in /sys/bus/pci/devices/*/vendor; do
        [ -r "$_v" ] && [ "$(cat "$_v")" = "0x1ae0" ] && return 0
    done
    # SMBIOS fallback for machine families that expose it.
    [ "$(cat /sys/class/dmi/id/product_name 2>/dev/null)" = "Google Compute Engine" ] && return 0
    [ "$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null)" = "Google" ] && return 0
    return 1
}

# Only gIB-capable GCP instance families expose mlx5 RDMA NICs.
_has_nvidia_rdma_nic() {
    for _d in /sys/bus/pci/drivers/mlx5_core/*:*; do
        [ -e "$_d" ] && return 0
    done
    return 1
}

if _is_gcp_host && _has_nvidia_rdma_nic && \
    [ -f /opt/gcp/nccl-plugins/lib64/libnccl-net-gcp.so ]; then
    export NCCL_NET_PLUGIN=gcp
    if [ -f /opt/gcp/nccl-plugins/lib64/libnccl-tuner-gcp.so ]; then
        export NCCL_TUNER_PLUGIN=gcp
    fi
    if [ -f /opt/gcp/nccl-plugins/lib64/libnccl-profiler-gcp.so ]; then
        export NCCL_PROFILER_PLUGIN=gcp
    fi
    if [ -f /opt/gcp/nccl-plugins/lib64/libnccl-env-gcp.so ]; then
        export NCCL_ENV_PLUGIN=gcp
    fi
    if [ -d /opt/gcp/nccl-plugins/configs ]; then
        export NCCL_TUNER_CONFIG_PATH=/opt/gcp/nccl-plugins/configs
    fi
fi

if [ "${NCCL_PROFILER_PLUGIN:-}" = gcp ]; then
    echo
    echo "NOTE: GCP NCCL telemetry enabled."
    echo "      See https://docs.cloud.google.com/ai-hypercomputer/docs/nccl/comma"
    echo "      for further information."
fi

unset -f _is_gcp_host
unset -f _has_nvidia_rdma_nic
unset _d
unset _v
