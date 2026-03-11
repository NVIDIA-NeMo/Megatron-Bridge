#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Fault Tolerance Example Launch Script
#
# This script uses ft_launcher from nvidia-resiliency-ext to run training
# with fault tolerance enabled.
#
# Usage:
#   ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh
#   ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh --simulate-fault
#   ./examples/resiliency/fault_tolerance/run_fault_tolerance.sh --nproc 4
#
# Fault Simulation Mode (--simulate-fault):
#   Demonstrates fault recovery by killing a rank after a delay.
#   
#   Timing requirements for successful recovery:
#     checkpoint_time < fault_delay < total_training_time
#   
#   Defaults are tuned for tiny model (~145M params):
#     - train_iters=2000 (~90s total training)
#     - save_interval=200 (first checkpoint at ~9s)
#     - fault_delay=60s (fault triggers after checkpoint, before completion)

set -euo pipefail

# Default values
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_ITERS=""  # Will be set based on mode
MASTER_PORT="${MASTER_PORT:-29500}"
SIMULATE_FAULT=false
MAX_RESTARTS=0
FAULT_DELAY="${FAULT_DELAY:-60}"  # Seconds before fault injection (must be after first checkpoint)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --simulate-fault)
            SIMULATE_FAULT=true
            MAX_RESTARTS=3
            shift
            ;;
        --fault-delay)
            FAULT_DELAY="$2"
            shift 2
            ;;
        --max-restarts)
            MAX_RESTARTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set default TRAIN_ITERS based on mode (if not explicitly provided)
# For --simulate-fault: Use 2000 iterations (~90s) so training outlasts the fault delay (60s)
# For basic mode: Use 50 iterations for quick demonstration
if [ -z "$TRAIN_ITERS" ]; then
    if [ "$SIMULATE_FAULT" = true ]; then
        TRAIN_ITERS=2000
    else
        TRAIN_ITERS=50
    fi
fi

# Check if ft_launcher is available via uv
if ! uv run ft_launcher --help &> /dev/null; then
    echo "Error: ft_launcher not found."
    echo "Please use the NeMo Framework Docker container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo"
    exit 1
fi

# Reduce torch distributed noise
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-error}"

# Set GROUP_RANK for single-node runs (required by use_infra_group_rank)
export GROUP_RANK=0

echo "Running Fault Tolerance Example"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "  Iterations: ${TRAIN_ITERS}"
echo "  Simulate Fault: ${SIMULATE_FAULT}"
echo "  Max Restarts: ${MAX_RESTARTS}"
if [ "$SIMULATE_FAULT" = true ]; then
    echo "  Fault Delay: ${FAULT_DELAY}s"
fi
echo ""

SCRIPT="${SCRIPT_DIR}/fault_tolerance_example.py"
if [ "$SIMULATE_FAULT" = true ]; then
    SCRIPT_ARGS="--train-iters ${TRAIN_ITERS} --simulate-fault --fault-delay ${FAULT_DELAY}"
else
    SCRIPT_ARGS="--train-iters ${TRAIN_ITERS}"
fi

uv run ft_launcher \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
    --nnodes=1 \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --ft-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
    --ft-rank_out_of_section_timeout=300 \
    --monitor-interval=5 \
    --max-restarts="${MAX_RESTARTS}" \
    "${SCRIPT}" ${SCRIPT_ARGS}
