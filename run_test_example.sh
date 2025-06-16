#!/bin/bash

# Training Initialization Sanity Test Runner
# Simple script to run the test using torchrun on a single node

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_training_initialization.py"

# Default to single GPU
NPROC_PER_NODE=1

# Parse simple arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus N    Number of GPUs to use (default: 1)"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0           # Single GPU test"
            echo "  $0 --gpus 4  # Multi-GPU test"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "Error: Test script not found at $TEST_SCRIPT"
    exit 1
fi

echo "================================================================================"
echo "Training Initialization Sanity Test"
echo "================================================================================"
echo "Using $NPROC_PER_NODE GPU(s)"

# Check CUDA availability
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Available GPUs: $AVAILABLE_GPUS"

if [[ $NPROC_PER_NODE -gt $AVAILABLE_GPUS ]]; then
    echo "Error: Requested $NPROC_PER_NODE GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

echo "================================================================================"

# Run the test using torchrun
torchrun --nproc-per-node="$NPROC_PER_NODE" "$TEST_SCRIPT"

# Check exit status
EXIT_CODE=$?

echo ""
echo "================================================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Test completed successfully!"
    echo "Training initialization pipeline is ready for PEFT training."
else
    echo "❌ Test failed with exit code $EXIT_CODE"
fi
echo "================================================================================"

exit $EXIT_CODE 