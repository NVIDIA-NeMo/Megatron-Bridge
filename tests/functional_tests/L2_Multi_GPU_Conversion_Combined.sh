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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Set up 2 GPUs for both TP and PP tests
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_OFFLINE=1

# Create a temporary directory for the test output
TEST_OUTPUT_DIR=$(mktemp -d)
echo "Test output directory: $TEST_OUTPUT_DIR"

# Function to run a conversion test configuration and validate output
run_conversion_test() {
    local tp=$1
    local pp=$2
    local test_name=$3
    local test_output_dir="$TEST_OUTPUT_DIR/${test_name}"
    
    echo "Running $test_name conversion test (tp=$tp, pp=$pp)..."
    
    # Create test-specific output directory
    mkdir -p "$test_output_dir"
    
    # Run multi_gpu_hf.py with specified configuration
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
        -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
        examples/models/multi_gpu_hf.py \
        --hf-model-id "meta-llama/Llama-3.2-1B" \
        --output-dir "$test_output_dir" \
        --tp $tp \
        --pp $pp

    # Verify that the converted model was saved
    if [ ! -d "$test_output_dir/Llama-3.2-1B" ]; then
        echo "ERROR: Converted model directory not found at $test_output_dir/Llama-3.2-1B for $test_name test"
        exit 1
    fi

    # Check that essential model files exist
    if [ ! -f "$test_output_dir/Llama-3.2-1B/config.json" ]; then
        echo "ERROR: config.json not found in converted model for $test_name test"
        exit 1
    fi

    if [ ! -f "$test_output_dir/Llama-3.2-1B/model.safetensors" ] && [ ! -f "$test_output_dir/Llama-3.2-1B/pytorch_model.bin" ]; then
        echo "ERROR: Model weights file not found in converted model for $test_name test"
        exit 1
    fi

    echo "SUCCESS: $test_name conversion test completed successfully"
    echo "Converted model saved at: $test_output_dir/Llama-3.2-1B"
    echo ""
}

# Test 1: Tensor Parallelism (tp=2, pp=1)
run_conversion_test 2 1 "TP"

# Test 2: Pipeline Parallelism (tp=1, pp=2)
run_conversion_test 1 2 "PP"

echo "SUCCESS: All Multi-GPU conversion tests (TP and PP) completed successfully"

# Combine coverage data
coverage combine

# Clean up temporary directory
rm -rf "$TEST_OUTPUT_DIR"
echo "Cleaned up temporary directory" 