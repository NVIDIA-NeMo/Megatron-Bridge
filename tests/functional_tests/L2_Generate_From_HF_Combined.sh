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

# Function to run a test configuration and validate output
run_test_config() {
    local tp=$1
    local pp=$2
    local test_name=$3
    local output_file="$TEST_OUTPUT_DIR/generation_output_${test_name}.txt"
    
    echo "Running $test_name test (tp=$tp, pp=$pp)..."
    
    # Run generate_from_hf.py with specified configuration
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
        -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
        examples/models/generate_from_hf.py \
        --hf_model_path "meta-llama/Llama-3.2-1B" \
        --prompt "Hello, how are you?" \
        --max_new_tokens 10 \
        --tp $tp \
        --pp $pp > "$output_file" 2>&1

    # Check that the generation completed successfully
    if ! grep -q "GENERATED TEXT OUTPUT" "$output_file"; then
        echo "ERROR: Generation output not found in $test_name test log"
        cat "$output_file"
        exit 1
    fi

    # Check that the prompt appears in the output
    if ! grep -q "Hello, how are you?" "$output_file"; then
        echo "ERROR: Original prompt not found in $test_name test generation output"
        cat "$output_file"
        exit 1
    fi

    # Check that generated text is present (should contain more than just the prompt)
    if ! grep -q "Generated:" "$output_file"; then
        echo "ERROR: Generated text section not found in $test_name test output"
        cat "$output_file"
        exit 1
    fi

    echo "SUCCESS: $test_name test completed successfully"
    echo "$test_name generation output:"
    cat "$output_file"
    echo ""
}

# Test 1: Tensor Parallelism (tp=2, pp=1)
run_test_config 2 1 "TP"

# Test 2: Pipeline Parallelism (tp=1, pp=2)
run_test_config 1 2 "PP"

echo "SUCCESS: All text generation tests (TP and PP) completed successfully"

# Combine coverage data
coverage combine

# Clean up temporary directory
rm -rf "$TEST_OUTPUT_DIR"
echo "Cleaned up temporary directory" 