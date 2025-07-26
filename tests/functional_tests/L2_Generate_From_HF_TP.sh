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

# Set up 2 GPUs for tp=2 and pp=1
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_OFFLINE=1

# Create a temporary directory for the test output
TEST_OUTPUT_DIR=$(mktemp -d)
echo "Test output directory: $TEST_OUTPUT_DIR"

# Run generate_from_hf.py with tp=2 and pp=1
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
    -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
    examples/models/generate_from_hf.py \
    --hf_model_path "meta-llama/Llama-3.2-1B" \
    --prompt "Hello, how are you?" \
    --max_new_tokens 10 \
    --tp 2 \
    --pp 1 > "$TEST_OUTPUT_DIR/generation_output.txt" 2>&1

# Check that the generation completed successfully
if ! grep -q "GENERATED TEXT OUTPUT" "$TEST_OUTPUT_DIR/generation_output.txt"; then
    echo "ERROR: Generation output not found in log"
    cat "$TEST_OUTPUT_DIR/generation_output.txt"
    exit 1
fi

# Check that the prompt appears in the output
if ! grep -q "Hello, how are you?" "$TEST_OUTPUT_DIR/generation_output.txt"; then
    echo "ERROR: Original prompt not found in generation output"
    cat "$TEST_OUTPUT_DIR/generation_output.txt"
    exit 1
fi

# Check that generated text is present (should contain more than just the prompt)
if ! grep -q "Generated:" "$TEST_OUTPUT_DIR/generation_output.txt"; then
    echo "ERROR: Generated text section not found in output"
    cat "$TEST_OUTPUT_DIR/generation_output.txt"
    exit 1
fi

echo "SUCCESS: Text generation test with Tensor Parallelism (tp=2) completed successfully"
echo "Generation output:"
cat "$TEST_OUTPUT_DIR/generation_output.txt"

# Combine coverage data
coverage combine

# Clean up temporary directory
rm -rf "$TEST_OUTPUT_DIR"
echo "Cleaned up temporary directory" 