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

# Set up 2 GPUs for tp=1 and pp=2
export CUDA_VISIBLE_DEVICES="0,1"
export TRANSFORMERS_OFFLINE=1

# Create a temporary directory for the test output
TEST_OUTPUT_DIR=$(mktemp -d)
echo "Test output directory: $TEST_OUTPUT_DIR"

# Run multi_gpu_hf.py with tp=1 and pp=2
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 \
    -m coverage run --data-file=/workspace/.coverage --source=/workspace/ --parallel-mode \
    examples/models/multi_gpu_hf.py \
    --hf-model-id "meta-llama/Llama-3.2-1B" \
    --output-dir "$TEST_OUTPUT_DIR" \
    --tp 1 \
    --pp 2

# Verify that the converted model was saved
if [ ! -d "$TEST_OUTPUT_DIR/Llama-3.2-1B" ]; then
    echo "ERROR: Converted model directory not found at $TEST_OUTPUT_DIR/Llama-3.2-1B"
    exit 1
fi

# Check that essential model files exist
if [ ! -f "$TEST_OUTPUT_DIR/Llama-3.2-1B/config.json" ]; then
    echo "ERROR: config.json not found in converted model"
    exit 1
fi

if [ ! -f "$TEST_OUTPUT_DIR/Llama-3.2-1B/model.safetensors" ] && [ ! -f "$TEST_OUTPUT_DIR/Llama-3.2-1B/pytorch_model.bin" ]; then
    echo "ERROR: Model weights file not found in converted model"
    exit 1
fi

echo "SUCCESS: Multi-GPU conversion test with Pipeline Parallelism (pp=2) completed successfully"
echo "Converted model saved at: $TEST_OUTPUT_DIR/Llama-3.2-1B"

# Combine coverage data
coverage combine

# Clean up temporary directory
rm -rf "$TEST_OUTPUT_DIR"
echo "Cleaned up temporary directory" 