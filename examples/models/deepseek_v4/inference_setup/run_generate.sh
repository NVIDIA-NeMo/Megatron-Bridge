#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Run DeepSeek-V4-Flash inference end-to-end on cw inside mbridge.
# Assumes you've already run convert.py to produce model{0..7}-mp8.safetensors.
#
# Invocation from a node with salloc already granted and tmux aa attached:
#
#   srun --container-image=.../mbridge-260321.sqsh \
#        --container-mounts=/lustre:/lustre \
#        --no-container-mount-home \
#        --gpus-per-task=8 --pty \
#        bash ./run_generate.sh
#
# Note: each srun starts a fresh container from the squashfs, so pip installs
# don't persist. This script reinstalls needed deps every invocation.

set -eo pipefail
set +u

DSV4_INFER=/lustre/fsw/portfolios/coreai/users/yuya/dsv4_infer
DSV4_MP8=/lustre/fsw/portfolios/coreai/users/yuya/dsv4_mp8

echo "=== installing inference deps into container ==="
pip install --no-cache-dir -q transformers safetensors 2>&1 | tail -2
# tilelang 0.1.9 works under Python 3.12 in mbridge-260321. 0.1.8 does NOT
# (hits a tvm-ffi __init_subclass__ bug). Pin 0.1.9.
pip install --no-cache-dir -q tilelang==0.1.9 2>&1 | tail -2

# fast_hadamard_transform's wheel won't build against the container's
# torch/cuda toolchain — use the pure-torch shim under PYTHONPATH.

cd "${DSV4_INFER}/inference"

export PYTHONPATH="${DSV4_INFER}:${PYTHONPATH}"
echo "=== starting torchrun ==="
torchrun --nproc-per-node 8 generate.py \
    --ckpt-path "${DSV4_MP8}" \
    --config "${DSV4_INFER}/inference/config.json" \
    --input-file "${DSV4_INFER}/prompts.txt" \
    --max-new-tokens 100 \
    2>&1 | tee "${DSV4_INFER}/generate.log"
