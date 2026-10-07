#!/bin/bash
# CI_TIMEOUT=30
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1
uv run python -m pytest -v -x tests/functional_tests/test_groups/diffusion/diffusion_gemma
