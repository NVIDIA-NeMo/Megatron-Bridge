#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CI_TIMEOUT=5
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 uv run python -m pytest -v \
  tests/unit_tests/models/test_fp8_param_export.py::TestFp8ParamExport::test_real_te_mxfp8_export_dequantization_parity
