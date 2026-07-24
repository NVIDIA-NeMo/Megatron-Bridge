#!/usr/bin/env python3
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
"""Dump perf recipe configs for comparison between branches.

Usage (run from project root, with uv run):

  # On main branch — dump old-path configs:
  uv run python -m scripts.performance.dump_perf_configs --mode old --out /tmp/configs_main

  # On PR branch — dump new-path configs:
  uv run python -m scripts.performance.dump_perf_configs --mode new --out /tmp/configs_pr

  # Diff:
  diff -rq /tmp/configs_main /tmp/configs_pr
  diff -ru /tmp/configs_main/kimi_k2_pretrain_256gpu_gb300_fp8cs.yaml \
            /tmp/configs_pr/kimi_k2_pretrain_256gpu_gb300_fp8cs.yaml

Both modes use the suffix-less flat recipe by default. They serialize the ConfigContainer via its to_dict() /
dataclasses.asdict path (same as save_config_filepath in production), so the
YAML is directly comparable.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


_DEFAULT_CHECKPOINT_LOAD_SUFFIX = Path("nemo_experiments") / "default" / "checkpoints"
_CANONICAL_DUMP_CHECKPOINT_LOAD = "/nemo_run/nemo_experiments/default/checkpoints"


# ---------------------------------------------------------------------------
# All (family, recipe, task, num_gpus, gpu, precision[, config_variant]) combos
# that exist in both the old scripts/performance/configs/ and the new flat perf
# recipes. Add entries here whenever a new recipe is added to either path.
# ---------------------------------------------------------------------------
COMBOS = [
    # Llama 3 8B
    ("llama", "llama3_8b", "pretrain", 8, "gb300", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "gb300", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "gb300", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "gb300", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "bf16"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 8, "gb200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "gb200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "gb200", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "gb200", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 32, "gb200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 32, "gb200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "vr200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "vr200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "vr200", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "vr200", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 8, "b300", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "b300", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "b300", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "b300", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 8, "b200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "b200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "b200", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "b200", "nvfp4"),
    ("llama", "llama3_8b", "pretrain", 64, "b200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 64, "b200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "h100", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "h100", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 64, "h100", "bf16"),
    ("llama", "llama3_8b", "pretrain", 64, "h100", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "r100", "bf16"),
    ("llama", "llama3_8b", "pretrain", 8, "r100", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 8, "r100", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 8, "r100", "nvfp4"),
    ("llama", "llama3_8b", "sft", 8, "gb200", "bf16"),
    ("llama", "llama3_8b", "sft", 8, "gb200", "fp8_cs"),
    ("llama", "llama3_8b", "sft", 8, "gb200", "fp8_mx"),
    ("llama", "llama3_8b", "sft", 8, "h100", "bf16"),
    ("llama", "llama3_8b", "sft", 8, "h100", "fp8_cs"),
    # Llama 3 70B
    ("llama", "llama3_70b", "pretrain", 32, "gb300", "bf16"),
    ("llama", "llama3_70b", "pretrain", 32, "gb300", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "gb300", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "gb300", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "gb300", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "gb300", "nvfp4"),
    ("llama", "llama3_70b", "pretrain", 32, "gb200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 32, "gb200", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "gb200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "gb200", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "gb200", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "gb200", "nvfp4"),
    ("llama", "llama3_70b", "pretrain", 64, "b300", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "b300", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "b300", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "b300", "nvfp4"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "nvfp4"),
    ("llama", "llama3_70b", "pretrain", 64, "h100", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "h100", "fp8_cs"),
    ("llama", "llama3_70b", "sft", 32, "gb300", "bf16"),
    ("llama", "llama3_70b", "sft", 32, "gb300", "fp8_cs"),
    ("llama", "llama3_70b", "sft", 32, "gb300", "fp8_mx"),
    ("llama", "llama3_70b", "sft", 32, "gb200", "bf16"),
    ("llama", "llama3_70b", "sft", 32, "gb200", "fp8_cs"),
    ("llama", "llama3_70b", "sft", 32, "gb200", "fp8_mx"),
    ("llama", "llama3_70b", "sft", 32, "h100", "bf16"),
    ("llama", "llama3_70b", "sft", 32, "h100", "fp8_cs"),
    ("llama", "llama3_70b", "peft", 8, "gb300", "bf16"),
    ("llama", "llama3_70b", "peft", 8, "gb300", "fp8_cs"),
    ("llama", "llama3_70b", "peft", 8, "gb300", "fp8_mx"),
    ("llama", "llama3_70b", "peft", 8, "gb200", "bf16"),
    ("llama", "llama3_70b", "peft", 8, "gb200", "fp8_cs"),
    ("llama", "llama3_70b", "peft", 8, "gb200", "fp8_mx"),
    ("llama", "llama3_70b", "peft", 8, "b300", "bf16"),
    ("llama", "llama3_70b", "peft", 8, "b300", "fp8_cs"),
    ("llama", "llama3_70b", "peft", 8, "b300", "fp8_mx"),
    ("llama", "llama3_70b", "peft", 8, "b200", "bf16"),
    ("llama", "llama3_70b", "peft", 8, "b200", "fp8_cs"),
    ("llama", "llama3_70b", "peft", 8, "b200", "fp8_mx"),
    ("llama", "llama3_70b", "peft", 8, "h100", "bf16"),
    ("llama", "llama3_70b", "peft", 8, "h100", "fp8_cs"),
    # Llama 3.1 405B
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 128, "b300", "bf16"),
    ("llama", "llama31_405b", "pretrain", 128, "b300", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 128, "b300", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 128, "b300", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 128, "b200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 128, "b200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 128, "b200", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 128, "b200", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 512, "h100", "bf16"),
    ("llama", "llama31_405b", "pretrain", 512, "h100", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 1024, "h100", "bf16"),
    ("llama", "llama31_405b", "pretrain", 1024, "h100", "fp8_cs"),
    # DeepSeek V3
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 64, "gb300", "bf16", "fsdp"),
    ("deepseek", "deepseek_v3", "pretrain", 64, "gb300", "fp8_mx", "fsdp"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "vr200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "vr200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "vr200", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "vr200", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b300", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b300", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b300", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b300", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 64, "h100", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 64, "h100", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 1024, "h100", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 1024, "h100", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 1024, "h100", "fp8_sc"),
    ("deepseek", "deepseek_v4_pro", "pretrain", 256, "gb300", "fp8_mx"),
    # Qwen3 MoE 30B-A3B
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb300", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb300", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb300", "fp8_mx"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb300", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb300", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "gb200", "fp8_mx"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b300", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b300", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b300", "fp8_mx"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "b200", "fp8_mx"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "b200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 16, "h100", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 16, "h100", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "h100", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "h100", "fp8_cs"),
    # Qwen3 MoE 235B-A22B
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b300", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b300", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b300", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b300", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "h100", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "h100", "fp8_cs"),
    # Qwen3-Next 80B-A3B
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "gb300", "bf16"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "gb300", "fp8_mx"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "gb200", "bf16"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "gb200", "fp8_mx"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "b300", "bf16"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "b300", "fp8_mx"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "b200", "bf16"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 64, "b200", "fp8_mx"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 128, "h100", "bf16"),
    ("qwen", "qwen3_next_80b_a3b", "pretrain", 128, "h100", "fp8_cs"),
    # Kimi K2
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "nvfp4"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "vr200", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "vr200", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "b300", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "b300", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "b300", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 1024, "h100", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 1024, "h100", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 1024, "h100", "fp8_sc"),
    # Nemotron 3 Nano
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb300", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb300", "fp8_mx"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb300", "nvfp4"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb200", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb200", "fp8_mx"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "gb200", "nvfp4"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b300", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b300", "fp8_mx"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b300", "nvfp4"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b200", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b200", "fp8_mx"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "b200", "nvfp4"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 16, "h100", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 16, "h100", "fp8_cs"),
    # Nemotron 3 Super
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb300", "bf16"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb300", "fp8_mx"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb300", "nvfp4"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb200", "bf16"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb200", "fp8_mx"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "gb200", "nvfp4"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b300", "bf16"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b300", "fp8_mx"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b300", "nvfp4"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b200", "bf16"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b200", "fp8_mx"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "b200", "nvfp4"),
    # NemotronH 56B
    ("nemotronh", "nemotronh_56b", "pretrain", 64, "gb300", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "gb300", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 64, "gb200", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 64, "b300", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 64, "b200", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "b200", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 64, "h100", "fp8_cs"),
    # GPT-OSS 20B
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "b300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "b300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "gb200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 72, "gb200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 512, "gb200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "gb300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 72, "gb300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 512, "gb300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "vr200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "vr200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "vr200", "nvfp4"),
    # GPT-OSS 120B
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb300", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb200", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b300", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b200", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "h100", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "h100", "fp8_cs"),
    # WAN 14B (diffusion)
    ("wan", "wan_14b", "pretrain", 16, "gb200", "bf16"),
    ("wan", "wan_14b", "pretrain", 32, "h100", "bf16"),
    # Qwen3-VL 30B-A3B
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb300", "bf16"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb200", "bf16"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "b200", "bf16"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "b200", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 8, "b200", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 16, "h100", "bf16"),
    ("qwen_vl", "qwen3_vl_30b_a3b", "pretrain", 16, "h100", "fp8_cs"),
    # Qwen3-VL 235B-A22B
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb300", "bf16"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb200", "bf16"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "b200", "bf16"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 64, "b200", "fp8_mx"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 256, "h100", "bf16"),
    ("qwen_vl", "qwen3_vl_235b_a22b", "pretrain", 256, "h100", "fp8_cs"),
    # Qwen3.5-VL 35B-A3B
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 16, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 16, "h100", "fp8_cs"),
    # Qwen3.5-VL 122B-A10B
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 128, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 128, "h100", "fp8_cs"),
    # Qwen3.5-VL 397B-A17B
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 256, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 256, "h100", "fp8_cs"),
    # Additional flattened coverage
    ("deepseek", "deepseek_v3", "pretrain", 128, "vr200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 128, "vr200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 128, "vr200", "fp8_mx"),
    ("deepseek", "deepseek_v3", "pretrain", 128, "vr200", "nvfp4"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "fp8_mx", "large_scale"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "fp8_mx", "large_scale"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b300", "fp8_mx", "large_scale"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "fp8_mx", "large_scale"),
    ("deepseek", "deepseek_v3", "pretrain", 1024, "h100", "fp8_sc", "large_scale"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "gb300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "gb200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "b300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "vr200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 8, "vr200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "b300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 64, "vr200", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 72, "gb300", "nvfp4"),
    ("gpt_oss", "gpt_oss_20b", "pretrain", 72, "gb200", "nvfp4"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "vr200", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "vr200", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "vr200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "vr200", "fp8_mx"),
    ("llama", "llama3_70b", "pretrain", 64, "vr200", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 128, "gb300", "bf16"),
    ("llama", "llama31_405b", "pretrain", 128, "gb300", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 128, "gb300", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 128, "gb300", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 128, "gb200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 128, "gb200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 128, "gb200", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 128, "gb200", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 256, "b300", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "b300", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "b300", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 256, "b300", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "nvfp4"),
    ("llama", "llama31_405b", "pretrain", 256, "vr200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "vr200", "fp8_mx"),
    ("llama", "llama31_405b", "pretrain", 256, "vr200", "nvfp4"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "vr200", "bf16"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "vr200", "fp8_mx"),
    ("nemotronh", "nemotron_3_nano", "pretrain", 8, "vr200", "nvfp4"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "vr200", "bf16"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "vr200", "fp8_mx"),
    ("nemotronh", "nemotron_3_super", "pretrain", 64, "vr200", "nvfp4"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "vr200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 8, "vr200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb300", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb300", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb300", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb300", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "gb200", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b300", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b300", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b300", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b300", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 64, "b200", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "vr200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "vr200", "fp8_mx"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "vr200", "nvfp4"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "fp8_mx", "large_scale"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "fp8_mx", "large_scale"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b300", "fp8_mx", "large_scale"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "fp8_mx", "large_scale"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "h100", "fp8_cs", "large_scale"),
]


def _dump_config_to_yaml(cfg, yaml_path: Path) -> None:
    """Dump a ConfigContainer to YAML using the production to_yaml() path."""
    _canonicalize_dump_only_paths(cfg)
    cfg.to_yaml(str(yaml_path))


def _canonicalize_dump_only_paths(cfg) -> None:
    """Canonicalize dump-only cwd-derived paths for branch-to-branch comparisons."""
    checkpoint_load = getattr(cfg.checkpoint, "load", None)
    if not isinstance(checkpoint_load, str):
        return

    try:
        checkpoint_load_suffix = Path(checkpoint_load).relative_to(Path.cwd())
    except ValueError:
        return

    if checkpoint_load_suffix == _DEFAULT_CHECKPOINT_LOAD_SUFFIX:
        cfg.checkpoint.load = _CANONICAL_DUMP_CHECKPOINT_LOAD


def _apply_flat_recipe_runtime_overrides(cfg, precision: str):
    """Apply runtime overrides that still live outside flat perf recipe functions."""
    if precision == "bf16" and cfg.optimizer.optimizer == "adam":
        cfg.optimizer.use_precision_aware_optimizer = True
    return cfg


def load_old_recipe(
    family: str,
    recipe: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None,
):
    """Load recipe using the OLD scripts/performance/configs/ path (main branch)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.overrides import set_post_overrides
    from utils.utils import get_perf_optimized_recipe

    variant_kwargs = {} if config_variant is None else {"config_variant": config_variant}
    cfg = get_perf_optimized_recipe(
        model_family_name=family,
        model_recipe_name=recipe,
        train_task=task,
        gpu=gpu,
        compute_dtype=precision,
        **variant_kwargs,
    )
    return set_post_overrides(
        cfg,
        family,
        recipe,
        gpu,
        num_gpus,
        precision,
        task,
        **variant_kwargs,
    )


def _flat_recipe_variant_suffix(config_variant: str | None) -> str:
    """Return the suffix used in flat perf recipe function names."""
    return "" if config_variant is None else f"_{config_variant}"


def load_new_recipe(
    family: str,
    recipe: str,
    task: str,
    num_gpus: int,
    gpu: str,
    precision: str,
    config_variant: str | None,
):
    """Load recipe using the NEW flat perf recipe path (PR branch)."""
    precision_map = {
        "bf16": "bf16",
        "fp8_cs": "fp8cs",
        "fp8_mx": "fp8mx",
        "fp8_sc": "fp8sc",
        "nvfp4": "nvfp4",
    }
    prec = precision_map.get(precision.lower(), precision.lower())
    variant_suffix = _flat_recipe_variant_suffix(config_variant)
    fn_name = f"{recipe}_{task}_{num_gpus}gpu_{gpu}_{prec}{variant_suffix}_config"

    mod = importlib.import_module(f"megatron.bridge.perf_recipes.{family}")
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise ValueError(f"Recipe function {fn_name!r} not found in megatron.bridge.perf_recipes.{family}")
    return _apply_flat_recipe_runtime_overrides(fn(), precision)


def _resolve_combo(combo: tuple, fallback_config_variant: str | None):
    """Return normalized combo fields plus the effective config variant."""
    if len(combo) == 6:
        family, recipe, task, num_gpus, gpu, precision = combo
        return family, recipe, task, num_gpus, gpu, precision, fallback_config_variant
    if len(combo) == 7:
        family, recipe, task, num_gpus, gpu, precision, combo_config_variant = combo
        return family, recipe, task, num_gpus, gpu, precision, combo_config_variant
    raise ValueError(f"Invalid combo length {len(combo)} for {combo!r}")


def dump_configs(mode: str, out_dir: Path, combos: list[tuple], config_variant: str | None):
    """Generate and dump all configs as YAML files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    load_fn = load_old_recipe if mode == "old" else load_new_recipe

    passed, failed = [], []
    for combo in combos:
        family, recipe, task, num_gpus, gpu, precision, combo_config_variant = _resolve_combo(combo, config_variant)
        variant_suffix = _flat_recipe_variant_suffix(combo_config_variant)
        name = f"{recipe}_{task}_{num_gpus}gpu_{gpu}_{precision}{variant_suffix}"
        yaml_path = out_dir / f"{name}.yaml"
        try:
            cfg = load_fn(family, recipe, task, num_gpus, gpu, precision, combo_config_variant)
            _dump_config_to_yaml(cfg, yaml_path)
            print(f"  OK  {name}")
            passed.append(name)
        except Exception as e:
            print(f"  ERR {name}: {e}")
            failed.append((name, str(e)))

    print(f"\n{len(passed)} OK, {len(failed)} failed")
    if failed:
        print("Failed recipes:")
        for name, err in failed:
            print(f"  {name}: {err}")
        raise SystemExit(1)


def main():
    """CLI entry-point: dump perf recipe configs for comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["old", "new"],
        required=True,
        help="'old' = use scripts/performance/configs/ (main branch), 'new' = use flat perf recipes (PR branch)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for YAML files")
    parser.add_argument("--family", help="Only dump recipes for this model family")
    parser.add_argument(
        "--config-variant",
        help="Config variant to compare against. Omit to use the suffix-less flat recipe.",
    )
    args = parser.parse_args()

    combos = COMBOS
    if args.family:
        combos = [combo for combo in combos if combo[0] == args.family]

    dump_configs(args.mode, args.out, combos, args.config_variant)


if __name__ == "__main__":
    main()
