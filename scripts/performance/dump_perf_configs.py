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

Both modes serialize the ConfigContainer via its to_dict() / dataclasses.asdict path
(same as save_config_filepath in production), so the YAML is directly comparable.
"""

import argparse
import importlib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# All (family, recipe, task, num_gpus, gpu, precision) combos that exist in
# both the old scripts/performance/configs/ and the new flat perf recipes.
# Add entries here whenever a new recipe is added to either path.
# ---------------------------------------------------------------------------
COMBOS = [
    # Llama 3
    ("llama", "llama3_8b", "pretrain", 64, "h100", "bf16"),
    ("llama", "llama3_8b", "pretrain", 64, "h100", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 64, "b200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 64, "b200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 32, "gb200", "bf16"),
    ("llama", "llama3_8b", "pretrain", 32, "gb200", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "bf16"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "fp8_cs"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "fp8_mx"),
    ("llama", "llama3_8b", "pretrain", 32, "gb300", "nvfp4"),
    # Llama 3 70B
    ("llama", "llama3_70b", "pretrain", 64, "h100", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "h100", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 64, "b200", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 32, "gb200", "bf16"),
    ("llama", "llama3_70b", "pretrain", 32, "gb200", "fp8_cs"),
    ("llama", "llama3_70b", "pretrain", 32, "gb300", "bf16"),
    ("llama", "llama3_70b", "pretrain", 32, "gb300", "fp8_cs"),
    # Llama 3.1 405B
    ("llama", "llama31_405b", "pretrain", 512, "h100", "bf16"),
    ("llama", "llama31_405b", "pretrain", 512, "h100", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "b200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "gb200", "fp8_cs"),
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "bf16"),
    ("llama", "llama31_405b", "pretrain", 256, "gb300", "fp8_cs"),
    # DeepSeek V3
    ("deepseek", "deepseek_v3", "pretrain", 64, "h100", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 64, "h100", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "b200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb200", "fp8_cs"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "bf16"),
    ("deepseek", "deepseek_v3", "pretrain", 256, "gb300", "fp8_cs"),
    # Qwen3 MoE 30B-A3B
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "h100", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "h100", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "b200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb200", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb200", "fp8_cs"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb300", "bf16"),
    ("qwen", "qwen3_30b_a3b", "pretrain", 32, "gb300", "fp8_cs"),
    # Qwen3 MoE 235B-A22B
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "h100", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "h100", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "b200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb200", "fp8_cs"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "bf16"),
    ("qwen", "qwen3_235b_a22b", "pretrain", 256, "gb300", "fp8_cs"),
    # Kimi K2
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb300", "nvfp4"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "gb200", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "fp8_cs"),
    ("kimi", "kimi_k2", "pretrain", 256, "b200", "fp8_mx"),
    ("kimi", "kimi_k2", "pretrain", 1024, "h100", "bf16"),
    ("kimi", "kimi_k2", "pretrain", 1024, "h100", "fp8_cs"),
    # NemotronH
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "b200", "bf16"),
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "b200", "fp8_cs"),
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "gb300", "bf16"),
    ("nemotronh", "nemotronh_56b", "pretrain", 256, "gb300", "fp8_cs"),
    # GPT-OSS 120B
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb300", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb200", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b300", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b200", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "h100", "bf16"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "gb200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b300", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "b200", "fp8_mx"),
    ("gpt_oss", "gpt_oss_120b", "pretrain", 64, "h100", "fp8_mx"),
    # WAN 14B (diffusion — BF16 only)
    ("wan", "wan_14b", "pretrain", 16, "gb200", "bf16"),
    ("wan", "wan_14b", "pretrain", 32, "h100", "bf16"),
    # Qwen3.5-VL 35B-A3B
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 8, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 16, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_35b_a3b", "pretrain", 16, "h100", "fp8_cs"),
    # Qwen3.5-VL 122B-A10B
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 32, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 128, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_122b_a10b", "pretrain", 128, "h100", "fp8_cs"),
    # Qwen3.5-VL 397B-A17B
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b300", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "gb200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "fp8_cs"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 64, "b200", "fp8_mx"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 256, "h100", "bf16"),
    ("qwen_vl", "qwen35_vl_397b_a17b", "pretrain", 256, "h100", "fp8_cs"),
]


def _dump_config_to_yaml(cfg, yaml_path: Path) -> None:
    """Dump a ConfigContainer to YAML using the production to_yaml() path."""
    cfg.to_yaml(str(yaml_path))


def load_old_recipe(family: str, recipe: str, task: str, num_gpus: int, gpu: str, precision: str):
    """Load recipe using the OLD scripts/performance/configs/ path (main branch)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.utils import get_perf_optimized_recipe

    return get_perf_optimized_recipe(
        model_family_name=family,
        model_recipe_name=recipe,
        train_task=task,
        gpu=gpu,
        compute_dtype=precision,
    )


def load_new_recipe(family: str, recipe: str, task: str, num_gpus: int, gpu: str, precision: str):
    """Load recipe using the NEW flat perf recipe path (PR branch)."""
    precision_map = {
        "bf16": "bf16",
        "fp8_cs": "fp8cs",
        "fp8_mx": "fp8mx",
        "nvfp4": "nvfp4",
    }
    prec = precision_map.get(precision.lower(), precision.lower())
    fn_name = f"{recipe}_{task}_{num_gpus}gpu_{gpu}_{prec}_config"

    mod = importlib.import_module(f"megatron.bridge.recipes.{family}")
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise ValueError(f"Recipe function {fn_name!r} not found in megatron.bridge.recipes.{family}")
    return fn()


def dump_configs(mode: str, out_dir: Path):
    """Generate and dump all configs as YAML files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    load_fn = load_old_recipe if mode == "old" else load_new_recipe

    passed, failed = [], []
    for family, recipe, task, num_gpus, gpu, precision in COMBOS:
        name = f"{recipe}_{task}_{num_gpus}gpu_{gpu}_{precision}"
        yaml_path = out_dir / f"{name}.yaml"
        try:
            cfg = load_fn(family, recipe, task, num_gpus, gpu, precision)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["old", "new"],
        required=True,
        help="'old' = use scripts/performance/configs/ (main branch), 'new' = use flat perf recipes (PR branch)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for YAML files")
    parser.add_argument("--family", help="Only dump recipes for this model family")
    args = parser.parse_args()

    combos = COMBOS
    if args.family:
        combos = [(f, r, t, n, g, p) for (f, r, t, n, g, p) in combos if f == args.family]

    dump_configs(args.mode, args.out)


if __name__ == "__main__":
    main()
