# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Keep builder-backed model construction as the default in examples."""

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

EXAMPLES_ROOT = Path(__file__).parents[3] / "examples"

# Remove entries as these integrations gain standalone ModelConfig/ModelBuilder
# support or no longer require a persistent provider-based HF loading hook.
TEMPORARY_FALLBACKS = {
    "to_megatron_provider": {
        Path("distillation/llama/distill_llama32_3b-1b.py"),
        Path("megatron_mimo/qwen35_vl/finetune_qwen35_vl.py"),
        Path("models/diffusion/inference_dllm.py"),
        Path("rl/rlhf_with_bridge.py"),
    },
    "provider_bridge": {
        Path("models/flux/conversion/convert_checkpoints.py"),
        Path("models/nemotron_labs_diffusion/inference_nemotron_labs_diffusion.py"),
        Path("models/wan/conversion/convert_checkpoints.py"),
    },
    "provide_distributed_model": {
        Path("models/flux/conversion/convert_checkpoints.py"),
        Path("models/wan/conversion/convert_checkpoints.py"),
    },
}


def test_examples_only_use_allowlisted_provider_fallbacks() -> None:
    calls: dict[str, set[Path]] = {name: set() for name in TEMPORARY_FALLBACKS}

    for path in EXAMPLES_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in calls:
                calls[node.func.attr].add(path.relative_to(EXAMPLES_ROOT))

    assert calls == TEMPORARY_FALLBACKS


def test_examples_do_not_use_removed_model_construction_aliases() -> None:
    forbidden_calls = {"to_megatron_model", "to_megatron_model_config"}
    violations: list[str] = []

    for path in EXAMPLES_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in forbidden_calls:
                relative_path = path.relative_to(EXAMPLES_ROOT)
                violations.append(f"{relative_path}:{node.lineno}:{node.func.attr}")

    assert not violations, "Deprecated model construction aliases remain: " + ", ".join(violations)
