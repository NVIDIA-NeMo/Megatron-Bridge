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

"""Smoke tests for the Llama2-70B LoRA (gov_report) rc5 parity overrides.

These pin the parity-relevant recipe-knob deviations that let the gov_report LoRA
workload train on the mbridge 26.06.rc5 stack and that define the measured rc5
throughput (cuda-graph off, fused_single_qkv_rope off, nvfp4 selective recompute).
The logic lives in ``overrides._apply_gov_report_recipe_overrides`` so it can be
exercised here without building a real recipe, a dataset, or touching a GPU.
"""

import sys
import types
from pathlib import Path

import pytest

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

try:
    from utils.overrides import _apply_gov_report_recipe_overrides

    HAS_OVERRIDES = True
except Exception:  # noqa: BLE001 - module pulls in megatron.bridge; only import in-container
    HAS_OVERRIDES = False

pytestmark = pytest.mark.skipif(
    not HAS_OVERRIDES, reason="scripts/performance/utils/overrides deps not importable"
)


def _make_recipe(**model_overrides):
    """Minimal stand-in recipe exposing only the attributes the helper touches."""
    model = types.SimpleNamespace(
        seq_length=8192,
        context_parallel_size=1,
        fused_single_qkv_rope=True,
        cuda_graph_impl="local",
        cuda_graph_scope="full_iteration",
        cuda_graph_modules=["attn"],
        recompute_modules=["core_attn"],
        recompute_granularity=None,
        calculate_per_token_loss=False,
    )
    for key, value in model_overrides.items():
        setattr(model, key, value)
    return types.SimpleNamespace(model=model)


@pytest.fixture(autouse=True)
def _clean_gov_report_env(monkeypatch):
    for var in (
        "LORA_GOVREPORT_TRY_CUDA_GRAPH",
        "LORA_GOVREPORT_NO_RECOMPUTE",
        "LORA_GOVREPORT_RECOMPUTE_ATTN_ONLY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_cuda_graph_and_fused_qkv_disabled_fp8_cp1():
    recipe = _make_recipe()
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="fp8_cs", cp_size=1)

    # CUDA graphs must be fully cleared (impl AND scope AND per-layer modules), else
    # config finalization re-derives full_iteration and the unpinned-copy crash returns.
    assert recipe.model.cuda_graph_impl == "none"
    assert recipe.model.cuda_graph_scope is None
    assert recipe.model.cuda_graph_modules == []
    assert recipe.model.fused_single_qkv_rope is False
    # cp==1 -> no per-token loss forced; fp8 -> no recompute activation.
    assert recipe.model.calculate_per_token_loss is False
    assert recipe.model.recompute_granularity is None


def test_per_token_loss_forced_when_cp_gt_1():
    recipe = _make_recipe(context_parallel_size=2)
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="fp8_cs", cp_size=2)

    assert recipe.model.calculate_per_token_loss is True
    # fp8 attention stays fp8 (fits) -> no recompute even at cp>1.
    assert recipe.model.recompute_granularity is None


def test_nvfp4_cp2_activates_core_attn_mlp_recompute():
    recipe = _make_recipe(context_parallel_size=2)
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="nvfp4", cp_size=2)

    assert recipe.model.recompute_granularity == "selective"
    assert recipe.model.recompute_modules == ["core_attn", "mlp"]
    assert recipe.model.calculate_per_token_loss is True


def test_nvfp4_cp1_recomputes_core_attn_and_mlp():
    # nvfp4 has no fp8-DPA backend on this stack at ANY cp (confirmed at cp1, job 2089391),
    # so attention falls back to bf16 and needs recompute even at cp1. core_attn-only OOMs the
    # cp1/mbs1/seq8192 shape at TP=1 (job 2089596), so we recompute core_attn+mlp, which fits
    # and lands ~3105 TFLOP/s/GPU (job 2089680).
    recipe = _make_recipe(context_parallel_size=1)
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="nvfp4", cp_size=1)

    assert recipe.model.recompute_granularity == "selective"
    assert recipe.model.recompute_modules == ["core_attn", "mlp"]


def test_recompute_attn_only_env_override(monkeypatch):
    monkeypatch.setenv("LORA_GOVREPORT_RECOMPUTE_ATTN_ONLY", "1")
    recipe = _make_recipe(context_parallel_size=2)
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="nvfp4", cp_size=2)

    assert recipe.model.recompute_modules == ["core_attn"]
    assert recipe.model.recompute_granularity == "selective"


def test_no_recompute_env_override(monkeypatch):
    monkeypatch.setenv("LORA_GOVREPORT_NO_RECOMPUTE", "1")
    recipe = _make_recipe(context_parallel_size=2)
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="nvfp4", cp_size=2)

    assert recipe.model.recompute_granularity is None


def test_try_cuda_graph_env_keeps_graphs_on(monkeypatch):
    monkeypatch.setenv("LORA_GOVREPORT_TRY_CUDA_GRAPH", "1")
    recipe = _make_recipe()
    _apply_gov_report_recipe_overrides(recipe, compute_dtype="fp8_cs", cp_size=1)

    # Opt-in escape hatch: leave whatever the base config set untouched.
    assert recipe.model.cuda_graph_impl == "local"
    assert recipe.model.cuda_graph_scope == "full_iteration"
