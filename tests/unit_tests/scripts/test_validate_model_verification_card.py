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

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _load_validator():
    script = (
        Path(__file__).resolve().parents[3]
        / "skills"
        / "create-model-verification-card"
        / "scripts"
        / "validate_card.py"
    )
    spec = importlib.util.spec_from_file_location("test_model_verification_card_validator", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_index_inputs(module):
    items = {item_name: {"status": "verified"} for item_name in module.MODEL_LEVEL_INDEX_SCOPE}
    hardware_groups = {item_name: {"H100": {"status": "verified"}} for item_name in module.TRAINING_INDEX_SCOPE}
    verification_index = {
        "model_level": {"verified": list(module.MODEL_LEVEL_INDEX_SCOPE)},
        "training": {"H100": {"verified": list(module.TRAINING_INDEX_SCOPE)}},
    }
    return verification_index, items, hardware_groups


def _fsdp_metrics():
    return {
        "initial_loss": 12.19034,
        "final_loss": 3.913218,
        "last_10_steps_step_time_ms_avg": 13917.0,
        "last_10_steps_model_tflops_per_gpu_avg": 795.39,
        "peak_allocated_memory_gib": 169.54,
        "peak_reserved_memory_gib": 173.86,
    }


def _matched_non_fsdp_comparison():
    return {
        "precision": "fp8_mx",
        "cuda_graphs": "disabled",
        "global_batch_size": 384,
        "micro_batch_size": 3,
        "metrics": {
            "initial_loss": 12.18152,
            "final_loss": 1.515283,
            "last_10_steps_step_time_ms_avg": 12661.0,
            "last_10_steps_model_tflops_per_gpu_avg": 874.15,
            "peak_allocated_memory_gib": 175.29,
            "peak_reserved_memory_gib": 180.65,
        },
        "fsdp_throughput_delta_percent": -9.01,
        "fsdp_reserved_memory_delta_gib": -6.79,
    }


def test_fsdp_index_mirrors_concrete_hardware_leaf():
    module = _load_validator()
    verification_index, items, hardware_groups = _complete_index_inputs(module)
    hardware_groups["pretrain_fsdp"] = {"GB200": {"status": "verified"}}
    verification_index["fsdp"] = {"GB200": "verified"}
    errors = []

    module._validate_verification_index(
        verification_index,
        items=items,
        hardware_groups=hardware_groups,
        errors=errors,
    )

    assert errors == []


def test_fsdp_index_is_required_for_concrete_leaf():
    module = _load_validator()
    verification_index, items, hardware_groups = _complete_index_inputs(module)
    hardware_groups["pretrain_fsdp"] = {"GB200": {"status": "verified"}}
    errors = []

    module._validate_verification_index(
        verification_index,
        items=items,
        hardware_groups=hardware_groups,
        errors=errors,
    )

    assert "/verification_index/fsdp: required to mirror pretrain_fsdp concrete leaves" in errors


def test_fsdp_index_rejects_mismatched_hardware_and_status():
    module = _load_validator()
    verification_index, items, hardware_groups = _complete_index_inputs(module)
    hardware_groups["pretrain_fsdp"] = {"GB200": {"status": "verified"}}
    verification_index["fsdp"] = {"H100": "unverified"}
    errors = []

    module._validate_verification_index(
        verification_index,
        items=items,
        hardware_groups=hardware_groups,
        errors=errors,
    )

    assert "/verification_index/fsdp/GB200: required to mirror pretrain_fsdp.GB200" in errors
    assert "/verification_index/fsdp/H100: no matching pretrain_fsdp.H100 leaf" in errors


def test_fsdp_index_is_omitted_for_terminal_all_leaf():
    module = _load_validator()
    verification_index, items, hardware_groups = _complete_index_inputs(module)
    hardware_groups["pretrain_fsdp"] = {"all": {"status": "not_applicable"}}
    errors = []

    module._validate_verification_index(
        verification_index,
        items=items,
        hardware_groups=hardware_groups,
        errors=errors,
    )

    assert errors == []


def test_verified_fsdp_metrics_and_matched_control_are_valid():
    module = _load_validator()
    pretrain_item = {
        "matched_non_fsdp_comparison": _matched_non_fsdp_comparison(),
    }
    fsdp_item = {"status": "verified", "precision": "fp8_mx", "metrics": _fsdp_metrics()}
    errors = []

    module._validate_metrics(
        fsdp_item,
        item_name="pretrain_fsdp",
        item_path=("items", "pretrain_fsdp", "GB200"),
        status="verified",
        errors=errors,
    )
    module._validate_matched_non_fsdp_comparison(
        pretrain_item,
        fsdp_item,
        pretrain_path=("items", "pretrain", "GB200"),
        fsdp_path=("items", "pretrain_fsdp", "GB200"),
        errors=errors,
    )

    assert errors == []


def test_verified_fsdp_requires_matched_control_under_pretrain():
    module = _load_validator()
    fsdp_item = {"status": "verified", "precision": "fp8_mx", "metrics": _fsdp_metrics()}
    errors = []

    module._validate_matched_non_fsdp_comparison(
        {},
        fsdp_item,
        pretrain_path=("items", "pretrain", "GB200"),
        fsdp_path=("items", "pretrain_fsdp", "GB200"),
        errors=errors,
    )

    assert errors == [
        "/items/pretrain/GB200/matched_non_fsdp_comparison: required for verified /items/pretrain_fsdp/GB200"
    ]


def test_verified_fsdp_metrics_require_peak_memory():
    module = _load_validator()
    item = {"metrics": _fsdp_metrics()}
    del item["metrics"]["peak_allocated_memory_gib"]
    del item["metrics"]["peak_reserved_memory_gib"]
    errors = []

    module._validate_metrics(
        item,
        item_name="pretrain_fsdp",
        item_path=("items", "pretrain_fsdp", "GB200"),
        status="verified",
        errors=errors,
    )

    assert "/items/pretrain_fsdp/GB200/metrics/peak_allocated_memory_gib: required key is missing" in errors
    assert "/items/pretrain_fsdp/GB200/metrics/peak_reserved_memory_gib: required key is missing" in errors


def test_fsdp_comparison_rejects_inconsistent_deltas():
    module = _load_validator()
    comparison = _matched_non_fsdp_comparison()
    comparison["fsdp_throughput_delta_percent"] = 9.01
    comparison["fsdp_reserved_memory_delta_gib"] = 6.79
    pretrain_item = {"matched_non_fsdp_comparison": comparison}
    fsdp_item = {"status": "verified", "precision": "fp8_mx", "metrics": _fsdp_metrics()}
    errors = []

    module._validate_matched_non_fsdp_comparison(
        pretrain_item,
        fsdp_item,
        pretrain_path=("items", "pretrain", "GB200"),
        fsdp_path=("items", "pretrain_fsdp", "GB200"),
        errors=errors,
    )

    assert any("fsdp_throughput_delta_percent" in error and "expected -9.01" in error for error in errors)
    assert any("fsdp_reserved_memory_delta_gib" in error and "expected -6.79" in error for error in errors)


def test_megatron_fsdp_feature_is_scoped_to_fsdp_item():
    module = _load_validator()
    features = {"moe_dispatcher": "hybridep", "megatron_fsdp": "optim_grads_params"}
    fsdp_errors = []
    pretrain_errors = []

    module._validate_enabled_features(
        features,
        item_name="pretrain_fsdp",
        item_path=("items", "pretrain_fsdp", "GB200"),
        errors=fsdp_errors,
    )
    module._validate_enabled_features(
        features,
        item_name="pretrain",
        item_path=("items", "pretrain", "GB200"),
        errors=pretrain_errors,
    )

    assert fsdp_errors == []
    assert pretrain_errors == ["/items/pretrain/GB200/enabled_features/megatron_fsdp: allowed only on pretrain_fsdp"]
