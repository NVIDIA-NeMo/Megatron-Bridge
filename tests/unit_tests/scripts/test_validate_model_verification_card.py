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


@pytest.mark.parametrize("token", ["<|im_start|>", "<|im_end|>", "<think>", "</think>"])
def test_literal_chat_tokens_are_not_placeholders_or_paths(token):
    module = _load_validator()
    errors = []

    assert not module._contains_placeholder(token)
    module._validate_privacy(token, {}, (), errors)

    assert errors == []


@pytest.mark.parametrize(
    "value",
    [
        "TODO",
        "TBD",
        "PLACEHOLDER",
        "<model>",
        "<|unknown|>",
        "<Think>",
        "</other>",
        "<think>TODO</think>",
        "<|im_start|><model><|im_end|>",
    ],
)
def test_chat_token_exception_does_not_hide_placeholders(value):
    assert _load_validator()._contains_placeholder(value)


@pytest.mark.parametrize("address", ["127.0.0.1", "127.2.3.4", "::1", "0:0:0:0:0:0:0:1", "[::1]"])
def test_privacy_accepts_loopback_coordinator(address):
    module = _load_validator()
    errors = []
    command = f"./scripts/inference/infer.sh --coordinator-host {address}"

    module._validate_privacy(command, {"items": {"inference": {"command": command}}}, (), errors)

    assert errors == []


@pytest.mark.parametrize(
    "address",
    [
        "10.2.3.4",
        "172.16.0.1",
        "192.168.1.2",
        "8.8.8.8",
        "0.0.0.0",
        "128.0.0.1",
        "127.0.0.999",
        "127.00.0.1",
        "::",
        "fe80::1",
        "fd00::1",
        "2001:db8::1",
        "::ffff:10.2.3.4",
        "[::ffff:7f00:1]",
        "[::ffff:127.0.0.1]",
    ],
)
def test_loopback_exception_does_not_hide_other_addresses(address):
    module = _load_validator()
    errors = []

    module._validate_privacy(f"127.0.0.1 ::1 <think>{address}</think>", {}, (), errors)

    assert "/: email or IP address detected" in errors


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("<think>/private/checkpoint</think>", "absolute path detected"),
        ("</think>/private/checkpoint", "absolute path detected"),
        ("<think>~/checkpoint</think>", "home-directory paths are forbidden"),
        ("<think>${PRIVATE_ROOT}</think>", "environment reference(s) detected"),
        ("<think>user@example.com</think>", "email or IP address detected"),
        ("http://127.0.0.1", "URLs are forbidden"),
        ("<think>HF_TOKEN=example</think>", "secret assignment detected"),
    ],
)
def test_chat_and_loopback_exceptions_preserve_privacy_checks(value, error):
    module = _load_validator()
    errors = []

    module._validate_privacy(value, {}, (), errors)

    assert any(error in message for message in errors)


def test_chat_token_exception_preserves_explicit_deny_terms():
    module = _load_validator()
    errors = []

    module._validate_privacy("<think>127.0.0.1</think>", {}, ("think", "127.0.0.1"), errors)

    assert errors == ["/: matched caller-supplied deny term #1", "/: matched caller-supplied deny term #2"]


def test_verified_inference_accepts_literal_chat_prompt_and_loopback():
    module = _load_validator()
    prompt = "<|im_start|>user\\nWhat is two plus two?<|im_end|>\\n<|im_start|>assistant\\n<think></think>"
    item = {
        "status": "verified",
        "precision": "bf16",
        "command": (
            "./scripts/inference/infer.sh --nodes 1 --gpus-per-node 4 --task text-generation "
            f"--prompt $'{prompt}' --coordinator-host 127.0.0.1 --max_new_tokens 32 "
            "--temperature 0 --top-k 1"
        ),
        "last_verified": "2026-09-27",
        "expected_result": (
            "One greedy run with a 32-token maximum generated 7 tokens and stopped at EOS. "
            'The literal completion was "Two plus two equals four."'
        ),
    }
    errors = []

    module._validate_item("inference", item, errors, path=("items", "inference"), model_revision=None)
    module._validate_privacy(item["command"], {"items": {"inference": item}}, (), errors)

    assert errors == []


def _fsdp_metrics():
    return {
        "initial_loss": 12.19034,
        "final_loss": 3.913218,
        "last_10_steps_step_time_ms_avg": 13917.0,
        "last_10_steps_model_tflops_per_gpu_avg": 795.39,
        "last_10_steps_tokens_per_second_per_gpu_avg": 28254.365,
        "peak_allocated_memory_gib": 169.54,
        "peak_reserved_memory_gib": 173.86,
    }


def test_verified_inference_accepts_canonical_bash_launcher():
    module = _load_validator()
    errors = []
    item = {
        "expected_result": 'The exact 1-token result produced completion "ok".',
    }

    module._validate_inference(
        item,
        item_name="inference",
        status="verified",
        errors=errors,
        command_override=(
            "./scripts/inference/infer.sh --nodes 4 --gpus-per-node 8 "
            "--task legacy-full-prefix-generation --legacy-full-prefix --prompt hello --max_new_tokens 1"
        ),
    )

    assert errors == []


def test_verified_inference_legacy_task_requires_full_prefix_flag():
    module = _load_validator()
    errors = []
    item = {
        "expected_result": 'The exact 1-token result produced completion "ok".',
    }

    module._validate_inference(
        item,
        item_name="inference",
        status="verified",
        errors=errors,
        command_override=(
            "./scripts/inference/infer.sh --nodes 4 --gpus-per-node 8 "
            "--task legacy-full-prefix-generation --prompt hello --max_new_tokens 1"
        ),
    )

    assert "/items/inference/command: legacy-full-prefix-generation requires --legacy-full-prefix" in errors


def test_verified_inference_bash_launcher_requires_task_and_resources():
    module = _load_validator()
    errors = []
    item = {
        "expected_result": 'The exact 1-token result produced completion "ok".',
    }

    module._validate_inference(
        item,
        item_name="inference",
        status="verified",
        errors=errors,
        command_override="./scripts/inference/infer.sh --prompt hello --max_new_tokens 1",
    )

    assert "/items/inference/command: infer.sh must specify one supported --task" in errors
    assert "/items/inference/command: infer.sh must specify --nodes exactly once" in errors
    assert "/items/inference/command: infer.sh must specify --gpus-per-node exactly once" in errors


def test_verified_manual_forward_pass_accepts_canonical_bash_launcher():
    module = _load_validator()
    errors = []
    item = {
        "status": "verified",
        "command": (
            "./scripts/inference/infer.sh --nodes 4 --gpus-per-node 8 --task model-comparison "
            "--hf_model_path org/model --hf-revision revision "
            "--megatron_model_path work/model/iter_0000000 --prompt hello"
        ),
        "last_verified": "2026-07-31",
        "expected_result": (
            "The next token matches. Cosine similarity is 0.99. "
            "Maximum absolute logit difference is 0.1. Mean absolute logit difference is 0.01."
        ),
    }

    module._validate_manual_forward_pass(
        item,
        status="verified",
        model_revision="revision",
        errors=errors,
    )

    assert errors == []


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


def test_verified_fsdp_metrics_are_valid():
    module = _load_validator()
    fsdp_item = {"status": "verified", "precision": "fp8_mx", "metrics": _fsdp_metrics()}
    errors = []

    module._validate_metrics(
        fsdp_item,
        item_name="pretrain_fsdp",
        item_path=("items", "pretrain_fsdp", "GB200"),
        status="verified",
        errors=errors,
    )

    assert errors == []


def test_verified_training_metrics_require_tps_per_gpu():
    module = _load_validator()
    item = {"metrics": _fsdp_metrics()}
    del item["metrics"]["last_10_steps_tokens_per_second_per_gpu_avg"]
    errors = []

    module._validate_metrics(
        item,
        item_name="pretrain",
        item_path=("items", "pretrain", "GB200"),
        status="verified",
        errors=errors,
    )

    assert (
        "/items/pretrain/GB200/metrics/last_10_steps_tokens_per_second_per_gpu_avg: required key is missing" in errors
    )


def test_verified_fsdp_variant_requires_positive_tps_per_gpu():
    module = _load_validator()
    metrics = _fsdp_metrics()
    metrics["last_10_steps_tokens_per_second_per_gpu_avg"] = 0
    errors = []

    module._validate_metrics(
        {"metrics": metrics},
        item_name="pretrain_fsdp",
        item_path=("items", "pretrain_fsdp", "GB200", "variants", "fp8_mx"),
        status="verified",
        errors=errors,
    )

    assert (
        "/items/pretrain_fsdp/GB200/variants/fp8_mx/metrics/"
        "last_10_steps_tokens_per_second_per_gpu_avg: verified performance metrics must be positive"
    ) in errors


def test_fsdp_hardware_leaf_accepts_multiple_precision_variants():
    module = _load_validator()
    errors = []
    variant = {
        "status": "verified",
        "precision": "bf16",
        "command": "./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --max_steps 20",
        "last_verified": "2026-07-27",
        "expected_result": "The run completes with finite losses.",
        "enabled_features": {"megatron_fsdp": "optim_grads_params"},
        "metrics": _fsdp_metrics(),
    }

    module._validate_fsdp_variant_group(
        {
            "status": "verified",
            "variants": {
                "bf16": variant,
                "fp8_mx": {**variant, "precision": "fp8_mx"},
            },
        },
        path=("items", "pretrain_fsdp", "GB200"),
        model_revision=None,
        errors=errors,
    )

    assert errors == []


def test_fsdp_variant_group_status_summarizes_all_variants():
    module = _load_validator()
    errors = []
    variant = {
        "status": "unverified",
        "precision": "bf16",
        "command": None,
        "last_verified": None,
        "expected_result": None,
        "enabled_features": {"megatron_fsdp": "optim_grads_params"},
        "metrics": {},
    }

    module._validate_fsdp_variant_group(
        {
            "status": "verified",
            "variants": {"bf16": variant},
        },
        path=("items", "pretrain_fsdp", "GB200"),
        model_revision=None,
        errors=errors,
    )

    assert "/items/pretrain_fsdp/GB200/status: must be unverified to summarize the precision variants" in errors


def test_fsdp_variant_precision_matches_mapping_key():
    module = _load_validator()
    errors = []
    variant = {
        "status": "verified",
        "precision": "fp8_mx",
        "command": "./scripts/training/train.sh --nodes 2 --gpus-per-node 4 --max_steps 20",
        "last_verified": "2026-07-27",
        "expected_result": "The run completes with finite losses.",
        "enabled_features": {"megatron_fsdp": "optim_grads_params"},
        "metrics": _fsdp_metrics(),
    }

    module._validate_fsdp_variant_group(
        {
            "status": "verified",
            "variants": {"bf16": variant},
        },
        path=("items", "pretrain_fsdp", "GB200"),
        model_revision=None,
        errors=errors,
    )

    assert "/items/pretrain_fsdp/GB200/variants/bf16/precision: must match the variant key" in errors


def test_performance_comparison_payload_is_not_an_item_key():
    module = _load_validator()
    errors = []

    module._check_keys(
        {"matched_non_fsdp_comparison": {}},
        allowed=module.ITEM_KEYS,
        required=frozenset(),
        path=("items", "pretrain", "GB200"),
        errors=errors,
    )

    assert errors == ["/items/pretrain/GB200/matched_non_fsdp_comparison: unknown key"]


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
