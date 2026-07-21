# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import ast
import copy
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "skills/create-model-verification-card/scripts/validate_card.py"
COMPARE_PATH = REPO_ROOT / "examples/conversion/compare_hf_and_megatron/compare.py"
CARD_PATH = REPO_ROOT / "examples/model_verification_cards/nemotron-3-nano-4b/card.yaml"
CORRELATION_CARD_PATH = REPO_ROOT / "examples/model_verification_cards/qwen3-8b/card.yaml"
PERFORMANCE_CARD_PATH = REPO_ROOT / "examples/model_verification_cards/qwen3-30b-a3b/card.yaml"
CARD_PATHS = sorted((REPO_ROOT / "examples/model_verification_cards").glob("*/card.yaml"))
EXPECTED_CARD_SLUGS = {"moonlight-16b-a3b", "nemotron-3-nano-4b", "qwen3-30b-a3b", "qwen3-8b"}


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_card_validator", VALIDATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _card(path: Path = CARD_PATH) -> dict[str, Any]:
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(card, dict)
    return card


def _errors(card: dict[str, Any]) -> list[str]:
    raw = yaml.safe_dump(card, sort_keys=False)
    return VALIDATOR._validate_card(card, raw, ())


def _assert_error(card: dict[str, Any], fragment: str) -> None:
    errors = _errors(card)
    assert any(fragment in error for error in errors), errors


def _hardware_item(card: dict[str, Any], item_name: str, hardware: str = "H100") -> dict[str, Any]:
    item = card["items"][item_name][hardware]
    assert isinstance(item, dict)
    return item


def _assigned_float(path: Path, name: str) -> float:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, float)
            return value
    raise AssertionError(f"{name} is not assigned in {path}")


@pytest.mark.parametrize("card_path", CARD_PATHS, ids=lambda path: path.parent.name)
def test_repository_model_verification_cards_are_valid(card_path: Path) -> None:
    assert _errors(_card(card_path)) == []


def test_repository_model_verification_card_inventory_is_discovered() -> None:
    assert {path.parent.name for path in CARD_PATHS} == EXPECTED_CARD_SLUGS


@pytest.mark.parametrize("card_path", CARD_PATHS, ids=lambda path: path.parent.name)
def test_repository_verification_indexes_include_h100_and_gb200(card_path: Path) -> None:
    training_index = _card(card_path)["verification_index"]["training"]
    assert {"H100", "GB200"} <= set(training_index)


def test_verification_index_is_required() -> None:
    card = _card()
    del card["verification_index"]
    _assert_error(card, "/verification_index: required key is missing")


def test_verification_index_rejects_empty_status_bucket() -> None:
    card = _card()
    card["verification_index"]["training"]["GB200"] = {"unverified": []}
    _assert_error(card, "expected all or a non-empty item list")


def test_verification_index_rejects_model_level_drift() -> None:
    card = _card()
    card["verification_index"]["model_level"] = {"unverified": "all"}
    _assert_error(card, "hf_to_megatron_cpu is indexed as unverified but detailed items project to verified")


def test_verification_index_rejects_training_drift_and_invalid_all() -> None:
    card = _card(CORRELATION_CARD_PATH)
    card["verification_index"]["training"]["H100"] = {"verified": "all"}
    _assert_error(card, "all is allowed only when every detailed item in the scope has status verified")
    _assert_error(card, "pretrain is indexed as verified but detailed items project to unverified")


def test_verification_index_requires_each_concrete_training_hardware() -> None:
    card = _card()
    card["items"]["pretrain"]["B200"] = copy.deepcopy(_hardware_item(card, "pretrain"))
    _assert_error(card, "training/B200: required because detailed training items use this hardware")


def test_verification_index_allows_an_unverified_future_hardware_target() -> None:
    card = _card()
    card["verification_index"]["training"]["B200"] = {"unverified": "all"}
    assert _errors(card) == []


def test_verification_index_performance_exactly_mirrors_concrete_leaves() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    del card["verification_index"]["performance"]
    _assert_error(card, "performance: required to mirror pretrain_performance concrete leaves")

    card = _card(PERFORMANCE_CARD_PATH)
    card["verification_index"]["performance"]["H100"] = "unverified"
    _assert_error(card, "pretrain_performance.H100 is verified")

    card = _card()
    card["verification_index"]["performance"] = {"H100": "unverified"}
    _assert_error(card, "omit performance when pretrain_performance has no concrete leaves")


def test_verification_index_rejects_duplicate_scope_item() -> None:
    card = _card(CORRELATION_CARD_PATH)
    card["verification_index"]["training"]["H100"]["verified"].append("pretrain")
    _assert_error(card, "pretrain must appear exactly once in this scope")


@pytest.mark.parametrize("card_path", CARD_PATHS, ids=lambda path: path.parent.name)
def test_repository_training_items_are_hardware_scoped(card_path: Path) -> None:
    card = _card(card_path)
    for item_name in VALIDATOR.HARDWARE_SCOPED_ITEMS:
        if item_name not in card["items"]:
            continue
        variants = card["items"][item_name]
        assert variants
        assert set(variants) <= VALIDATOR.PUBLIC_HARDWARE_KEYS | {"all"}
        assert all("gpu_type" not in leaf for leaf in variants.values())


def test_untuned_card_requires_opening_performance_disclaimer() -> None:
    card = _card()
    disclaimer = VALIDATOR.UNTUNED_PERFORMANCE_DISCLAIMER
    assert card["summary"].startswith(disclaimer)

    card["summary"] = card["summary"].removeprefix(disclaimer).strip()
    _assert_error(card, "must start with the untuned performance disclaimer")

    card = _card()
    card["summary"] = f"{card['summary'].removeprefix(disclaimer).strip()} {disclaimer}"
    _assert_error(card, "must start with the untuned performance disclaimer")

    card = _card()
    card["summary"] = card["summary"].replace("; reported", ";\n  reported")
    assert _errors(card) == []


def test_verified_performance_card_scopes_its_tuned_result() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    assert _hardware_item(card, "pretrain_performance")["status"] == "verified"
    assert card["summary"].startswith("Performance scope: only pretrain_performance.H100")
    assert _errors(card) == []

    card["summary"] = "Qwen3-30B-A3B has a tuned canonical performance recipe."
    _assert_error(card, "scope the tuned claim to pretrain_performance.H100")


def test_unverified_canonical_performance_recipe_does_not_require_disclaimer() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    performance = _hardware_item(card, "pretrain_performance")
    performance["status"] = "unverified"
    performance.pop("bridge_commit")
    card["verification_index"]["performance"]["H100"] = "unverified"
    assert _errors(card) == []


def test_canonical_performance_recipe_rejects_stale_untuned_disclaimer() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    card["summary"] = f"{VALIDATOR.UNTUNED_PERFORMANCE_DISCLAIMER} {card['summary']}"
    _assert_error(card, "remove the untuned performance disclaimer")


def test_performance_recipe_cannot_use_global_terminal_placeholder() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    performance = _hardware_item(card, "pretrain_performance")
    performance.update(
        status="unsupported",
        precision=None,
        command=None,
        last_verified=None,
        metrics={name: None for name in VALIDATOR.METRIC_NAMES},
        expected_result="No canonical performance recipe is available for this model.",
    )
    performance.pop("bridge_commit")
    card["items"]["pretrain_performance"] = {"all": performance}
    _assert_error(card, "omit pretrain_performance when no canonical hardware recipe exists")


@pytest.mark.parametrize("status", ["unsupported", "not_applicable"])
def test_concrete_performance_recipe_rejects_terminal_status(status: str) -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    performance = _hardware_item(card, "pretrain_performance")
    performance.update(
        status=status,
        precision=None,
        command=None,
        last_verified=None,
        metrics={name: None for name in VALIDATOR.METRIC_NAMES},
        expected_result="No canonical performance recipe is available for this hardware.",
    )
    performance.pop("bridge_commit")
    _assert_error(card, "a canonical performance recipe must be verified or unverified")


def test_summary_scopes_each_canonical_performance_hardware() -> None:
    card = _card(PERFORMANCE_CARD_PATH)
    card["items"]["pretrain_performance"]["B200"] = copy.deepcopy(_hardware_item(card, "pretrain_performance"))
    card["verification_index"]["performance"]["B200"] = "verified"
    _assert_error(card, "scope the tuned claim to pretrain_performance.B200")

    card["summary"] += " pretrain_performance.B200 is also a tuned canonical recipe."
    assert _errors(card) == []


def test_flat_training_item_is_rejected() -> None:
    card = _card()
    card["items"]["pretrain"] = _hardware_item(card, "pretrain")
    _assert_error(card, "hardware-scoped items must be keyed by hardware")


def test_hardware_group_must_be_nonempty_and_canonical() -> None:
    card = _card()
    card["items"]["pretrain"] = {}
    _assert_error(card, "expected at least one hardware entry")

    card = _card()
    card["items"]["pretrain"]["private_cluster"] = card["items"]["pretrain"].pop("H100")
    _assert_error(card, "expected a supported public hardware key")

    card = _card()
    card["items"]["pretrain"]["FOO"] = card["items"]["pretrain"].pop("H100")
    _assert_error(card, "expected a supported public hardware key")


def test_hardware_key_replaces_gpu_type() -> None:
    card = _card()
    _hardware_item(card, "pretrain")["gpu_type"] = "H100"
    _assert_error(card, "/items/pretrain/H100/gpu_type: unknown key")


def test_hardware_dependencies_do_not_fall_back_across_variants() -> None:
    card = _card()
    card["items"]["checkpoint_resume"]["B200"] = copy.deepcopy(_hardware_item(card, "checkpoint_resume"))
    _assert_error(card, "/items/checkpoint_resume/B200/depends_on: missing pretrain.B200")

    card = _card()
    card["items"]["sft_export_inference"]["B200"] = copy.deepcopy(_hardware_item(card, "sft_export_inference"))
    _assert_error(card, "/items/sft_export_inference/B200/depends_on: missing sft.B200")


def test_additional_hardware_variants_validate_independently() -> None:
    card = _card()
    for item_name in ("pretrain", "checkpoint_resume", "sft", "sft_export_inference"):
        card["items"][item_name]["B200"] = copy.deepcopy(_hardware_item(card, item_name))
    card["verification_index"]["training"]["B200"] = {
        "verified": ["pretrain", "sft", "sft_export_inference", "checkpoint_resume"],
        "unverified": ["sft_long_context", "peft"],
    }
    assert _errors(card) == []


def test_global_hardware_key_is_terminal_only() -> None:
    card = _card()
    card["items"]["pretrain"]["all"] = card["items"]["pretrain"].pop("H100")
    _assert_error(card, "all is reserved for global terminal limitations")


def test_global_terminal_dependencies_do_not_require_global_sources() -> None:
    card = _card()
    resume = copy.deepcopy(_hardware_item(card, "checkpoint_resume"))
    resume.update(
        status="unsupported",
        precision=None,
        command=None,
        last_verified=None,
        metrics={name: None for name in VALIDATOR.METRIC_NAMES},
        resume_comparison=None,
        expected_result="Checkpoint resume is not supported for this model.",
    )
    resume.pop("bridge_commit", None)
    card["items"]["checkpoint_resume"] = {"all": resume}

    export = copy.deepcopy(_hardware_item(card, "sft_export_inference"))
    export.update(
        status="not_applicable",
        precision=None,
        commands=None,
        last_verified=None,
        expected_result="SFT export is not applicable to this model.",
    )
    export.pop("bridge_commit", None)
    card["items"]["sft_export_inference"] = {"all": export}

    card["verification_index"]["training"]["H100"] = {
        "verified": ["pretrain", "sft", "sft_long_context", "peft"],
        "unsupported": ["checkpoint_resume"],
        "not_applicable": ["sft_export_inference"],
    }
    card["verification_index"]["training"]["GB200"] = {
        "unverified": ["pretrain", "sft", "sft_long_context", "peft"],
        "unsupported": ["checkpoint_resume"],
        "not_applicable": ["sft_export_inference"],
    }

    assert _errors(card) == []


def test_nested_commit_override_must_be_nonredundant() -> None:
    card = _card()
    _hardware_item(card, "pretrain")["bridge_commit"] = card["verification_environment"]["bridge_commit"]
    _assert_error(card, "/items/pretrain/H100/bridge_commit: omit a redundant override")


def test_nested_training_commands_receive_privacy_validation() -> None:
    card = _card()
    _hardware_item(card, "pretrain")["command"] += " $(whoami)"
    _assert_error(card, "shell command substitution is forbidden")


def test_manual_forward_requires_one_percent_correlation() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    assert "0.999969" in expected
    manual["expected_result"] = expected.replace("0.999969", "0.989999")
    _assert_error(card, "cosine similarity must be at least 0.99")

    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    manual["expected_result"] = manual["expected_result"].replace("0.999969", "0.990000")
    assert _errors(card) == []


def test_manual_forward_gate_matches_comparison_helper() -> None:
    assert VALIDATOR.MANUAL_FORWARD_COSINE_THRESHOLD == 0.99
    assert _assigned_float(COMPARE_PATH, "SIMILARITY_THRESHOLD") == VALIDATOR.MANUAL_FORWARD_COSINE_THRESHOLD


def test_manual_forward_grandfathers_documented_historical_evidence() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    assert "--hf-revision" not in manual["command"]
    assert "historical" in manual["expected_result"]
    assert _errors(card) == []

    manual["expected_result"] = (
        manual["expected_result"].replace("historical", "earlier").replace("predates", "preceded")
    )
    _assert_error(card, "missing --hf-revision is allowed only for verified historical evidence")

    card = _card(CORRELATION_CARD_PATH)
    card["items"]["manual_forward_pass"]["expected_result"] = card["items"]["manual_forward_pass"][
        "expected_result"
    ].replace("recorded immutable HF revision", "model revision")
    _assert_error(card, "tied to the recorded immutable HF revision")

    card = _card(CORRELATION_CARD_PATH)
    card["items"]["manual_forward_pass"]["last_verified"] = "2026-07-20"
    _assert_error(card, "evidence dated before 2026-07-20")


def test_unverified_manual_forward_command_requires_matching_revision() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    manual["status"] = "unverified"
    manual["last_verified"] = None
    _assert_error(card, "missing --hf-revision is allowed only for verified historical evidence")

    revision = card["model"]["hf_revision"]
    manual["command"] += f" --hf-revision {revision}"
    card["verification_index"]["model_level"] = {
        "verified": [
            "hf_to_megatron_cpu",
            "hf_to_megatron_gpu",
            "megatron_to_hf_cpu",
            "megatron_to_hf_gpu",
            "inference",
        ],
        "unverified": ["manual_forward_pass"],
    }
    assert _errors(card) == []

    manual["command"] = manual["command"].replace(revision, "1" * 40)
    _assert_error(card, "--hf-revision must equal model.hf_revision")


def test_manual_forward_revision_must_match_card() -> None:
    card = _card()
    manual = card["items"]["manual_forward_pass"]
    revision = card["model"]["hf_revision"]
    assert f"--hf-revision {revision}" in manual["command"]
    manual["command"] = manual["command"].replace(revision, "1" * 40)
    _assert_error(card, "--hf-revision must equal model.hf_revision")


def test_manual_forward_revision_may_appear_only_once() -> None:
    card = _card()
    manual = card["items"]["manual_forward_pass"]
    revision = card["model"]["hf_revision"]
    manual["command"] += f" --hf-revision {revision}"
    _assert_error(card, "specify --hf-revision at most once with a value")


def test_manual_forward_absolute_differences_are_report_only() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    assert "0.187500" in expected
    assert "0.030649" in expected
    manual["expected_result"] = expected.replace("0.187500", "12.000000").replace("0.030649", "3.000000")
    assert _errors(card) == []


def test_manual_forward_requires_affirmative_next_token_match() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    manual["expected_result"] = manual["expected_result"].replace(
        "next-token predictions match",
        "next-token predictions do not match",
    )
    _assert_error(card, "record that the next token matches")


@pytest.mark.parametrize("failure", ["failed to match", "never match"])
def test_manual_forward_rejects_other_token_mismatch_wording(failure: str) -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    manual["expected_result"] = manual["expected_result"].replace("predictions match", f"predictions {failure}")
    _assert_error(card, "record that the next token matches")


def test_manual_forward_requires_numeric_absolute_differences() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    manual["expected_result"] = expected.replace("0.187500", "not reported").replace("0.030649", "not reported")
    _assert_error(card, "record the numeric max absolute logit difference")
    _assert_error(card, "record the numeric mean absolute logit difference")


def test_manual_forward_accepts_helper_native_absolute_differences() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    start = expected.index("The maximum and mean absolute logit differences")
    manual["expected_result"] = expected[:start] + "Logits diff - max: 0.187500, mean: 0.030649."
    assert _errors(card) == []


def test_manual_forward_accepts_natural_absolute_differences() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    start = expected.index("The maximum and mean absolute logit differences")
    manual["expected_result"] = (
        expected[:start]
        + "The maximum absolute logit difference was 0.187500 and the mean absolute logit difference was 0.030649."
    )
    assert _errors(card) == []


def test_manual_forward_rejects_negative_absolute_differences() -> None:
    card = _card(CORRELATION_CARD_PATH)
    manual = card["items"]["manual_forward_pass"]
    expected = manual["expected_result"]
    manual["expected_result"] = expected.replace("0.187500", "-0.187500").replace("0.030649", "-0.030649")
    _assert_error(card, "record the numeric max absolute logit difference")
    _assert_error(card, "record the numeric mean absolute logit difference")


def test_resume_must_load_the_reference_save_directory() -> None:
    card = _card()
    resume = _hardware_item(card, "checkpoint_resume")
    resume["command"] = resume["command"].replace(
        "pretrain-reference-checkpoints",
        "another-reference",
    )
    _assert_error(card, "--load_dir must equal the pretrain --save_dir")


def test_resume_must_keep_reference_launch_settings() -> None:
    card = _card()
    resume = _hardware_item(card, "checkpoint_resume")
    resume["command"] = resume["command"].replace(
        "nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config",
        "another_recipe",
    )
    _assert_error(card, "reference and resume launch settings must match")


def test_resume_cannot_disable_canonical_checkpoint_paths() -> None:
    card = _card()
    resume = _hardware_item(card, "checkpoint_resume")
    resume["command"] += " checkpoint.load=null checkpoint.save=null"
    errors = _errors(card)
    assert any("not checkpoint.load" in error for error in errors), errors
    assert any("not checkpoint.save" in error for error in errors), errors


def test_reference_must_save_resume_state() -> None:
    card = _card()
    pretrain = _hardware_item(card, "pretrain")
    pretrain["command"] += " checkpoint.save_rng=false"
    _assert_error(card, "checkpoint.save_rng must remain true")


def test_verified_resume_requires_matching_verified_commit() -> None:
    card = _card()
    pretrain = _hardware_item(card, "pretrain")
    pretrain["status"] = "unverified"
    _assert_error(card, "pretrain must be verified first")

    card = _card()
    resume = _hardware_item(card, "checkpoint_resume")
    resume["bridge_commit"] = "1" * 40
    _assert_error(card, "resume and pretrain must use the same Bridge commit")


def test_item_commit_override_must_be_verified_and_nonredundant() -> None:
    card = _card()
    manual = card["items"]["manual_forward_pass"]
    manual["status"] = "unverified"
    manual["bridge_commit"] = "1" * 40
    _assert_error(card, "item overrides are allowed only when verified")

    card = _card()
    manual = card["items"]["manual_forward_pass"]
    manual["bridge_commit"] = card["verification_environment"]["bridge_commit"]
    _assert_error(card, "omit a redundant override")


def test_resume_setting_sort_handles_flag_and_flag_value() -> None:
    settings = VALIDATOR._resume_reference_settings("./scripts/training/train.sh --deterministic --deterministic=true")
    assert settings is not None
    assert len(settings) == 2


def test_missing_environment_commit_does_not_cascade_redundant_errors() -> None:
    card = copy.deepcopy(_card())
    del card["verification_environment"]["bridge_commit"]
    errors = _errors(card)
    assert any("verification_environment/bridge_commit" in error for error in errors)
    assert not any("omit a redundant override" in error for error in errors)
