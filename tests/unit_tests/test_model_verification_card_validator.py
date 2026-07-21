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
CARD_PATH = REPO_ROOT / "model_cards/nemotron-3-nano-4b/card.yaml"
CORRELATION_CARD_PATH = REPO_ROOT / "model_cards/qwen3-8b/card.yaml"
CARD_PATHS = sorted((REPO_ROOT / "model_cards").glob("*/card.yaml"))


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
def test_repository_model_cards_are_valid(card_path: Path) -> None:
    assert _errors(_card(card_path)) == []


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
    resume = card["items"]["checkpoint_resume"]
    resume["command"] = resume["command"].replace(
        "pretrain-reference-checkpoints",
        "another-reference",
    )
    _assert_error(card, "--load_dir must equal the pretrain --save_dir")


def test_resume_must_keep_reference_launch_settings() -> None:
    card = _card()
    resume = card["items"]["checkpoint_resume"]
    resume["command"] = resume["command"].replace(
        "nemotron_3_nano_4b_pretrain_8gpu_h100_bf16_config",
        "another_recipe",
    )
    _assert_error(card, "reference and resume launch settings must match")


def test_resume_cannot_disable_canonical_checkpoint_paths() -> None:
    card = _card()
    resume = card["items"]["checkpoint_resume"]
    resume["command"] += " checkpoint.load=null checkpoint.save=null"
    errors = _errors(card)
    assert any("not checkpoint.load" in error for error in errors), errors
    assert any("not checkpoint.save" in error for error in errors), errors


def test_reference_must_save_resume_state() -> None:
    card = _card()
    pretrain = card["items"]["pretrain"]
    pretrain["command"] += " checkpoint.save_rng=false"
    _assert_error(card, "checkpoint.save_rng must remain true")


def test_verified_resume_requires_matching_verified_commit() -> None:
    card = _card()
    pretrain = card["items"]["pretrain"]
    pretrain["status"] = "unverified"
    _assert_error(card, "pretrain must be verified first")

    card = _card()
    resume = card["items"]["checkpoint_resume"]
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
