import copy
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "skills/create-model-verification-card/scripts/validate_card.py"
CARD_PATH = REPO_ROOT / "model_cards/qwen3-8b/card.yaml"


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_card_validator", VALIDATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _card() -> dict[str, Any]:
    card = yaml.safe_load(CARD_PATH.read_text(encoding="utf-8"))
    assert isinstance(card, dict)
    return card


def _errors(card: dict[str, Any]) -> list[str]:
    raw = yaml.safe_dump(card, sort_keys=False)
    return VALIDATOR._validate_card(card, raw, ())


def _assert_error(card: dict[str, Any], fragment: str) -> None:
    errors = _errors(card)
    assert any(fragment in error for error in errors), errors


def test_repository_model_card_is_valid() -> None:
    assert _errors(_card()) == []


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
        "qwen3_8b_pretrain_4gpu_h100_bf16_config",
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
    settings = VALIDATOR._resume_reference_settings(
        "./scripts/training/train.sh --deterministic --deterministic=true"
    )
    assert settings is not None
    assert len(settings) == 2


def test_missing_environment_commit_does_not_cascade_redundant_errors() -> None:
    card = copy.deepcopy(_card())
    del card["verification_environment"]["bridge_commit"]
    errors = _errors(card)
    assert any("verification_environment/bridge_commit" in error for error in errors)
    assert not any("omit a redundant override" in error for error in errors)
