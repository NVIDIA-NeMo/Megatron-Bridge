import importlib.util
from pathlib import Path


_VALIDATOR_PATH = (
    Path(__file__).parents[3] / "skills" / "create-model-verification-card" / "scripts" / "validate_card.py"
)
_SPEC = importlib.util.spec_from_file_location("model_verification_card_validator", _VALIDATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VALIDATOR)


def _validate_precision(item_name: str, precision: str) -> list[str]:
    errors: list[str] = []
    item = {
        "status": "unverified",
        "precision": precision,
        "last_verified": None,
        "expected_result": "Pending exact-model verification.",
        "command": None,
    }
    if item_name in _VALIDATOR.TRAINING_ITEMS:
        item["metrics"] = {name: None for name in _VALIDATOR.METRIC_NAMES}
    if item_name in _VALIDATOR.FEATURE_ITEMS:
        item["enabled_features"] = {}
    _VALIDATOR._validate_item(
        item_name,
        item,
        errors,
        path=("items", item_name),
        model_revision=None,
    )
    return errors


def test_fp32_is_allowed_for_direct_model_items() -> None:
    assert _validate_precision("inference", "fp32") == []


def test_fp32_is_rejected_for_training_items() -> None:
    errors = _validate_precision("sft", "fp32")

    assert errors == ["/items/sft/precision: fp32 is supported only on direct items"]
