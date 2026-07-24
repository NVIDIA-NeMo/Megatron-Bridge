"""Focused tests for model-verification-card validation."""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = REPO_ROOT / "skills" / "create-model-verification-card" / "scripts" / "validate_card.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("model_card_validator_under_test", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_forward_accepts_inference_launcher():
    validator = _load_validator()
    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
    errors = []

    validator._validate_manual_forward_pass(
        {
            "command": (
                "./scripts/inference/infer.sh --task model-comparison "
                "--hf_model_path hf/model --megatron_model_path work/checkpoint "
                f"--hf-revision {revision} --prompt 'Describe the image'"
            ),
            "last_verified": "2026-07-23",
            "expected_result": (
                "The next-token predictions matched. Cosine similarity: 0.999156. "
                "Maximum and mean absolute logit differences were 0.484375 and 0.082185."
            ),
        },
        status="verified",
        model_revision=revision,
        errors=errors,
    )

    assert errors == []


def test_inference_accepts_vlm_generation_launcher():
    validator = _load_validator()
    errors = []

    validator._validate_inference(
        {
            "command": (
                "./scripts/inference/infer.sh --task vlm-generation --prompt 'Describe the image' --max_new_tokens 32"
            ),
            "expected_result": (
                "Two independent runs produced byte-identical 32-token output. "
                'The exact completion was "The image contains a sufficiently long deterministic description."'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert errors == []


def test_cpu_conversion_accepts_one_runtime_gpu():
    validator = _load_validator()
    errors = []

    validator._validate_conversion_launcher(
        "./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --gpus-per-node 1",
        operation="import",
        device="cpu",
        path=("items", "hf_to_megatron_cpu", "command"),
        errors=errors,
    )

    assert errors == []


def test_cpu_conversion_rejects_multiple_runtime_gpus():
    validator = _load_validator()
    errors = []

    validator._validate_conversion_launcher(
        "./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --gpus-per-node 2",
        operation="export",
        device="cpu",
        path=("items", "megatron_to_hf_cpu", "command"),
        errors=errors,
    )

    assert errors == ["/items/megatron_to_hf_cpu/command: CPU conversion may request at most one shared runtime GPU"]
