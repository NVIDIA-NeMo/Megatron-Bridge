"""Focused tests for model-verification-card validation."""

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = REPO_ROOT / "skills" / "create-model-verification-card" / "scripts" / "validate_card.py"
pytestmark = pytest.mark.unit


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
                "./scripts/inference/infer.sh --nodes 1 --gpus-per-node 8 --task model-comparison "
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
                "./scripts/inference/infer.sh --nodes 1 --gpus-per-node 8 --task vlm-generation "
                "--prompt 'Describe the image' --max_new_tokens 32"
            ),
            "expected_result": (
                "One greedy run produced an exact 32-token output. "
                'The exact completion was "The image contains a sufficiently long deterministic description."'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert errors == []


def test_base_inference_rejects_hf_export_launcher():
    validator = _load_validator()
    errors = []

    validator._validate_inference(
        {
            "command": (
                "./scripts/inference/infer.sh --task hf-inference --prompt 'Describe the image' --max-new-tokens 32"
            ),
            "expected_result": (
                "One greedy run produced an exact 32-token output. "
                'The exact completion was "The image contains a sufficiently long deterministic description."'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert errors == ["/items/inference/command: inference must use uv run"]


def test_inference_accepts_natural_eos_before_maximum():
    validator = _load_validator()
    errors = []

    validator._validate_inference(
        {
            "command": (
                "./scripts/inference/infer.sh --nodes 1 --gpus-per-node 8 --task vlm-generation "
                "--prompt 'Describe the image' --max_new_tokens 2"
            ),
            "expected_result": (
                "One greedy run stopped at EOS after exactly 1 generated token under the 2-token maximum. "
                'The literal completion was "image".'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert errors == []


@pytest.mark.parametrize("nonblocking_flag", ["--detach", "--dry-run", "--submission-dry-run"])
def test_inference_launcher_rejects_nonblocking_modes(nonblocking_flag):
    validator = _load_validator()
    errors = []

    validator._validate_inference(
        {
            "command": (
                "./scripts/inference/infer.sh --nodes 1 --gpus-per-node 8 --task vlm-generation "
                f"--prompt 'Describe the image' --max_new_tokens 2 {nonblocking_flag}"
            ),
            "expected_result": (
                'One greedy run produced an exact 2-token output. The exact completion was "The image".'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert errors == ["/items/inference/command: verified inference must wait for completion"]


@pytest.mark.parametrize(
    "resources",
    [
        "--gpus-per-node 8",
        "--nodes 1",
        "--nodes 0 --gpus-per-node 8",
        "--nodes 1 --gpus-per-node 0",
    ],
)
def test_inference_launcher_requires_positive_resources(resources):
    validator = _load_validator()
    errors = []

    validator._validate_inference(
        {
            "command": (
                f"./scripts/inference/infer.sh {resources} --task vlm-generation "
                "--prompt 'Describe the image' --max_new_tokens 2"
            ),
            "expected_result": (
                'One greedy run produced an exact 2-token output. The exact completion was "The image".'
            ),
        },
        item_name="inference",
        status="verified",
        errors=errors,
    )

    assert len(errors) == 1
    assert "requires exactly one positive integer" in errors[0]


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


def test_sft_export_inference_accepts_hf_inference_launcher():
    validator = _load_validator()
    errors = []

    validator._validate_sft_export_inference(
        {
            "status": "verified",
            "depends_on": "sft",
            "commands": [
                "./scripts/conversion/convert.sh export --executor slurm --device gpu "
                "--nodes 1 --gpus-per-node 8 --megatron-path work/sft/iter_0000100 --hf-path work/hf",
                "./scripts/inference/infer.sh --task hf-inference --nodes 1 --gpus-per-node 1 "
                "--hf-model work/hf --prompt 'Describe the image' --max-new-tokens 2",
            ],
            "expected_result": (
                "Transformers reloaded the export and one greedy run produced "
                'the exact 2-token completion "The image".'
            ),
        },
        {
            "status": "verified",
            "command": "./scripts/training/train.sh --save_dir work/sft --max_steps 100",
        },
        item_path=("items", "sft_export_inference", "H100"),
        sft_path=("items", "sft", "H100"),
        errors=errors,
    )

    assert errors == []
