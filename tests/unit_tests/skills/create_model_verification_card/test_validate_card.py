# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused tests for model-verification-card validation."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR_PATH = REPO_ROOT / "skills" / "create-model-verification-card" / "scripts" / "validate_card.py"
pytestmark = pytest.mark.unit

# Audited from recipe-owned GBS, the resolved card sequence or pack length, and
# the public command topology: (sequence_or_pack_length, global_batch_size, GPUs).
TRAINING_THROUGHPUT_INPUTS = {
    ("gemma-4-26b-a4b-it", "sft", "H100"): (4096, 32, 8),
    ("gemma-4-26b-a4b-it", "peft", "H100"): (4096, 32, 4),
    ("glm5-2", "pretrain", "H100"): (2048, 1024, 352),
    ("glm5-2", "pretrain", "GB200"): (4096, 1024, 192),
    ("glm5-2", "sft", "H100"): (2048, 32, 416),
    ("glm5-2", "sft", "GB200"): (8192, 8, 192),
    ("glm5-2", "sft_long_context", "H100"): (200000, 13, 608),
    ("glm5-2", "sft_long_context", "GB200"): (131072, 56, 192),
    ("glm5-2", "peft", "H100"): (2048, 32, 208),
    ("glm5-2", "peft", "GB200"): (2048, 32, 192),
    ("glm5-2", "checkpoint_resume", "H100"): (2048, 1024, 352),
    ("glm5-2", "checkpoint_resume", "GB200"): (4096, 1024, 192),
    ("gpt-oss-20b", "pretrain", "H100"): (4096, 512, 16),
    ("gpt-oss-20b", "sft", "H100"): (2048, 128, 8),
    ("gpt-oss-20b", "sft_long_context", "H100"): (32768, 32, 8),
    ("gpt-oss-20b", "peft", "H100"): (2048, 128, 1),
    ("gpt-oss-20b", "checkpoint_resume", "H100"): (4096, 512, 16),
    ("moonlight-16b-a3b", "pretrain", "H100"): (4096, 1024, 16),
    ("moonlight-16b-a3b", "sft", "H100"): (8192, 8, 8),
    ("moonlight-16b-a3b", "sft_long_context", "H100"): (8192, 128, 8),
    ("moonlight-16b-a3b", "peft", "H100"): (2048, 32, 4),
    ("moonlight-16b-a3b", "checkpoint_resume", "H100"): (4096, 1024, 16),
    ("nemotron-3-nano-4b", "pretrain", "H100"): (4096, 1024, 8),
    ("nemotron-3-nano-4b", "sft", "H100"): (2048, 32, 8),
    ("nemotron-3-nano-4b", "sft_long_context", "H100"): (32768, 8, 8),
    ("nemotron-3-nano-4b", "peft", "H100"): (2048, 32, 8),
    ("nemotron-3-nano-4b", "checkpoint_resume", "H100"): (4096, 1024, 8),
    ("nemotron-3-nano-omni-30b-a3b-reasoning", "sft", "H100"): (4096, 64, 16),
    ("nemotron-3-nano-omni-30b-a3b-reasoning", "peft", "H100"): (4096, 64, 8),
    ("nemotron-3-super-120b-a12b", "pretrain", "H100"): (4096, 1280, 64),
    ("nemotron-3-super-120b-a12b", "sft_long_context", "H100"): (32768, 2, 16),
    ("nemotron-3-super-120b-a12b", "peft", "GB200"): (8192, 16, 16),
    ("nemotron-3-super-120b-a12b", "pretrain_performance", "H100"): (4096, 1280, 64),
    ("nemotron-3-super-120b-a12b", "pretrain_performance", "GB200"): (4096, 512, 64),
    ("nemotron-3.5-lightning", "pretrain", "H100"): (8192, 512, 16),
    ("nemotron-3.5-lightning", "pretrain", "GB200"): (8192, 512, 8),
    ("nemotron-3.5-lightning", "pretrain_fsdp", "GB200", "bf16"): (8192, 512, 8),
    ("nemotron-3.5-lightning", "pretrain_fsdp", "GB200", "fp8_mx"): (8192, 384, 8),
    ("nemotron-3.5-lightning", "sft", "H100"): (4096, 128, 16),
    ("nemotron-3.5-lightning", "sft", "GB200"): (4096, 128, 8),
    ("nemotron-3.5-lightning", "sft_long_context", "H100"): (32768, 128, 16),
    ("nemotron-3.5-lightning", "sft_long_context", "GB200"): (32768, 128, 8),
    ("nemotron-3.5-lightning", "peft", "H100"): (4096, 128, 8),
    ("nemotron-3.5-lightning", "peft", "GB200"): (4096, 128, 8),
    ("nemotron-3.5-lightning", "checkpoint_resume", "H100"): (8192, 512, 16),
    ("nemotron-3.5-lightning", "checkpoint_resume", "GB200"): (8192, 512, 8),
    ("nemotron-3.5-lightning", "pretrain_performance", "H100"): (8192, 512, 16),
    ("nemotron-3.5-lightning", "pretrain_performance", "GB200"): (8192, 512, 8),
    ("qwen3-30b-a3b", "pretrain", "H100"): (4096, 1024, 16),
    ("qwen3-30b-a3b", "pretrain", "GB200"): (4096, 512, 8),
    ("qwen3-30b-a3b", "sft", "H100"): (2048, 32, 16),
    ("qwen3-30b-a3b", "sft_long_context", "H100"): (32768, 32, 16),
    ("qwen3-30b-a3b", "peft", "H100"): (2048, 32, 4),
    ("qwen3-30b-a3b", "checkpoint_resume", "H100"): (4096, 1024, 16),
    ("qwen3-30b-a3b", "checkpoint_resume", "GB200"): (4096, 512, 8),
    ("qwen3-30b-a3b", "pretrain_performance", "H100"): (4096, 1024, 16),
    ("qwen3-30b-a3b", "pretrain_performance", "GB200"): (4096, 512, 8),
    ("qwen3-8b", "pretrain", "H100"): (4096, 1024, 16),
    ("qwen3-8b", "sft", "H100"): (2048, 32, 4),
    ("qwen3-8b", "sft_long_context", "H100"): (32768, 8, 8),
    ("qwen3-8b", "peft", "H100"): (2048, 32, 1),
    ("qwen3-8b", "checkpoint_resume", "H100"): (4096, 1024, 16),
    ("qwen3.6-35b-a3b", "pretrain", "H100"): (4096, 512, 8),
    ("qwen3.6-35b-a3b", "sft", "H100"): (4096, 32, 16),
    ("qwen3.6-35b-a3b", "sft", "GB200"): (4096, 32, 8),
    ("qwen3.6-35b-a3b", "sft_long_context", "H100"): (8192, 512, 32),
    ("qwen3.6-35b-a3b", "peft", "H100"): (4096, 32, 16),
    ("qwen3.6-35b-a3b", "peft", "GB200"): (4096, 32, 8),
    ("qwen3.6-35b-a3b", "checkpoint_resume", "H100"): (4096, 512, 8),
    ("qwen3.6-35b-a3b", "pretrain_performance", "H100"): (4096, 512, 16),
    ("qwen3.6-35b-a3b", "pretrain_performance", "GB200"): (4096, 480, 8),
}


def _load_validator():
    spec = importlib.util.spec_from_file_location("model_card_validator_under_test", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_training_tps_matches_audited_token_slot_inputs():
    validator = _load_validator()
    verified_leaves = {}

    for card_path in sorted((REPO_ROOT / "examples" / "model_verification_cards").glob("*/card.yaml")):
        card = yaml.safe_load(card_path.read_text())
        slug = card_path.parent.name
        for item_name in validator.TRAINING_ITEMS:
            for hardware, leaf in card["items"].get(item_name, {}).items():
                if not isinstance(leaf, dict):
                    continue
                if "metrics" in leaf:
                    if leaf.get("status") == "verified":
                        verified_leaves[(slug, item_name, hardware)] = leaf
                    else:
                        assert leaf["metrics"]["last_10_steps_tokens_per_second_per_gpu_avg"] is None
                for variant_name, variant in leaf.get("variants", {}).items():
                    if variant.get("status") == "verified":
                        verified_leaves[(slug, item_name, hardware, variant_name)] = variant
                    else:
                        assert variant["metrics"]["last_10_steps_tokens_per_second_per_gpu_avg"] is None

    assert verified_leaves.keys() == TRAINING_THROUGHPUT_INPUTS.keys()
    for leaf_key, (sequence_or_pack_length, global_batch_size, total_gpus) in TRAINING_THROUGHPUT_INPUTS.items():
        metrics = verified_leaves[leaf_key]["metrics"]
        token_slots_per_step = sequence_or_pack_length * global_batch_size
        expected_tps_per_gpu = token_slots_per_step / (metrics["last_10_steps_step_time_ms_avg"] / 1000) / total_gpus

        assert metrics["last_10_steps_tokens_per_second_per_gpu_avg"] == pytest.approx(
            expected_tps_per_gpu, abs=0.0005
        )


def test_shipped_greedy_inference_cards_validate():
    cards = [
        REPO_ROOT / "examples" / "model_verification_cards" / model / "card.yaml" for model in ("glm5-2", "kimi-k3")
    ]

    result = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), *(str(card) for card in cards)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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

    assert errors == [
        "/items/inference/command: inference must use ./scripts/inference/infer.sh or a local uv run helper"
    ]


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


@pytest.mark.parametrize(
    ("resume_initialization", "expected_errors"),
    [
        ("", []),
        (
            "--pretrained_checkpoint work/imported",
            [
                "/items/checkpoint_resume/H100/command: direct resume must omit the pretrained checkpoint "
                "and load only the reference checkpoint"
            ],
        ),
        (
            "checkpoint.pretrained_checkpoint=work/imported",
            [
                "/items/checkpoint_resume/H100/command: direct resume must omit the pretrained checkpoint "
                "and load only the reference checkpoint"
            ],
        ),
    ],
)
def test_resume_reference_only_warm_start(resume_initialization, expected_errors):
    validator = _load_validator()
    errors = []
    common_command = (
        "./scripts/training/train.sh --nodes 1 --gpus-per-node 8 "
        "--recipe vlm_pretrain --mode pretrain --dataset energon --max_steps 100"
    )

    validator._validate_resume_against_pretrain(
        {
            "status": "verified",
            "bridge_commit": "commit",
            "command": (
                f"{common_command} {resume_initialization} "
                "--load_dir work/reference --save_dir work/resume "
                "checkpoint.ckpt_step=50 logger.save_config_filepath=work/resume.yaml"
            ),
        },
        {
            "status": "verified",
            "bridge_commit": "commit",
            "command": (
                f"{common_command} "
                "--pretrained_checkpoint work/imported --save_dir work/reference checkpoint.load=null "
                "logger.save_config_filepath=work/reference.yaml"
            ),
        },
        resume_path=("items", "checkpoint_resume", "H100"),
        pretrain_path=("items", "pretrain", "H100"),
        default_bridge_commit=None,
        errors=errors,
    )

    assert errors == expected_errors
