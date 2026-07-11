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


EXPECTED_TARGET_IDS = {
    "deepseek-v2",
    "deepseek-v2-lite",
    "deepseek-v4-flash",
    "ernie45-21b-a3b",
    "falcon-h1-500m",
    "gemma-2b",
    "gemma2-2b",
    "gemma3-1b",
    "gemma4-26b-a4b",
    "gemma4-31b",
    "glm45-355b",
    "glm47-355b",
    "glm47-flash-31b",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "hy3-preview-base",
    "ling-flash-2",
    "ling-mini-2",
    "llama2-7b",
    "llama3-8b",
    "llama31-8b",
    "llama31-nemotron-nano-4b",
    "llama32-1b",
    "llama33-70b",
    "mimo-7b",
    "mimo-v2-flash",
    "minimax-m2",
    "minimax-m2-5",
    "minimax-m2-7",
    "mistral-7b",
    "moonlight-16b",
    "nemotron-h-4b",
    "nemotron-nano-9b-v2",
    "nemotron3-nano",
    "nemotron3-super",
    "olmoe-7b",
    "qwen2-7b",
    "qwen25-7b",
    "qwen3-30b-a3b",
    "qwen3-8b",
    "qwen3-next-80b-a3b",
    "qwen35-27b",
    "qwen35-35b-a3b",
    "sarvam-30b",
    "step35-flash",
}


def _load_module():
    script = Path(__file__).resolve().parents[4] / "scripts" / "training" / "submit_text_pretrain_validation.py"
    spec = importlib.util.spec_from_file_location("test_submit_text_pretrain_validation_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_required_args(module, tmp_path: Path, *extra_args: str):
    return module._build_parser().parse_args(
        [
            "--account",
            "account",
            "--partition",
            "interactive",
            "--container-image",
            "image.sqsh",
            "--dataset-path",
            str(tmp_path / "data" / "dclm_text_document"),
            "--dataset-cache",
            str(tmp_path / "cache"),
            "--output-root",
            str(tmp_path / "output"),
            *extra_args,
        ]
    )


def test_manifest_exactly_matches_pr4805_text_only_scope():
    module = _load_module()

    targets = module.load_manifest(module.DEFAULT_MANIFEST)

    assert len(targets) == 45
    assert {target.id for target in targets} == EXPECTED_TARGET_IDS
    assert all(target.revision.isalnum() and len(target.revision) == 40 for target in targets)
    assert all(target.recipe.endswith("_config") for target in targets)


def test_manifest_revision_segments_are_restored_to_immutable_sha():
    module = _load_module()

    targets = module.load_manifest(module.DEFAULT_MANIFEST)

    assert targets[0].revision == "18ca64a019b553be57bab50af3207fb2f3675edc"  # pragma: allowlist secret


def test_manifest_model_segments_are_restored():
    module = _load_module()

    targets = module.load_manifest(module.DEFAULT_MANIFEST)

    target = next(target for target in targets if target.id == "nemotron3-super")
    assert target.hf_model == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"  # pragma: allowlist secret


def test_all_target_parallelism_is_valid_for_two_by_eight_world():
    module = _load_module()
    targets = module.load_manifest(module.DEFAULT_MANIFEST)

    assert all(target.minimum_world_size <= 16 for target in targets)
    assert all(16 % target.minimum_world_size == 0 for target in targets)

    nemotron3_nano = next(target for target in targets if target.id == "nemotron3-nano")
    assert (nemotron3_nano.tensor_parallelism, nemotron3_nano.expert_parallelism) == (2, 8)

    gemma4_dense = next(target for target in targets if target.id == "gemma4-31b")
    assert (gemma4_dense.tensor_parallelism, gemma4_dense.pipeline_parallelism) == (8, 1)

    qwen3_next = next(target for target in targets if target.id == "qwen3-next-80b-a3b")
    assert "model.recompute_granularity=full" in qwen3_next.overrides
    assert "model.recompute_modules=null" in qwen3_next.overrides

    gpt_oss_120b = next(target for target in targets if target.id == "gpt-oss-120b")
    assert (gpt_oss_120b.pipeline_parallelism, gpt_oss_120b.expert_parallelism) == (2, 8)
    assert "optimizer.use_precision_aware_optimizer=true" in gpt_oss_120b.overrides
    assert "optimizer.main_params_dtype=float16" in gpt_oss_120b.overrides
    assert "optimizer.exp_avg_sq_dtype=bfloat16" in gpt_oss_120b.overrides

    nemotron3_super = next(target for target in targets if target.id == "nemotron3-super")
    assert (nemotron3_super.tensor_parallelism, nemotron3_super.pipeline_parallelism) == (4, 2)
    assert "model.pipeline_model_parallel_layout=null" in nemotron3_super.overrides
    assert "optimizer.main_params_dtype=float16" in nemotron3_super.overrides
    assert "optimizer.exp_avg_sq_dtype=bfloat16" in nemotron3_super.overrides

    mimo_v2_flash = next(target for target in targets if target.id == "mimo-v2-flash")
    assert "ddp.grad_reduce_in_fp32=false" in mimo_v2_flash.overrides
    assert "optimizer.main_params_dtype=float16" in mimo_v2_flash.overrides

    ling_flash = next(target for target in targets if target.id == "ling-flash-2")
    assert "model.moe_permute_fusion=false" in ling_flash.overrides
    assert "ddp.grad_reduce_in_fp32=false" in ling_flash.overrides
    assert "optimizer.main_params_dtype=float16" in ling_flash.overrides


def test_command_uses_fixed_real_data_and_wandb_contract(tmp_path):
    module = _load_module()
    target = module.load_manifest(module.DEFAULT_MANIFEST)[0]
    args = _parse_required_args(
        module,
        tmp_path,
        "--hf-home",
        str(tmp_path / "hf"),
        "--runtime-venv",
        str(tmp_path / "venv"),
        "--wandb-netrc",
        str(tmp_path / ".netrc"),
    )
    module._validate_args(args, [target])

    command = module.build_command(args, target)

    assert command[0].endswith("scripts/training/train.sh")
    assert command[command.index("--nodes") + 1] == "2"
    assert command[command.index("--gpus-per-node") + 1] == "8"
    assert command[command.index("--dataset") + 1] == "dclm"
    assert command[command.index("--max-steps") + 1] == "100"
    assert command[command.index("--global-batch-size") + 1] == "128"
    assert command[command.index("--log-interval") + 1] == "1"
    assert command[command.index("--cp") + 1] == "1"
    assert command[command.index("--vp") + 1] == "none"
    assert command[command.index("--etp") + 1] == "1"
    assert command[command.index("--wandb-project") + 1] == "megatron-bridge-text-pretrain-validation"
    assert "WANDB_RUN_GROUP=mb747-text-pretrain-dclm-20260710" in command  # pragma: allowlist secret
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in command
    assert f"{tmp_path / '.netrc'}:/root/.netrc" in command
    assert f"VIRTUAL_ENV={tmp_path / 'venv'}" in command
    assert any(value.startswith(f"PATH={tmp_path / 'venv' / 'bin'}:") for value in command)
    assert "checkpoint.load=null" in command


def test_sensitive_environment_values_must_be_inherited_by_name(tmp_path):
    module = _load_module()
    target = module.load_manifest(module.DEFAULT_MANIFEST)[0]
    args = _parse_required_args(module, tmp_path, "--env", "HF_TOKEN=literal")

    with pytest.raises(ValueError, match="by name"):
        module._validate_args(args, [target])


def test_non_contract_hardware_is_rejected(tmp_path):
    module = _load_module()
    target = module.load_manifest(module.DEFAULT_MANIFEST)[0]
    args = _parse_required_args(module, tmp_path, "--nodes", "1")

    with pytest.raises(ValueError, match="exactly 2 nodes x 8 GPUs"):
        module._validate_args(args, [target])
