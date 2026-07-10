# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Command-construction smokes for every VLM/Energon example launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class LauncherCase:
    path: str
    expected: tuple[str, ...]
    args: tuple[str, ...] = field(default_factory=tuple)


LAUNCHERS = (
    LauncherCase(
        "examples/megatron_mimo/qwen35_vl/slurm_sft.sh",
        ("finetune_qwen35_vl.py", "--dataset-name", "cord_v2"),
    ),
    LauncherCase("examples/models/gemma/gemma3_vl/peft.sh", ("gemma3_vl_4b_peft_config", "vlm_step")),
    LauncherCase("examples/models/gemma/gemma3_vl/sft.sh", ("gemma3_vl_4b_sft_config", "vlm_step")),
    LauncherCase("examples/models/gemma/gemma4_vl/peft.sh", ("gemma4_vl_26b_peft_config", "vlm_step")),
    LauncherCase("examples/models/gemma/gemma4_vl/sft.sh", ("gemma4_vl_26b_sft_config", "vlm_step")),
    LauncherCase("examples/models/gemma/gemma4_vl/slurm_peft.sh", ("gemma4_vl_26b_peft_config", "vlm_step")),
    LauncherCase("examples/models/gemma/gemma4_vl/slurm_sft.sh", ("gemma4_vl_26b_sft_config", "vlm_step")),
    LauncherCase("examples/models/glm/glm_45v/slurm_peft.sh", ("glm_45v_peft_config", "vlm_step")),
    LauncherCase("examples/models/glm/glm_45v/slurm_sft.sh", ("glm_45v_sft_config", "vlm_step")),
    LauncherCase("examples/models/mistral/ministral3/peft_unpacked.sh", ("ministral3_3b_peft_config", "vlm_step")),
    LauncherCase("examples/models/mistral/ministral3/sft_unpacked.sh", ("ministral3_3b_sft_config", "vlm_step")),
    LauncherCase(
        "examples/models/nemotron/nemotron_3_omni/slurm_peft_cord_v2.sh",
        ("nemotron_omni_cord_v2_peft_config", "nemotron_omni_step"),
    ),
    LauncherCase(
        "examples/models/nemotron/nemotron_3_omni/slurm_sft_cord_v2.sh",
        ("nemotron_omni_cord_v2_sft_config", "nemotron_omni_step"),
    ),
    LauncherCase(
        "examples/models/nemotron/nemotron_3_omni/slurm_peft_valor32k_avqa.sh",
        ("nemotron_omni_valor32k_peft_config", "nemotron_omni_step", "dataset.path="),
    ),
    LauncherCase(
        "examples/models/nemotron/nemotron_3_omni/slurm_sft_valor32k_avqa.sh",
        ("nemotron_omni_valor32k_sft_config", "nemotron_omni_step", "dataset.path="),
    ),
    LauncherCase("examples/models/qwen/qwen2_audio/sft.sh", ("qwen2_audio_7b_finetune_config", "audio_lm_step")),
    LauncherCase(
        "examples/models/qwen/qwen35_vl/slurm_peft.sh",
        ("qwen35_vl_800m_peft_config", "qwen3_vl_step"),
        ("0.8B",),
    ),
    LauncherCase(
        "examples/models/qwen/qwen35_vl/slurm_sft.sh",
        ("qwen35_vl_800m_sft_config", "qwen3_vl_step"),
        ("0.8B",),
    ),
    LauncherCase(
        "examples/models/qwen/qwen35_vl/slurm_sft_fsdp.sh",
        ("qwen35_vl_35b_a3b_fsdp_sft_config", "qwen3_vl_step"),
    ),
    LauncherCase(
        "examples/models/qwen/qwen3_omni/local_train_thinker_full.sh",
        ("qwen3_omni_30b_a3b_sft_hf_json_config", "qwen3_omni_step", "--nproc_per_node=8"),
    ),
    LauncherCase(
        "examples/models/qwen/qwen3_vl/peft.sh",
        ("qwen3_vl_8b_peft_config", "qwen3_vl_30b_a3b_peft_config", "qwen3_vl_step"),
    ),
    LauncherCase("examples/models/qwen/qwen3_vl/sft.sh", ("qwen3_vl_8b_sft_config", "qwen3_vl_step")),
    LauncherCase(
        "examples/models/qwen/qwen3_vl/peft_unpacked.sh",
        ("qwen3_vl_8b_peft_config", "qwen3_vl_30b_a3b_peft_config", "qwen3_vl_step"),
    ),
    LauncherCase("examples/models/qwen/qwen3_vl/sft_unpacked.sh", ("qwen3_vl_8b_sft_config", "qwen3_vl_step")),
    LauncherCase(
        "examples/models/qwen/qwen3_vl/peft_energon.sh",
        ("qwen3_vl_8b_peft_energon_config", "--dataset", "vlm-energon", "qwen3_vl_step"),
    ),
    LauncherCase(
        "examples/models/stepfun/step37/slurm_pretrain.sh",
        ("step37_flickr8k_sft_smoke_config", "step37_flickr8k_step"),
    ),
)


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize("case", LAUNCHERS, ids=lambda case: case.path)
def test_vlm_dataset_launcher_builds_expected_command(case: LauncherCase, tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "commands.txt"
    _write_fake_command(
        fake_bin / "uv",
        "{ printf 'uv'; printf ' <%s>' \"$@\"; printf '\\n'; } >> \"${COMMAND_CAPTURE}\"\n",
    )
    _write_fake_command(
        fake_bin / "srun",
        "{ printf 'srun'; printf ' <%s>' \"$@\"; printf '\\n'; } >> \"${COMMAND_CAPTURE}\"\n",
    )
    _write_fake_command(
        fake_bin / "scontrol",
        "printf 'node001\\nnode002\\nnode003\\nnode004\\nnode005\\nnode006\\nnode007\\nnode008\\n'\n",
    )

    workspace = tmp_path / "workspace"
    model_path = tmp_path / "hf-model"
    energon_path = tmp_path / "energon"
    train_jsonl = tmp_path / "train.jsonl"
    model_path.mkdir()
    energon_path.mkdir()
    train_jsonl.write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "COMMAND_CAPTURE": str(capture),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "MB_CONTAINER_IMAGE": "/tmp/mbridge.sqsh",
            "WORKSPACE": str(workspace),
            "ENERGON_PATH": str(energon_path),
            "HF_MODEL_PATH": str(model_path),
            "TRAIN_JSONL": str(train_jsonl),
            "VALID_JSONL": "",
            "TEST_JSONL": "",
            "PYTHON_BIN": sys.executable,
            "WANDB_MODE": "disabled",
            "DRY_RUN": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "SLURM_JOB_ID": "4774",
            "SLURM_JOB_NUM_NODES": "8",
            "SLURM_GPUS_PER_NODE": "8",
            "SLURM_NTASKS": "64",
            "SLURM_JOB_NODELIST": "nodes",
            "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0",
            "SLURM_NODEID": "0",
        }
    )
    env.pop("CONTAINER_IMAGE", None)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / case.path), *case.args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, f"{case.path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    evidence = result.stdout + result.stderr
    if capture.exists():
        evidence += capture.read_text(encoding="utf-8")
    for expected in case.expected:
        assert expected in evidence, f"{expected!r} missing from {case.path} command:\n{evidence}"


@pytest.mark.parametrize(
    "path",
    (
        "examples/megatron_mimo/qwen35_vl/finetune_qwen35_vl.py",
        "examples/models/nemotron/nemotron_vl/finetune_nemotron_nano_v2_vl.py",
        "examples/models/nemotron/nemotron_3_omni/cord_v2_inference.py",
        "examples/models/qwen/qwen3_vl/prepare_mantis_energon.py",
    ),
)
def test_python_vlm_dataset_entrypoint_help(path: str):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / path), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, f"{path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "usage:" in result.stdout.lower()
