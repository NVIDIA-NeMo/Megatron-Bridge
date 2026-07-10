# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import runpy
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "examples" / "models" / "qwen" / "qwen3_vl" / "prepare_mantis_energon.py"


def _load_module() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_sample_split_is_stable_and_validates_fraction():
    module = _load_module()

    assert module["_sample_split"]("sample", 0.0) == "train"
    assert module["_sample_split"]("sample", 0.5) == module["_sample_split"]("sample", 0.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        module["convert"]("missing", "output", 1, validation_fraction=1.0)


def test_prepare_energon_dataset_indexes_nonempty_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    calls = []
    monkeypatch.setattr(module["subprocess"], "run", lambda command, check: calls.append((command, check)))

    module["prepare_energon_dataset"](
        str(tmp_path),
        counts={"train": 10, "val": 0},
        num_workers=4,
    )

    command, check = calls[0]
    assert check is True
    assert command[:3] == ["energon", "prepare", str(tmp_path)]
    assert "train:train-shard-.*" in command
    assert "val:val-shard-.*" not in command
    dataset_yaml = (tmp_path / ".nv-meta" / "dataset.yaml").read_text(encoding="utf-8")
    assert "megatron.bridge.data.energon.task_encoder_utils" in dataset_yaml
    assert "imgs: jpgs" in dataset_yaml
