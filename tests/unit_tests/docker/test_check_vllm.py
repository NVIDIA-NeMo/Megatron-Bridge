"""Check installed-package validation without loading vLLM or CUDA."""

import importlib.util
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import pytest


SPEC = importlib.util.spec_from_file_location(
    "check_vllm", Path(__file__).resolve().parents[3] / "docker/common/check_vllm.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


@pytest.mark.unit
@pytest.mark.parametrize("installed", ["0.29.1.dev0+g98dff2a81.d20261010.cu134", "0.30.0rc1"])
def test_rejects_unpatched_vllm(installed):
    with patch.object(CHECK, "version", return_value=installed), pytest.raises(ValueError, match="rebuild with"):
        CHECK.main()


@pytest.mark.unit
def test_accepts_patched_vllm():
    with patch.object(CHECK, "version", return_value="0.30.0"):
        CHECK.main()


@pytest.mark.unit
def test_allows_images_without_vllm():
    with patch.object(CHECK, "version", side_effect=PackageNotFoundError("vllm")):
        CHECK.main()
