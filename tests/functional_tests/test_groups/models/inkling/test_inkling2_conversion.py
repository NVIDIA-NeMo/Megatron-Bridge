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

"""Hugging Face to Megatron to Hugging Face round trips of a toy Inkling2 checkpoint, byte-exact and plain BF16."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from megatron.bridge.models.inkling.inkling2_quantization import (
    companion_keys,
    dequantize_mxfp8,
    dequantize_nvfp4,
    expert_format,
)
from tests.functional_tests.test_groups.models.inkling.inkling2_toy_checkpoint import write_inkling2_toy_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[5]
_AUDIT_SCRIPT = Path(__file__).with_name("inkling2_roundtrip_audit.py")


@pytest.fixture(scope="module")
def inkling2_toy_model_path(tmp_path_factory) -> str:
    """Write the toy checkpoint once for every layout."""
    return str(write_inkling2_toy_checkpoint(tmp_path_factory.mktemp("inkling2_toy") / "inkling2_toy"))


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize(
    ("tp", "pp", "ep"),
    [(2, 1, 1), (1, 2, 1), (1, 1, 2)],
    ids=["tp2", "pp2", "ep2"],
)
def test_inkling2_roundtrip_is_byte_exact(
    inkling2_toy_model_path: str, tmp_path: Path, tp: int, pp: int, ep: int
) -> None:
    """Every source tensor, including the NVFP4 and MXFP8 experts and their scales, is exported unchanged."""
    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "-m",
        "coverage",
        "run",
        f"--data-file={_REPO_ROOT / '.coverage'}",
        f"--source={_REPO_ROOT}",
        "--parallel-mode",
        str(_AUDIT_SCRIPT),
        "--hf-model",
        inkling2_toy_model_path,
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--ep",
        str(ep),
        "--report",
        str(report_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.returncode == 0, f"Inkling2 round trip failed at TP{tp}/PP{pp}/EP{ep}"

    report = json.loads(report_path.read_text())
    assert report["mismatches"] == []
    assert report["missing"] == []
    assert report["extra"] == []
    assert report["exported_matching"] == report["source_tensors"] == 160


def _run_conversion(*args: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "-m",
        "coverage",
        "run",
        f"--data-file={_REPO_ROOT / '.coverage'}",
        f"--source={_REPO_ROOT}",
        "--parallel-mode",
        "scripts/conversion/run_conversion.py",
        *args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.returncode == 0, f"run_conversion.py {args[0]} failed"


def _read_safetensors(directory: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                tensors[name] = handle.get_tensor(name)
    return tensors


@pytest.fixture(scope="module")
def inkling2_toy_megatron_path(inkling2_toy_model_path: str, tmp_path_factory) -> Path:
    """Import the toy checkpoint once at TP1/PP2 for the export tests."""
    megatron_path = tmp_path_factory.mktemp("inkling2_toy_megatron")
    _run_conversion(
        "import",
        "--device",
        "gpu",
        "--hf-model",
        inkling2_toy_model_path,
        "--megatron-path",
        str(megatron_path),
        "--tp",
        "1",
        "--pp",
        "2",
        "--ep",
        "1",
        "--etp",
        "1",
    )
    return megatron_path


@pytest.mark.run_only_on("GPU")
def test_inkling2_megatron_checkpoint_roundtrip(
    inkling2_toy_model_path: str, inkling2_toy_megatron_path: Path, tmp_path: Path
) -> None:
    """A Megatron checkpoint saved at TP1/PP2 and exported at TP2/EP2 reproduces every source tensor.

    The layers have different parameters, so the distributed checkpoint stores each
    layer under its own keys; loading it at another layout reshards those keys.
    """
    megatron_path = inkling2_toy_megatron_path
    hf_path = tmp_path / "hf"
    _run_conversion(
        "export",
        "--device",
        "gpu",
        "--hf-model",
        inkling2_toy_model_path,
        "--megatron-path",
        str(megatron_path),
        "--hf-path",
        str(hf_path),
        "--tp",
        "2",
        "--pp",
        "1",
        "--ep",
        "2",
        "--etp",
        "1",
    )

    source = _read_safetensors(Path(inkling2_toy_model_path))
    exported = _read_safetensors(hf_path)
    assert sorted(exported) == sorted(source)
    for name, tensor in source.items():
        assert exported[name].dtype == tensor.dtype, name
        assert torch.equal(exported[name].view(torch.uint8), tensor.view(torch.uint8)), name
    assert (hf_path / "hf_quant_config.json").read_text() == (
        Path(inkling2_toy_model_path) / "hf_quant_config.json"
    ).read_text()
    # As in the release, config.json has no transformers_version, so Transformers applies the release's quantization.
    assert "transformers_version" not in json.loads((hf_path / "config.json").read_text())


def _dequantized_bf16(name: str, source: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a quantized source expert tensor as the BF16 weights Bridge imports."""
    fmt = expert_format(name, source)
    weight, scale = source[name], source[f"{name}_scale"]
    experts = []
    for expert in range(weight.shape[0]):
        if fmt == "nvfp4":
            experts.append(
                dequantize_nvfp4(
                    weight[expert],
                    scale[expert],
                    source[f"{name}_amax"][expert],
                    source[f"{name}_outer_scale_denominator"],
                )
            )
        else:
            experts.append(dequantize_mxfp8(weight[expert], scale[expert]))
    return torch.stack(experts).to(torch.bfloat16)


@pytest.mark.run_only_on("GPU")
def test_inkling2_bf16_export(inkling2_toy_model_path: str, inkling2_toy_megatron_path: Path, tmp_path: Path) -> None:
    """A plain BF16 export writes the experts as Bridge holds them and stays a valid Inkling2 checkpoint.

    The NVFP4 and MXFP8 experts come out in BF16 without their scale tensors, the FP32
    tensors stay FP32, every other tensor is unchanged, and hf_quant_config.json lists
    the experts as BF16. Importing that export and exporting it again reproduces it.
    """
    hf_path = tmp_path / "hf-bf16"
    _run_conversion(
        "export",
        "--device",
        "gpu",
        "--hf-model",
        inkling2_toy_model_path,
        "--megatron-path",
        str(inkling2_toy_megatron_path),
        "--hf-path",
        str(hf_path),
        "--export-weight-dtype",
        "bfloat16",
        "--tp",
        "2",
        "--pp",
        "1",
        "--ep",
        "2",
        "--etp",
        "1",
    )

    source = _read_safetensors(Path(inkling2_toy_model_path))
    exported = _read_safetensors(hf_path)
    experts = {name for name in source if expert_format(name, source) is not None}
    scales = {key for name in experts for key in companion_keys(name, expert_format(name, source))}
    assert experts
    assert sorted(exported) == sorted(set(source) - scales)
    for name, tensor in source.items():
        if name in scales:
            continue
        if name in experts:
            assert exported[name].dtype == torch.bfloat16, name
            assert torch.equal(exported[name], _dequantized_bf16(name, source)), name
        else:
            assert exported[name].dtype == tensor.dtype, name
            assert torch.equal(exported[name].view(torch.uint8), tensor.view(torch.uint8)), name
    assert any(tensor.dtype == torch.float32 for tensor in exported.values())

    precision = json.loads((hf_path / "hf_quant_config.json").read_text())
    groups = precision["model"]["parameter_groups"]
    assert {group["compute_dtype"] for group in groups} <= {"float32", "bfloat16"}
    listed = {name: group["output_dtype"] for group in groups for name in group["tensors"]}
    assert {name: str(exported[name].dtype).removeprefix("torch.") for name in listed} == listed
    # The recorded version tells Transformers that the BF16 experts are not quantized.
    assert "transformers_version" in json.loads((hf_path / "config.json").read_text())

    report_path = tmp_path / "report.json"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--nnodes=1",
        "-m",
        "coverage",
        "run",
        f"--data-file={_REPO_ROOT / '.coverage'}",
        f"--source={_REPO_ROOT}",
        "--parallel-mode",
        str(_AUDIT_SCRIPT),
        "--hf-model",
        str(hf_path),
        "--tp",
        "2",
        "--report",
        str(report_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    assert result.returncode == 0, "BF16 Inkling2 round trip failed"
    report = json.loads(report_path.read_text())
    assert report["mismatches"] == [] and report["missing"] == [] and report["extra"] == []
    assert report["exported_matching"] == report["source_tensors"] == len(exported)
