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

"""Forward and backward passes of the assembled toy Inkling2 model on two GPUs, with plain, packed, and image batches."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tests.functional_tests.test_groups.models.inkling.inkling2_toy_checkpoint import write_inkling2_toy_checkpoint


_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCRIPT = Path(__file__).with_name("inkling2_forward_backward.py")
# Data parallelism, tensor and sequence parallelism, and two pipeline stages whose boundary falls
# inside the first AttnRes block (the toy model groups two decoder blocks per AttnRes block).
_LAYOUTS = {
    "tp1": ["--tp", "1"],
    "tp2": ["--tp", "2"],
    "pp2": ["--pp", "2", "--pattern", "*E|*E*E*E"],
}
_BATCHES = ("plain", "packed", "image")


@pytest.fixture(scope="module")
def inkling2_toy_model_path(tmp_path_factory) -> str:
    """Write the toy checkpoint once for every layout."""
    return str(write_inkling2_toy_checkpoint(tmp_path_factory.mktemp("inkling2_toy") / "inkling2_toy"))


@pytest.fixture(scope="module")
def run_layout(inkling2_toy_model_path: str, tmp_path_factory):
    """Run each layout once and return its losses."""
    results: dict[str, dict] = {}

    def run(layout: str) -> dict:
        if layout not in results:
            output = tmp_path_factory.mktemp(layout) / "losses.json"
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
                str(_SCRIPT),
                "--model",
                inkling2_toy_model_path,
                *_LAYOUTS[layout],
                "--output",
                str(output),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO_ROOT)
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            assert result.returncode == 0, f"Inkling2 forward and backward failed at {layout}"
            results[layout] = json.loads(output.read_text())
        return results[layout]

    return run


@pytest.mark.run_only_on("GPU")
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_inkling2_trains_plain_packed_and_image_batches(run_layout, layout: str) -> None:
    """Every batch trains at the layout, and zero-initialized LoRA adapters leave the loss unchanged."""
    result = run_layout(layout)

    assert sorted(result["losses"]) == sorted(_BATCHES)
    assert all(math.isfinite(loss) and loss > 0 for loss in result["losses"].values())
    assert math.isclose(result["lora_loss"], result["losses"]["plain"], rel_tol=1e-3)


@pytest.mark.run_only_on("GPU")
def test_inkling2_layouts_agree(run_layout) -> None:
    """Tensor and pipeline parallelism, including a stage boundary inside an AttnRes block, keep the losses."""
    reference = run_layout("tp1")["losses"]
    for layout in ("tp2", "pp2"):
        losses = run_layout(layout)["losses"]
        for batch in _BATCHES:
            # TP2 differs from TP1 by about 1e-4 relative in BF16; the pipeline split matches it bit for bit.
            assert math.isclose(losses[batch], reference[batch], rel_tol=5e-3), (
                f"{layout} {batch} loss {losses[batch]} differs from tp1 {reference[batch]}"
            )
