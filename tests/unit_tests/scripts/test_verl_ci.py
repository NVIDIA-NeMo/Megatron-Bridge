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

"""Check the Bridge-owned verl launcher without installing or running verl."""

import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/workflows/verl_ci/run_megatron_moe_expert_parallel.sh"


@pytest.mark.parametrize("fail_quantized", [False, True])
def test_expert_parallel_variants(tmp_path: Path, fail_quantized: bool) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("pip3", "python3", "ray"):
        stub = bin_dir / command
        stub.write_text("#!/bin/bash\nexit 0\n")
        stub.chmod(0o755)

    # Intercept only the trainer invocation. Run the launcher itself with /bin/bash.
    trainer = bin_dir / "bash"
    trainer.write_text(
        "#!/bin/bash\n"
        'test "$1" = tests/special_e2e/run_ppo_trainer_megatron.sh || exit 99\n'
        "mkdir -p checkpoints\n"
        'printf "%s\\n" "$COMMON_PP,$COMMON_TP,$COMMON_ETP,$COMMON_EP,$INFER_TP,'
        "${ROLLOUT_QUANTIZATION:-null},${LORA_RANK:-0},${CRITIC_LORA_RANK:-0},"
        '${LORA_MERGE:-False}" >> "$CALL_LOG"\n'
        'if [[ "$FAIL_QUANTIZED" = 1 && "${ROLLOUT_QUANTIZATION:-null}" != null ]]; then\n'
        "    exit 23\n"
        "fi\n"
    )
    trainer.chmod(0o755)
    call_log = tmp_path / "calls"
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)],
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:{os.defpath}",
            "HOME": str(tmp_path),
            "CALL_LOG": str(call_log),
            "FAIL_QUANTIZED": str(int(fail_quantized)),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == (23 if fail_quantized else 0), result.stderr
    expected = [
        "2,4,1,4,8,null,0,0,False",
        "2,4,1,4,2,fp8_block,0,0,False",
        "2,4,1,2,8,null,8,8,True",
    ]
    if fail_quantized:
        expected = expected[:2]
    assert call_log.read_text().splitlines() == expected
    assert not (tmp_path / "checkpoints").exists()
