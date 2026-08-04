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
"""Consistency check for examples/models/qwen/qwen2_audio/sft.sh.

Qwen2-Audio 7B weights and optimizer state fill roughly two thirds of an 80 GB
H100, so the launcher has to run at the smallest micro batch size in-batch
packing permits.

Deliberately stdlib-only (re over the sources) so it runs without torch/GPU.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SFT_SH = REPO_ROOT / "examples" / "models" / "qwen" / "qwen2_audio" / "sft.sh"
TRAINING_CONFIG = REPO_ROOT / "src" / "megatron" / "bridge" / "training" / "config.py"

MIN_IN_BATCH_PACKING_MICRO_BATCH_SIZE = 2


def test_in_batch_packing_still_rejects_micro_batch_size_one():
    """ConfigContainer rejects micro_batch_size=1 while in-batch packing is on."""
    source = TRAINING_CONFIG.read_text(encoding="utf-8")
    assert "enable_in_batch_packing and self.train.micro_batch_size == 1" in source, (
        "the in-batch-packing micro_batch_size guard moved; re-derive the minimum micro batch size"
    )


def test_sft_sh_uses_smallest_packing_micro_batch_size():
    """sft.sh enables in-batch packing and runs at its smallest legal micro batch size."""
    source = SFT_SH.read_text(encoding="utf-8")
    assert "dataset.enable_in_batch_packing=true" in source, "sft.sh no longer enables in-batch packing"
    assert "train.micro_batch_size=$MICRO_BATCH_SIZE" in source, (
        "sft.sh no longer forwards MICRO_BATCH_SIZE to train.micro_batch_size"
    )

    match = re.search(r"^MICRO_BATCH_SIZE=(\d+)$", source, flags=re.MULTILINE)
    assert match, f"MICRO_BATCH_SIZE assignment not found in {SFT_SH}"
    assert int(match.group(1)) == MIN_IN_BATCH_PACKING_MICRO_BATCH_SIZE, (
        f"sft.sh sets MICRO_BATCH_SIZE={match.group(1)}; anything above "
        f"{MIN_IN_BATCH_PACKING_MICRO_BATCH_SIZE} overruns 80 GB per rank"
    )


if __name__ == "__main__":
    test_in_batch_packing_still_rejects_micro_batch_size_one()
    test_sft_sh_uses_smallest_packing_micro_batch_size()
    print("OK")
