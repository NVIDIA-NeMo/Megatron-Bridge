# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import runpy
from pathlib import Path

import pytest
import torch


_EXAMPLE_ROOT = Path(__file__).parents[3] / "examples" / "models" / "nemotron" / "nemotron_3_omni"


@pytest.mark.unit
@pytest.mark.parametrize(
    "script_name",
    [
        "cord_v2_inference.py",
        "hf_to_megatron_generate_nemotron_omni.py",
        "valor32k_avqa_inference.py",
    ],
)
def test_inference_forward_step_forwards_num_image_tiles_to_pipeline_stages(script_name):
    script_globals = runpy.run_path(_EXAMPLE_ROOT / script_name)
    iterator_cls = script_globals["SingleBatchIterator"]
    forward_step = script_globals["vlm_forward_step"]
    input_ids = torch.tensor([[10, 11, 12]])
    num_image_tiles = torch.tensor([256, 128], dtype=torch.int)
    seen = {}

    class _Model:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return torch.zeros(1, 3, 8)

    iterator = iterator_cls(
        input_ids,
        torch.arange(3).unsqueeze(0),
        torch.ones_like(input_ids, dtype=torch.bool),
        images=torch.zeros(1, 2, 3),
        num_image_tiles=num_image_tiles,
    )

    output, _ = forward_step(iterator, _Model())

    assert output.shape == (1, 3, 8)
    assert seen["num_image_tiles"] is num_image_tiles
