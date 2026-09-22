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

"""Two-rank equivalence check for Nemotron Omni vision_dp_over_cp.

The sharded vision tower must return the same features on every CP rank as the
unsharded tower would, for every split shape: fewer images than ranks (MCore
injects placeholder images), an exact split, and a remainder that lands on the
last rank.

The test self-launches a two-rank job so it runs under the plain unit-test
sweep instead of needing a torchrun launcher script.
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


_CP_SIZE = 2
_REPO_ROOT = Path(__file__).resolve().parents[4]
# One image exercises placeholder padding, two an exact split, three the
# remainder that MCore hands to the last rank.
_IMAGE_COUNTS = (1, 2, 3)


@pytest.mark.gpu
def test_vision_dp_over_cp_matches_unsharded_encoder() -> None:
    if torch.cuda.device_count() < _CP_SIZE:
        pytest.skip(f"requires {_CP_SIZE} visible GPUs")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(_CP_SIZE),
            str(Path(__file__).resolve()),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _run_worker() -> None:
    import datetime

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    sys.path.insert(0, str(_REPO_ROOT))
    from tests.unit_tests.models.nemotron_omni.test_nemotron_omni_model import _TinyOmniProvider

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=10))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=_CP_SIZE,
    )
    model_parallel_cuda_manual_seed(123)

    provider = dataclasses.replace(
        _TinyOmniProvider(),
        context_parallel_size=_CP_SIZE,
        vision_dp_over_cp=True,
        mamba_num_heads=8,
        mamba_num_groups=_CP_SIZE,
    )
    provider.finalize()
    model = provider.provide().cuda().eval()

    # Each rank initializes its own weights, so replicate rank 0's. Real
    # training keeps the vision tower identical across CP ranks; without this
    # the comparison below would fail on weight divergence, not on sharding.
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.data, src=0)

    for num_images in _IMAGE_COUNTS:
        generator = torch.Generator(device="cpu").manual_seed(1234 + num_images)
        images = torch.randn(num_images, 3, 32, 32, generator=generator).cuda()
        imgs_sizes = torch.tensor([[32, 32]] * num_images, dtype=torch.int32, device="cuda")
        num_frames = torch.tensor([1] * num_images, dtype=torch.int32, device="cuda")

        with torch.no_grad():
            model.vision_dp_over_cp = True
            sharded = model._encode_images(images, imgs_sizes, None, num_frames)

            model.vision_dp_over_cp = False
            reference = model._encode_images(images, imgs_sizes, None, num_frames)
        model.vision_dp_over_cp = True

        assert sharded.shape == reference.shape, (
            f"num_images={num_images}: sharded {tuple(sharded.shape)} != reference {tuple(reference.shape)}"
        )
        max_delta = (sharded.float() - reference.float()).abs().max().item()
        assert max_delta < 1e-2, f"num_images={num_images}: max abs delta {max_delta}"

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    _run_worker()
