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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.models.common.vision_module.vision_module import VisionModule
from torch import nn

from megatron.bridge.diffusion.models.wan.wan_model import WanModel


pytestmark = [pytest.mark.unit]


def _tiny_wan(value: float) -> WanModel:
    """Build the smallest WAN module that still exercises its checkpoint methods."""
    model = WanModel.__new__(WanModel)
    VisionModule.__init__(model, SimpleNamespace())
    model.register_parameter("weight", nn.Parameter(torch.tensor([value], dtype=torch.float32)))
    model.tp_group = torch.distributed.group.WORLD
    return model


def _optimizer_state(model_state, value: float):
    """Build an optimizer shard whose storage key embeds the model key."""
    model_shard = next(iter(model_state.values()))
    optimizer_shard = make_sharded_optimizer_tensor(
        model_shard,
        torch.tensor([value], dtype=torch.float32),
        prefix="optimizer.state.exp_avg",
    )
    return {"state": {0: {"exp_avg": optimizer_shard}}}


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="Gloo is required for the distributed checkpoint round trip",
)
def test_wan_checkpoint_round_trip_strict(tmp_path):
    """Save and strictly reload the canonical WAN checkpoint namespace."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'distributed_init'}",
        rank=0,
        world_size=1,
    )

    try:
        metadata = {"dp_cp_group": torch.distributed.group.WORLD}
        source = _tiny_wan(7.0)
        source_model_state = source.sharded_state_dict(metadata=metadata)
        assert source_model_state.keys() == source.state_dict().keys()
        source_optimizer_state = _optimizer_state(source_model_state, 3.0)

        with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.cuda.synchronize"):
            save(
                {"model": source_model_state, "optimizer": source_optimizer_state},
                checkpoint_dir,
                async_sharded_save=False,
            )

            destination = _tiny_wan(-1.0)
            destination_model_state = destination.sharded_state_dict(metadata=metadata)
            load_scaffold = {
                "model": destination_model_state,
                "optimizer": _optimizer_state(destination_model_state, -3.0),
            }
            loaded_state = load(load_scaffold, checkpoint_dir)

        load_result = destination.load_state_dict(loaded_state["model"], strict=True)
        assert not load_result.missing_keys
        assert not load_result.unexpected_keys
        torch.testing.assert_close(destination.weight, source.weight)
        torch.testing.assert_close(
            loaded_state["optimizer"]["state"][0]["exp_avg"],
            source_optimizer_state["state"][0]["exp_avg"].data,
        )
    finally:
        torch.distributed.destroy_process_group()
