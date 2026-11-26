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

"""Exercise Bridge's DTensor checkpoint backend with interleaved model rows."""

import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.glu import permute_glu_rows
from megatron.core.transformer import TransformerConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from megatron.bridge.training.checkpointing import load_fsdp_dtensor_checkpoint, save_fsdp_dtensor_checkpoint
from megatron.bridge.training.config import CheckpointConfig


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires a two-GPU distributed launch")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
    parallel_state.initialize_model_parallel()
    yield
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


def test_fsdp_backend_converts_glu_rows_and_records_metadata(distributed_group):
    model = torch.nn.Module()
    model.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=1,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
    )
    model.mlp = torch.nn.Module()
    model.mlp.config = model.config
    model.mlp._checkpoint_glu_interleave_size = lambda: 2
    model.mlp.linear_fc1 = torch.nn.Linear(3, 32, bias=False, device="cuda")
    model.mlp.linear_fc1._glu_interleave_size = 2
    mesh = init_device_mesh("cuda", (2,))
    canonical = torch.arange(96, device="cuda").reshape(32, 3).float()
    runtime = permute_glu_rows(canonical, 2, interleave=True)
    key = "mlp.linear_fc1.weight"
    checkpoint_cfg = CheckpointConfig(ckpt_format="fsdp_dtensor", finetune=True)
    cfg = SimpleNamespace(model=model.config, checkpoint=checkpoint_cfg)
    with tempfile.TemporaryDirectory() as local_path:
        paths = [local_path]
        torch.distributed.broadcast_object_list(paths, src=0)
        path = paths[0]
        source = {"model": {key: distribute_tensor(runtime, mesh, [Shard(0)])}}
        save_fsdp_dtensor_checkpoint(path, source, cfg=cfg, model=model)
        reader = torch.distributed.checkpoint.FileSystemReader(path)
        assert "glu_checkpoint_metadata_json" in reader.read_metadata().state_dict_metadata
        target = {"model": {key: distribute_tensor(torch.zeros_like(runtime), mesh, [Shard(0)])}, "_model": [model]}
        restored, _, _, _ = load_fsdp_dtensor_checkpoint(
            path, checkpoint_cfg, False, target, 1, checkpoint_path_override=path, cfg=cfg
        )
        torch.testing.assert_close(restored["model"][key].full_tensor(), runtime)
        # Save preprocessing never mutates the live source parameter storage.
        torch.testing.assert_close(source["model"][key].full_tensor(), runtime)
        torch.distributed.barrier()
