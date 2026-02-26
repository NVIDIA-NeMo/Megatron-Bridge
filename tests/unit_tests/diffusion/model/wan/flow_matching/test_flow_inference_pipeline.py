# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

import pytest
import torch

from megatron.bridge.diffusion.models.wan.flow_matching.flow_inference_pipeline import FlowInferencePipeline


def test_select_checkpoint_dir_latest(tmp_path):
    base = tmp_path / "ckpts"
    os.makedirs(base / "iter_0000100")
    os.makedirs(base / "iter_0000200")

    # Minimal inference config object
    class _Cfg:
        num_train_timesteps = 1000
        param_dtype = torch.float32
        text_len = 512
        t5_dtype = torch.float32
        vae_stride = (1, 1, 1)
        patch_size = (1, 1, 1)

    # Instantiate object without running heavy init by patching __init__ to a no-op
    pip = object.__new__(FlowInferencePipeline)

    pip.inference_cfg = _Cfg()

    latest = FlowInferencePipeline._select_checkpoint_dir(pip, str(base), checkpoint_step=None)
    assert latest.endswith("iter_0000200")

    specific = FlowInferencePipeline._select_checkpoint_dir(pip, str(base), checkpoint_step=100)
    assert specific.endswith("iter_0000100")

    with pytest.raises(FileNotFoundError):
        FlowInferencePipeline._select_checkpoint_dir(pip, str(base), checkpoint_step=999)


def test_forward_pp_step_no_pp(monkeypatch):
    # Build a minimal instance skipping heavy init
    pip = object.__new__(FlowInferencePipeline)

    class _Model:
        class _Cfg:
            hidden_size = 16
            qkv_format = "sbhd"

        config = _Cfg()

        def __call__(self, x, grid_sizes, t, **kwargs):
            return x  # echo input

        def set_input_tensor(self, x):
            pass

    pip.model = _Model()

    # Patch parallel state to no-PP path
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)

    S, B, H = 8, 1, pip.model.config.hidden_size
    latent_model_input = torch.randn(S, B, H, dtype=torch.float32)
    grid_sizes = [(2, 2, 2)]
    timestep = torch.tensor([10.0], dtype=torch.float32)
    arg_c = {}

    out = FlowInferencePipeline.forward_pp_step(
        pip,
        latent_model_input=latent_model_input,
        grid_sizes=grid_sizes,
        max_video_seq_len=S,
        timestep=timestep,
        arg_c=arg_c,
    )
    assert out.shape == latent_model_input.shape
