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

"""Tests of the GLM stage runner around Core's HybridStack."""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import WrappedTensor

from megatron.bridge.models.glm_moe_dsa.glm5_hybrid import _FORWARD_STATE, _forward_glm_stack


pytestmark = pytest.mark.unit


def _stack(*, pre_process, recompute):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        ffn_hidden_size=32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        recompute_granularity=recompute,
        recompute_method="uniform" if recompute == "full" else None,
        recompute_num_layers=1 if recompute == "full" else None,
    )
    return HybridStack(
        config=config,
        submodules=hybrid_stack_spec.submodules,
        layer_config_list=validate_segment_layers("-", config),
        pre_process=pre_process,
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()


class TestForwardGlmStackPipelineInput:
    """The runner must checkpoint the real stage input, not the ``None`` placeholder."""

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            pytest.skip("tensor_parallel.checkpoint needs the CUDA RNG tracker")
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29511")
            torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl", world_size=1, rank=0, timeout=datetime.timedelta(minutes=5))
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()

    @pytest.mark.parametrize(
        "pre_process,recompute",
        [(False, "full"), (True, "full"), (False, None)],
        ids=["pipeline-recompute", "first-stage-wrapped-recompute", "pipeline-no-recompute"],
    )
    def test_forward_and_gradients_match_native_stack(self, pre_process, recompute):
        stack = _stack(pre_process=pre_process, recompute=recompute)
        reference = _stack(pre_process=pre_process, recompute=None)
        reference.load_state_dict(stack.state_dict())
        stage_input = torch.randn(4, 2, 16, device="cuda", requires_grad=True)
        reference_input = stage_input.detach().clone().requires_grad_()
        if not pre_process:
            stack.set_input_tensor(stage_input)
            reference.set_input_tensor(reference_input)
        original_config = stack.config
        calls = []

        def record_layer_input(module, args, kwargs):
            state = _FORWARD_STATE.get()
            assert state is not None, "stage forward must run inside GLM forward state"
            calls.append((kwargs["hidden_states"], state))

        stack.layers[0].register_forward_pre_hook(record_layer_input, with_kwargs=True)
        out = _forward_glm_stack(stack, WrappedTensor(stage_input) if pre_process else None, None)
        expected = reference(reference_input if pre_process else None, attention_mask=None)
        torch.testing.assert_close(out, expected)
        assert out.requires_grad, "checkpointed stage output lost its autograd graph"
        assert _FORWARD_STATE.get() is None
        out.sum().backward()
        expected.sum().backward()
        assert stage_input.grad is not None
        torch.testing.assert_close(stage_input.grad, reference_input.grad)
        for parameter, reference_parameter in zip(stack.parameters(), reference.parameters(), strict=True):
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)

        assert len(calls) == (2 if recompute == "full" else 1)
        assert calls[0][0] is stage_input
        if recompute == "full":
            # Native HybridStack must read the checkpoint's detached replay input.
            assert calls[1][0] is not stage_input
            assert calls[1][1] is not calls[0][1]
        assert stack.input_tensor is (None if pre_process else stage_input)
        assert stack.config is original_config
        assert stack.config.recompute_granularity == recompute
        assert _FORWARD_STATE.get() is None

    def test_run_restores_input_tensor_when_stage_forward_raises(self):
        stack = _stack(pre_process=False, recompute=None)
        stage_input = torch.ones(4, 2, 16, device="cuda")
        stack.set_input_tensor(stage_input)

        def boom(module, args, kwargs):
            assert _FORWARD_STATE.get() is not None
            raise RuntimeError("boom")

        stack.layers[0].register_forward_pre_hook(boom, with_kwargs=True)
        with pytest.raises(RuntimeError, match="boom"):
            _forward_glm_stack(stack, None, None)

        assert stack.input_tensor is stage_input
        assert _FORWARD_STATE.get() is None
