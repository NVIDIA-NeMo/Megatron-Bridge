"""Tests of the GLM stage runner around Core's HybridStack."""

import datetime
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import WrappedTensor

from megatron.bridge.models.glm_moe_dsa.glm5_hybrid import _FORWARD_STATE, _forward_glm_stack


pytestmark = pytest.mark.unit


def _fake_hybrid_stack_forward(self, hidden_states, attention_mask, **kwargs):
    """Mimic the parts of Core's HybridStack.forward the runner depends on."""
    assert _FORWARD_STATE.get() is not None, "stage forward must run inside GLM forward state"
    if not self.pre_process:
        # See HybridStack.set_input_tensor(): non-first stages ignore the argument.
        hidden_states = self.input_tensor
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()
    self.calls.append(hidden_states)
    return hidden_states * self.weight


def _stack(*, pre_process, recompute, device):
    return SimpleNamespace(
        pre_process=pre_process,
        input_tensor=None,
        training=True,
        config=SimpleNamespace(recompute_granularity=recompute, distribute_saved_activations=False),
        weight=torch.nn.Parameter(torch.full((4,), 3.0, device=device)),
        calls=[],
    )


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

    @pytest.fixture(autouse=True)
    def _patch_core_forward(self, monkeypatch):
        monkeypatch.setattr(HybridStack, "forward", _fake_hybrid_stack_forward)

    def test_non_first_stage_full_recompute_keeps_autograd_graph(self):
        stack = _stack(pre_process=False, recompute="full", device="cuda")
        stage_input = torch.arange(4.0, device="cuda", requires_grad=True)
        stack.input_tensor = stage_input  # what the pipeline schedule injects

        # The schedule passes ``None`` as the forward argument on this stage.
        out = _forward_glm_stack(stack, None, None)

        assert out.requires_grad, "checkpointed stage output lost its autograd graph"
        out.sum().backward()
        assert stage_input.grad is not None
        torch.testing.assert_close(stage_input.grad, torch.full((4,), 3.0, device="cuda"))
        torch.testing.assert_close(stack.weight.grad, stage_input.detach())
        # Checkpoint forward + one replay in backward, both consuming a stage input.
        assert len(stack.calls) == 2
        # The replay must consume the checkpoint's detached copy, not the live input.
        assert stack.calls[1] is not stage_input
        assert stack.input_tensor is stage_input, "input_tensor must be restored after the stage"
        assert _FORWARD_STATE.get() is None

    def test_first_stage_full_recompute_unwraps_and_backprops(self):
        stack = _stack(pre_process=True, recompute="full", device="cuda")
        stage_input = torch.arange(4.0, device="cuda", requires_grad=True)

        out = _forward_glm_stack(stack, WrappedTensor(stage_input), None)

        assert out.requires_grad
        out.sum().backward()
        torch.testing.assert_close(stage_input.grad, torch.full((4,), 3.0, device="cuda"))
        assert stack.input_tensor is None

    def test_non_first_stage_without_recompute_uses_input_tensor(self):
        stack = _stack(pre_process=False, recompute=None, device="cuda")
        stage_input = torch.arange(4.0, device="cuda", requires_grad=True)
        stack.input_tensor = stage_input

        out = _forward_glm_stack(stack, None, None)

        torch.testing.assert_close(out, stage_input * 3.0)
        assert stack.calls == [stage_input]
        assert stack.input_tensor is stage_input


def test_run_restores_input_tensor_when_stage_forward_raises(monkeypatch):
    def boom(self, hidden_states, attention_mask, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(HybridStack, "forward", boom)
    stack = _stack(pre_process=False, recompute=None, device="cpu")
    stack.training = False
    stage_input = torch.ones(4)
    stack.input_tensor = stage_input

    with pytest.raises(RuntimeError, match="boom"):
        _forward_glm_stack(stack, None, None)

    assert stack.input_tensor is stage_input
    assert _FORWARD_STATE.get() is None
