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

"""Numerical coverage for public TE ops with externally owned LoRA weights."""

from unittest.mock import patch

import pytest
import torch
from megatron.core.model_parallel_config import ModelParallelConfig

from megatron.bridge.peft import utils as peft_utils
from megatron.bridge.peft.utils import GroupedExpertLinearAdapter


pytestmark = [pytest.mark.unit, pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA")]


def _recipe(name):
    from transformer_engine.common import recipe

    if name == "mxfp8":
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("MXFP8 requires Blackwell")
        return recipe.MXFP8BlockScaling()
    if name == "current":
        return recipe.Float8CurrentScaling()
    return recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID, amax_history_len=16)


def _adapter(zero_init):
    adapter = GroupedExpertLinearAdapter(
        in_features=32,
        out_features=64,
        dim=32,
        alpha=48,
        num_local_experts=3,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        activation="swish",
        input_is_parallel=True,
        model_parallel_config=ModelParallelConfig(params_dtype=torch.bfloat16),
        params_device=torch.device("cuda"),
        params_dtype=torch.bfloat16,
    )
    with torch.no_grad():
        adapter.linear_in.weight.normal_(std=0.1)
        if not zero_init:
            adapter.linear_out.weight.normal_(std=0.1)
        for expert, scale in enumerate((0.5, 1.0, 2.0)):
            adapter.linear_in.weight[expert].mul_(scale)
            adapter.linear_out.weight[expert].mul_(scale)
    return adapter


def _reference(x, weight_in, weight_out, splits):
    outputs = []
    for expert, tokens in enumerate(x.split(splits)):
        hidden = torch.nn.functional.silu(torch.nn.functional.linear(tokens, weight_in[expert]))
        outputs.append(torch.nn.functional.linear(hidden * 1.5, weight_out[expert]))
    return torch.cat(outputs)


def _assert_fp8_close(actual, expected):
    assert torch.isfinite(actual).all()
    # FP8 rounding accumulates across two projections and the activation. A
    # normalized error also catches wrong scaling or gradients going to a new
    # parameter without letting elementwise tolerances hide small gradients.
    error = torch.linalg.vector_norm(actual.float() - expected.float())
    scale = torch.linalg.vector_norm(expected.float())
    # A cold delayed-scaling expert starts with scale=1 and can exceed 12%
    # gradient error across two projections. The separate TE-owned-parameter
    # reference below checks the FP8 calculation itself with tight tolerances.
    assert error <= 0.20 * scale + 1e-6, (error.item(), scale.item())


class _TEReference(torch.nn.Module):
    """Same adapter math with ordinary TE-owned, per-expert leaf parameters."""

    def __init__(self, adapter, te):
        super().__init__()
        self.linear_in = te.ops.GroupedLinear(3, 32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
        self.linear_out = te.ops.GroupedLinear(3, 32, 64, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for expert in range(3):
                getattr(self.linear_in, f"weight{expert}").copy_(adapter.linear_in.weight[expert])
                getattr(self.linear_out, f"weight{expert}").copy_(adapter.linear_out.weight[expert])

    def forward(self, x, splits, alignment):
        padded_splits = [(n + alignment - 1) // alignment * alignment for n in splits]
        chunks = [
            torch.nn.functional.pad(chunk, (0, 0, 0, padded - n))
            for chunk, n, padded in zip(x.split(splits), splits, padded_splits)
        ]
        sizes = torch.tensor(padded_splits, device=x.device, dtype=torch.int64)
        hidden = torch.nn.functional.silu(self.linear_in(torch.cat(chunks), sizes)) * 1.5
        output = self.linear_out(hidden, sizes)
        return torch.cat([chunk[:n] for chunk, n in zip(output.split(padded_splits), splits)])


@pytest.mark.parametrize("recipe_name", ["delayed", "current", "mxfp8"])
@pytest.mark.parametrize("input_requires_grad", [False, True])
@pytest.mark.parametrize("zero_init", [False, True])
def test_fp8_adapter_math_routing_and_optimizer(recipe_name, input_requires_grad, zero_init):
    """Compare outputs, input/A/B gradients and optimizer updates with FP32 math."""
    te = pytest.importorskip("transformer_engine.pytorch")
    if not peft_utils.HAVE_TE_PYTORCH_GROUPED_LINEAR_OP:
        pytest.skip("TE grouped linear ops unavailable")
    recipe = _recipe(recipe_name)
    torch.manual_seed(173)
    adapter = _adapter(zero_init)
    te_reference = _TEReference(adapter, te)
    alignment = 128 if recipe_name == "mxfp8" else 16
    parameter_ids = {name: id(p) for name, p in adapter.named_parameters()}
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.25)
    with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
        warmup = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
        adapter(warmup, [0, 16, 0])
        te_reference(warmup, [0, 16, 0], alignment)

    # Different active experts and an empty expert in each forward. Keep two
    # forward graphs alive together to catch stale parameter bindings in TE.
    routing_steps = [([5, 0, 19], [0, 11, 7]), ([13, 9, 0], [3, 0, 17])]
    with (
        patch.object(adapter, "_forward_per_expert", side_effect=AssertionError("Must exercise FP8 ops")),
        patch.object(adapter, "_can_use_grouped_mm", side_effect=AssertionError("Must exercise FP8 ops")),
    ):
        for routing in routing_steps:
            optimizer.zero_grad(set_to_none=True)
            te_reference.zero_grad(set_to_none=True)
            ref_in = adapter.linear_in.weight.detach().float().requires_grad_()
            ref_out = adapter.linear_out.weight.detach().float().requires_grad_()
            losses, te_losses, ref_losses, inputs = [], [], [], []
            for splits in routing:
                x = torch.randn(sum(splits), 32, device="cuda", dtype=torch.bfloat16)
                x.requires_grad_(input_requires_grad)
                ref_x = x.detach().float().requires_grad_(input_requires_grad)
                te_x = x.detach().clone().requires_grad_(input_requires_grad)
                with te.autocast(enabled=True, recipe=recipe):
                    output = adapter(x, splits)
                    te_output = te_reference(te_x, splits, alignment)
                torch.testing.assert_close(output, te_output, rtol=0, atol=0)
                expected = _reference(ref_x, ref_in, ref_out, splits)
                _assert_fp8_close(output, expected)
                dy = torch.randn_like(expected) / sum(splits)
                losses.append((output.float() * dy).sum())
                te_losses.append((te_output.float() * dy).sum())
                ref_losses.append((expected * dy).sum())
                inputs.append((x, te_x, ref_x))
            (sum(losses) + sum(te_losses)).backward()
            sum(ref_losses).backward()
            for x, te_x, ref_x in inputs:
                if input_requires_grad:
                    torch.testing.assert_close(x.grad, te_x.grad, rtol=0, atol=0)
                    _assert_fp8_close(x.grad, ref_x.grad)
                else:
                    assert x.grad is None
            for weight, te_layer, ref_weight in (
                (adapter.linear_in.weight, te_reference.linear_in, ref_in),
                (adapter.linear_out.weight, te_reference.linear_out, ref_out),
            ):
                assert weight.grad is not None
                for expert in range(3):
                    # Packed autograd accumulation may round once at a different
                    # point than accumulation into individual BF16 parameters.
                    torch.testing.assert_close(
                        weight.grad[expert], getattr(te_layer, f"weight{expert}").grad, rtol=1e-2, atol=1e-4
                    )
                    _assert_fp8_close(weight.grad[expert], ref_weight.grad[expert])
            before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
            optimizer.step()
            for name, p in adapter.named_parameters():
                torch.testing.assert_close(p, before[name].add(p.grad, alpha=-0.25), rtol=0, atol=0)
            # Compare identical weights next step even if BF16 gradient
            # accumulation rounded differently. Keep the reference's TE state.
            with torch.no_grad():
                for expert in range(3):
                    getattr(te_reference.linear_in, f"weight{expert}").copy_(adapter.linear_in.weight[expert])
                    getattr(te_reference.linear_out, f"weight{expert}").copy_(adapter.linear_out.weight[expert])

        assert {name: id(p) for name, p in adapter.named_parameters()} == parameter_ids
        assert set(adapter.state_dict()) == {"linear_in.weight", "linear_out.weight"}
        assert len(adapter._te_grouped_linear_helpers) == 2
        for helper in adapter._te_grouped_linear_helpers.values():
            assert all(p.grad is None for p in helper.parameters())

        adapter.eval()
        with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
            x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
            output = adapter(x, [0, 16, 0])
            expected = _reference(
                x.float(), adapter.linear_in.weight.float(), adapter.linear_out.weight.float(), [0, 16, 0]
            )
            _assert_fp8_close(output, expected)


@pytest.mark.parametrize("recipe_name", ["delayed", "current", "mxfp8"])
def test_fp8_adapter_zero_tokens_and_recompute(recipe_name):
    """Empty batches keep gradient edges; recomputation still uses packed weights."""
    te = pytest.importorskip("transformer_engine.pytorch")
    from torch.utils.checkpoint import checkpoint

    recipe = _recipe(recipe_name)
    torch.manual_seed(194)
    adapter = _adapter(False)
    x = torch.randn(23, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def forward(inp):
        with te.autocast(enabled=True, recipe=recipe):
            return adapter(inp, [7, 0, 16])

    output = checkpoint(forward, x, use_reentrant=True)
    ref_x = x.detach().float().requires_grad_()
    ref_in = adapter.linear_in.weight.detach().float().requires_grad_()
    ref_out = adapter.linear_out.weight.detach().float().requires_grad_()
    expected = _reference(ref_x, ref_in, ref_out, [7, 0, 16])
    _assert_fp8_close(output, expected)
    output.float().square().mean().backward()
    expected.square().mean().backward()
    _assert_fp8_close(x.grad, ref_x.grad)
    _assert_fp8_close(adapter.linear_in.weight.grad, ref_in.grad)
    _assert_fp8_close(adapter.linear_out.weight.grad, ref_out.grad)
    assert torch.count_nonzero(adapter.linear_in.weight.grad[1]) == 0
    assert torch.count_nonzero(adapter.linear_out.weight.grad[1]) == 0

    adapter.zero_grad(set_to_none=True)
    empty = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        result = adapter(empty, [0, 0, 0])
    assert result.shape == (0, 64)
    result.sum().backward()
    # The existing empty-batch path anchors adapter weights; it need not
    # materialize a gradient for an input containing no elements.
    for p in adapter.parameters():
        assert p.grad is not None
        assert torch.count_nonzero(p.grad) == 0


def test_fp8_adapter_device_dtype_changes_clear_helpers():
    te = pytest.importorskip("transformer_engine.pytorch")
    adapter = _adapter(False)
    recipe = _recipe("delayed")
    with torch.no_grad(), te.autocast(enabled=True, recipe=recipe):
        adapter(torch.randn(16, 32, device="cuda", dtype=torch.bfloat16), [0, 16, 0])
    assert len(adapter._te_grouped_linear_helpers) == 2
    adapter.cpu()
    assert not adapter._te_grouped_linear_helpers
    assert all(p.device.type == "cpu" for p in adapter.parameters())
    adapter.to(device="cuda", dtype=torch.float16)
    x = torch.randn(16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    with te.autocast(enabled=True, recipe=recipe):
        output = adapter(x, [16, 0, 0])
    ref_x = x.detach().float().requires_grad_()
    ref_in = adapter.linear_in.weight.detach().float().requires_grad_()
    ref_out = adapter.linear_out.weight.detach().float().requires_grad_()
    expected = _reference(ref_x, ref_in, ref_out, [16, 0, 0])
    output.float().square().mean().backward()
    expected.square().mean().backward()
    _assert_fp8_close(output, expected)
    _assert_fp8_close(x.grad, ref_x.grad)
    _assert_fp8_close(adapter.linear_in.weight.grad, ref_in.grad)
    _assert_fp8_close(adapter.linear_out.weight.grad, ref_out.grad)
