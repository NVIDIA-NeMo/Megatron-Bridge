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

"""Unit tests for PEFT-specific recompute helpers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.peft import recompute as recompute_mod
from megatron.bridge.peft.recompute import maybe_enable_recompute_inputs_grad


class DummyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None

    def forward(self, hidden_states, *args, **kwargs):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class DummyHybridStack(DummyTransformerBlock):
    pass


class DummyModel(torch.nn.Module):
    def __init__(self, block_cls=DummyTransformerBlock) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = block_cls()

        # Frozen base parameter (not trainable)
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False

        # Trainable adapter parameter whose name contains ".adapter."
        # Use a ModuleDict with key "adapter" so that the full parameter
        # name includes the expected substring (".adapter.") used by
        # maybe_enable_recompute_inputs_grad.
        self.adapter = torch.nn.ModuleDict({"adapter": DummyAdapter()})

    def modules(self):
        for module in super().modules():
            yield module


def _patch_recompute_blocks(monkeypatch):
    import megatron.core.models.hybrid.hybrid_block as hybrid_block
    import megatron.core.transformer.transformer_block as transformer_block

    monkeypatch.setattr(hybrid_block, "HybridStack", DummyHybridStack, raising=False)
    monkeypatch.setattr(
        transformer_block,
        "TransformerBlock",
        DummyTransformerBlock,
        raising=False,
    )


@pytest.mark.parametrize("block_cls", [DummyTransformerBlock, DummyHybridStack])
def test_maybe_enable_recompute_inputs_grad_patches_block(monkeypatch, block_cls):
    _patch_recompute_blocks(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyModel(block_cls)
    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) in patched_registry

    patched_forward = model.block.forward

    input_tensor = torch.zeros(2, 2)
    assert input_tensor.requires_grad is False

    model.block(input_tensor)
    assert model.block.last_input_requires_grad is True

    # Second invocation should be a no-op (no duplicate patch)
    maybe_enable_recompute_inputs_grad(model, patched_registry)
    assert model.block.forward is patched_forward
