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

"""Unit tests for the multi-adapter LoRA layer (:class:`MultiLoRALinear`).

Mirrors ``test_lora_layers.py``: covers per-slot rank/alpha bookkeeping and rank
masking on ``MultiLoRALinear``, the standalone model-level slot helpers
(routing, init/clear, expose/hide, load), and the bridge export seam that the
expose/hide lifecycle feeds.

The heavy ``ParallelLinearAdapter`` dependency is replaced with a CPU fake that
shares the same weight layout, so these tests run without a GPU or parallel
state. The :class:`MultiLoRA` PEFT object (config + transform) is covered in
``test_multi_lora.py``.
"""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge
from megatron.bridge.peft import multi_lora_layers as multi_lora_layers_module
from megatron.bridge.peft.multi_lora_layers import (
    MultiLoRALinear,
    _iter_multi_lora_modules,
    clear_adapter_slot,
    expose_adapter_slot,
    hide_adapters,
    init_adapter_slot,
    load_adapter,
    set_tokens_per_adapter_slot,
)
from megatron.bridge.peft.utils import AdapterAttributes


# ======================================================================
# Test doubles
# ======================================================================


class _FakeParallelLinearAdapter(nn.Module):
    """CPU stand-in for ``ParallelLinearAdapter`` with the same weight layout.

    For TP=1 the real adapter exposes ``linear_in.weight`` of shape
    ``(dim, in_features)`` and ``linear_out.weight`` of shape
    ``(out_features, dim)``; plain ``nn.Linear`` layers reproduce that exactly,
    which is all the rank-mask / slot bookkeeping logic touches.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str,
        *,
        alpha: float | None = None,
        input_is_parallel: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha if alpha is not None else dim
        self.base_linear_name = base_linear_name
        # Attributes the bridge export path reads off the exposed `.adapter`.
        self.input_is_parallel = input_is_parallel
        self.base_linear_is_parallel = True
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        nn.init.xavier_normal_(self.linear_in.weight)
        nn.init.zeros_(self.linear_out.weight)


def _fake_get_attrs(module: nn.Module, *args, **kwargs) -> AdapterAttributes:
    """Return adapter attributes for a plain ``nn.Linear`` ``to_wrap``."""
    return AdapterAttributes(
        input_is_parallel=getattr(module, "_test_input_is_parallel", False),
        in_features=module.in_features,
        out_features=module.out_features,
        disable_tensor_parallel_comm=False,
        disable_sequence_parallel_comm=True,
        base_linear_is_parallel=True,
    )


def _build_multi_lora_linear(
    in_features: int = 16,
    out_features: int = 32,
    n_adapters: int = 2,
    dim: int = 8,
    alpha: float = 16,
    full_name: str = "decoder.layers.0.self_attention.linear_proj",
) -> MultiLoRALinear:
    """Construct a ``MultiLoRALinear`` (requires the fake-adapter patches to be active)."""
    return MultiLoRALinear(
        to_wrap=nn.Linear(in_features, out_features),
        n_adapters=n_adapters,
        dim=dim,
        alpha=alpha,
        full_name=full_name,
    )


def adapter_deps_patch() -> ExitStack:
    """Patch the layer module's adapter construction dependencies for CPU use."""
    stack = ExitStack()
    stack.enter_context(patch.object(multi_lora_layers_module, "ParallelLinearAdapter", _FakeParallelLinearAdapter))
    stack.enter_context(patch.object(multi_lora_layers_module, "get_adapter_attributes_from_linear", _fake_get_attrs))
    return stack


# ======================================================================
# MultiLoRALinear: per-slot rank/alpha bookkeeping + rank masking
# ======================================================================


class TestMultiLoRALinearSlots:
    """Slot init/clear, rank masking, weight reset, and state-dict layout."""

    @pytest.fixture(autouse=True)
    def _patch_adapter_deps(self):
        with adapter_deps_patch():
            yield

    def test_slot_defaults_after_construction(self) -> None:
        layer = _build_multi_lora_linear(n_adapters=3, dim=8)

        assert layer.n_adapters == 3
        assert layer.max_rank == 8
        assert layer.tokens_per_adapter is None
        assert torch.equal(layer.alpha_values, torch.ones(3))
        assert torch.equal(layer.rank_values, torch.full((3,), 8.0))

    def test_init_adapter_slot_sets_rank_alpha_and_masks(self) -> None:
        layer = _build_multi_lora_linear(dim=8)
        with torch.no_grad():
            layer.adapters[0].linear_in.weight.fill_(1.0)
            layer.adapters[0].linear_out.weight.fill_(1.0)

        layer.init_adapter_slot(0, rank=4, alpha=16)

        assert layer.alpha_values[0] == 16
        assert layer.rank_values[0] == 4
        a = layer.adapters[0].linear_in.weight  # (dim, in)
        b = layer.adapters[0].linear_out.weight  # (out, dim)
        assert torch.all(a[4:] == 0)
        assert torch.all(a[:4] == 1)
        assert torch.all(b[:, 4:] == 0)
        assert torch.all(b[:, :4] == 1)

    def test_init_adapter_slot_full_rank_does_not_mask(self) -> None:
        layer = _build_multi_lora_linear(dim=8)
        with torch.no_grad():
            layer.adapters[1].linear_in.weight.fill_(1.0)
            layer.adapters[1].linear_out.weight.fill_(1.0)

        layer.init_adapter_slot(1, rank=8, alpha=8)

        assert layer.rank_values[1] == 8
        assert torch.all(layer.adapters[1].linear_in.weight == 1)
        assert torch.all(layer.adapters[1].linear_out.weight == 1)

    @pytest.mark.parametrize("bad_rank", [0, -1, 9])
    def test_init_adapter_slot_rejects_out_of_range_rank(self, bad_rank: int) -> None:
        layer = _build_multi_lora_linear(dim=8)
        with pytest.raises(AssertionError):
            layer.init_adapter_slot(0, rank=bad_rank, alpha=16)

    def test_clear_adapter_slot_resets_state_and_weights(self) -> None:
        layer = _build_multi_lora_linear(dim=8)
        layer.init_adapter_slot(0, rank=4, alpha=16)
        with torch.no_grad():
            layer.adapters[0].linear_out.weight.fill_(1.0)

        layer.clear_adapter_slot(0)

        assert layer.alpha_values[0] == 0
        assert layer.rank_values[0] == layer.max_rank
        # B is re-initialised to zero on clear.
        assert torch.all(layer.adapters[0].linear_out.weight == 0)

    def test_reset_adapter_zeroes_b_matrix(self) -> None:
        layer = _build_multi_lora_linear(dim=8)
        with torch.no_grad():
            layer.adapters[1].linear_out.weight.fill_(1.0)

        layer.reset_adapter(1)

        assert torch.all(layer.adapters[1].linear_out.weight == 0)

    def test_state_dict_contains_base_and_all_adapter_slots(self) -> None:
        layer = _build_multi_lora_linear(n_adapters=2, dim=8)

        keys = set(layer.state_dict().keys())

        assert {"weight", "bias"}.issubset(keys)
        assert "adapters.0.linear_in.weight" in keys
        assert "adapters.0.linear_out.weight" in keys
        assert "adapters.1.linear_in.weight" in keys
        assert "adapters.1.linear_out.weight" in keys


# ======================================================================
# Standalone model-level slot helpers
# ======================================================================


class _MultiLoRAContainer(nn.Module):
    """Container with several ``MultiLoRALinear`` modules plus an unrelated linear."""

    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.mods = nn.ModuleList([_build_multi_lora_linear() for _ in range(n_layers)])
        self.other = nn.Linear(4, 4)


class TestMultiLoRAModelHelpers:
    """Routing, init/clear, expose/hide and load helpers operating over a model."""

    @pytest.fixture(autouse=True)
    def _patch_adapter_deps(self):
        with adapter_deps_patch():
            yield

    def test_iter_multi_lora_modules_single_model(self) -> None:
        container = _MultiLoRAContainer(n_layers=3)

        found = list(_iter_multi_lora_modules(container))

        assert len(found) == 3
        assert {id(m) for m in found} == {id(m) for m in container.mods}

    def test_iter_multi_lora_modules_list_of_chunks(self) -> None:
        chunks = [_MultiLoRAContainer(n_layers=2), _MultiLoRAContainer(n_layers=1)]

        found = list(_iter_multi_lora_modules(chunks))

        assert len(found) == 3

    def test_set_tokens_per_adapter_slot(self) -> None:
        container = _MultiLoRAContainer(n_layers=2)
        tokens = torch.tensor([3, 5], dtype=torch.int32)

        set_tokens_per_adapter_slot(container, tokens)

        for module in container.mods:
            assert module.tokens_per_adapter is tokens

    def test_init_and_clear_adapter_slot_across_model(self) -> None:
        container = _MultiLoRAContainer(n_layers=2)

        init_adapter_slot(container, 1, rank=4, alpha=16)
        for module in container.mods:
            assert module.rank_values[1] == 4
            assert module.alpha_values[1] == 16

        clear_adapter_slot(container, 1)
        for module in container.mods:
            assert module.alpha_values[1] == 0
            assert module.rank_values[1] == module.max_rank

    def test_expose_adapter_slot_exposes_then_restores(self) -> None:
        container = _MultiLoRAContainer(n_layers=2)
        slot0 = [m.adapters[0] for m in container.mods]
        adapters_lists = [m.adapters for m in container.mods]

        with expose_adapter_slot(container, 0):
            for module, expected in zip(container.mods, slot0):
                assert "adapters" not in module._modules
                assert module.adapter is expected

        for module, expected_list, expected_slot in zip(container.mods, adapters_lists, slot0):
            assert "adapter" not in module._modules
            assert module.adapters is expected_list
            assert module.adapters[0] is expected_slot

    def test_hide_adapters_hides_then_restores(self) -> None:
        container = _MultiLoRAContainer(n_layers=2)
        adapters_lists = [m.adapters for m in container.mods]

        with hide_adapters(container):
            for module in container.mods:
                assert "adapters" not in module._modules

        for module, expected_list in zip(container.mods, adapters_lists):
            assert module.adapters is expected_list

    def test_load_adapter_copies_into_target_slot(self) -> None:
        container = _MultiLoRAContainer(n_layers=2)

        # Snapshot slot 0 and build a checkpoint from its (slot-independent) names.
        slot0_before = {}
        target_state = {}
        with expose_adapter_slot(container, 0):
            for name, param in container.named_parameters():
                if ".adapter." in name:
                    slot0_before[name] = param.detach().clone()
                    target_state[name] = torch.randn_like(param)

        # Saving from slot 0 and loading into slot 1 must work: the slot index is
        # stripped from the names while a slot is exposed.
        loaded = load_adapter(container, 1, target_state)
        assert loaded == len(target_state)

        with expose_adapter_slot(container, 1):
            slot1 = {name: p for name, p in container.named_parameters() if ".adapter." in name}
            for name, expected in target_state.items():
                assert torch.equal(slot1[name], expected)

        # Slot 0 must be untouched by the load into slot 1.
        with expose_adapter_slot(container, 0):
            for name, param in container.named_parameters():
                if ".adapter." in name:
                    assert torch.equal(param, slot0_before[name])


# ======================================================================
# Bridge export integration (CPU): lifecycle methods drive the real export seam
# ======================================================================


class _ExportSelfAttention(nn.Module):
    def __init__(self, wrapper: nn.Module) -> None:
        super().__init__()
        self.linear_proj = wrapper


class _ExportLayer(nn.Module):
    def __init__(self, wrapper: nn.Module) -> None:
        super().__init__()
        self.self_attention = _ExportSelfAttention(wrapper)


class _ExportModel(nn.Module):
    """Minimal ``decoder.layers.N.self_attention.linear_proj`` tree for export discovery."""

    def __init__(self, wrapper: nn.Module) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_ExportLayer(wrapper)])


class TestMultiLoRAExportIntegration:
    """Drive the real bridge export consumer through the expose/hide lifecycle.

    The HF export path (:class:`MegatronPeftBridge`) locates adapters via
    :meth:`MegatronPeftBridge._get_adapter_wrap_module`, which reads a single-LoRA
    ``.adapter`` attribute off each wrapped module. ``MultiLoRALinear`` keeps its
    slots under ``.adapters`` (plural), so they are invisible to export until
    :func:`expose_adapter_slot` re-exposes one slot as ``.adapter``. These tests
    assert that contract against the actual bridge method rather than just the
    module-swap mechanics.
    """

    _PREFIX = "decoder.layers.0.self_attention.linear_proj"

    @pytest.fixture(autouse=True)
    def _patch_adapter_deps(self):
        with adapter_deps_patch():
            yield

    def test_adapter_hidden_from_export_without_expose(self) -> None:
        wrapper = _build_multi_lora_linear(full_name=self._PREFIX)
        model = _ExportModel(wrapper)

        adapter, to_wrap = MegatronPeftBridge()._get_adapter_wrap_module(self._PREFIX, [model], vp_stage=0)

        # Export reaches the wrapped base linear but finds no adapter to convert.
        assert adapter is None
        assert to_wrap is wrapper.to_wrap

    def test_expose_makes_slot_visible_to_export(self) -> None:
        wrapper = _build_multi_lora_linear(full_name=self._PREFIX)
        model = _ExportModel(wrapper)
        bridge = MegatronPeftBridge()
        slot0, slot1 = wrapper.adapters[0], wrapper.adapters[1]

        with expose_adapter_slot(model, 0):
            adapter, to_wrap = bridge._get_adapter_wrap_module(self._PREFIX, [model], vp_stage=0)
            assert adapter is slot0
            assert to_wrap is wrapper.to_wrap
            # The exposed slot exposes the single-LoRA interface the task builder reads.
            assert adapter.dim == wrapper.max_rank
            for attr in ("linear_in", "linear_out", "alpha", "input_is_parallel", "base_linear_is_parallel"):
                assert hasattr(adapter, attr)

        # A different slot index exposes a different adapter object.
        with expose_adapter_slot(model, 1):
            adapter, _ = bridge._get_adapter_wrap_module(self._PREFIX, [model], vp_stage=0)
            assert adapter is slot1

    def test_export_view_restored_after_expose(self) -> None:
        wrapper = _build_multi_lora_linear(full_name=self._PREFIX)
        model = _ExportModel(wrapper)
        bridge = MegatronPeftBridge()

        with expose_adapter_slot(model, 0):
            pass

        # Once the context exits the slot is hidden again (multi-slot layout restored).
        adapter, to_wrap = bridge._get_adapter_wrap_module(self._PREFIX, [model], vp_stage=0)
        assert adapter is None
        assert to_wrap is wrapper.to_wrap
        assert "adapters" in wrapper._modules
