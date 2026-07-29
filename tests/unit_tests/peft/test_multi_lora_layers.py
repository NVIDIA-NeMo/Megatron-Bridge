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
expose/hide lifecycle feeds. The heavy ``ParallelLinearAdapter`` dependency is
replaced with a CPU fake that shares the same weight layout, so those tests run
without a GPU or parallel state. The :class:`MultiLoRA` PEFT object (config +
transform) is covered in ``test_multi_lora.py``.

Hardening coverage (disaggregated multi-LoRA):

Mock-level (no distributed):
  * B7: grouped-GEMM path rejects adapter dropout > 0 (it cannot apply it)
  * B2: expose_adapter_slot / hide_adapters restore the ModuleList even if the body raises
  * B8: MoE expert linears are skipped with a one-time warning, not silently
  * B9: load_adapter raises on a checkpoint/model mismatch in either direction
    (params missing from the checkpoint, or checkpoint tensors no param consumed)

Single-GPU integration (needs CUDA + model-parallel init):
  * grouped-GEMM forward smoke: output shape / finiteness / dtype (no fp32 promotion)
  * B4: reset_adapter re-inits through the model-parallel RNG tracker —
    deterministic given tracker state, mirroring the construction-time init methods
"""

import os
from contextlib import ExitStack, nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.models.conversion.peft_bridge import MegatronPeftBridge
from megatron.bridge.peft import multi_lora as multi_lora_mod
from megatron.bridge.peft import multi_lora_layers as multi_lora_layers_module
from megatron.bridge.peft.multi_lora import MultiLoRA
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
        **extra_kwargs: object,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha if alpha is not None else dim
        self.base_linear_name = base_linear_name
        self.extra_kwargs = extra_kwargs
        # Attributes the bridge export path reads off the exposed `.adapter`.
        self.input_is_parallel = input_is_parallel
        self.base_linear_is_parallel = True
        self.linear_in = nn.Linear(in_features, dim, bias=False)
        self.linear_out = nn.Linear(dim, out_features, bias=False)
        nn.init.xavier_normal_(self.linear_in.weight)
        nn.init.zeros_(self.linear_out.weight)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {
            f"{prefix}linear_in.weight": ("sharded", self.linear_in.weight),
            f"{prefix}linear_out.weight": ("sharded", self.linear_out.weight),
        }


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
    # ``reset_adapter`` re-inits through the model-parallel RNG tracker, which
    # has no initialized CUDA state on CPU; stub ``fork()`` to a no-op context.
    tracker = MagicMock()
    tracker.fork.side_effect = lambda *args, **kwargs: nullcontext()
    stack.enter_context(patch("megatron.core.tensor_parallel.random.get_cuda_rng_tracker", return_value=tracker))
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

    def test_constructor_forwards_wrapped_module_runtime_config(self) -> None:
        """Adapter construction mirrors the single-LoRA path (LoRA.transform)."""
        base = nn.Linear(16, 32)
        base.config = object()

        layer = MultiLoRALinear(to_wrap=base, n_adapters=2, dim=8, alpha=16, full_name="linear_proj")

        for adapter in layer.adapters:
            assert adapter.extra_kwargs["model_parallel_config"] is base.config
            assert adapter.extra_kwargs["disable_tensor_parallel_comm"] is False
            assert adapter.extra_kwargs["base_linear_is_parallel"] is True

    def test_slot_metadata_registered_as_buffers(self) -> None:
        layer = _build_multi_lora_linear(n_adapters=2, dim=8)

        buffers = dict(layer.named_buffers())
        assert "alpha_values" in buffers
        assert "rank_values" in buffers

        layer.to(torch.float64)
        assert layer.alpha_values.dtype == torch.float64
        assert layer.rank_values.dtype == torch.float64

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

    def test_sharded_state_dict_delegates_to_adapter_sharding(self) -> None:
        layer = _build_multi_lora_linear(n_adapters=2, dim=8)
        layer.to_wrap.sharded_state_dict = lambda prefix, sharded_offsets, metadata: {f"{prefix}weight": "base"}

        sharded_sd = layer.sharded_state_dict(prefix="decoder.layers.0.linear_proj.")

        assert sharded_sd["decoder.layers.0.linear_proj.weight"] == "base"
        for i in range(2):
            entry = sharded_sd[f"decoder.layers.0.linear_proj.adapters.{i}.linear_in.weight"]
            assert entry[0] == "sharded"
            assert entry[1] is layer.adapters[i].linear_in.weight
            entry = sharded_sd[f"decoder.layers.0.linear_proj.adapters.{i}.linear_out.weight"]
            assert entry[0] == "sharded"
            assert entry[1] is layer.adapters[i].linear_out.weight


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

    def test_expose_adapter_slot_syncs_export_scaling(self) -> None:
        """Exposed .alpha yields the slot's runtime scaling under alpha/dim."""
        container = _MultiLoRAContainer(n_layers=1)
        module = container.mods[0]
        module.init_adapter_slot(0, rank=4, alpha=16)

        with expose_adapter_slot(container, 0):
            assert module.adapter.dim == module.max_rank
            assert module.adapter.alpha == pytest.approx(16 * 8 / 4)

        with expose_adapter_slot(container, 1):
            assert module.adapter.alpha == pytest.approx(1.0)

        assert module.adapters[0].alpha == 16
        assert module.adapters[1].alpha == 16

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


# --------------------------------------------------------------------------- #
# B7: dropout is rejected. The assert is the first statement in __init__, so a
# dummy to_wrap is never touched.
# --------------------------------------------------------------------------- #
def test_dropout_rejected():
    with pytest.raises(AssertionError, match="does not apply adapter dropout"):
        MultiLoRALinear(
            to_wrap=nn.Linear(4, 4),
            n_adapters=2,
            dim=8,
            alpha=16,
            full_name="x",
            dropout=0.1,
        )


# --------------------------------------------------------------------------- #
# B2: the exposure/hiding context managers must restore state on exception.
# --------------------------------------------------------------------------- #
class _FakeMultiLoRALinear(MultiLoRALinear):
    """MultiLoRALinear instance without the heavy __init__ (isinstance still holds)."""

    def __init__(self):
        nn.Module.__init__(self)
        self.adapters = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        for adapter in self.adapters:
            adapter.alpha = 2.0
        self.alpha_values = torch.ones(2)
        self.rank_values = torch.full((2,), 2.0)
        self.max_rank = 2


def test_expose_adapter_slot_restores_on_exception():
    m = _FakeMultiLoRALinear()
    with pytest.raises(RuntimeError):
        with expose_adapter_slot(m, 0):
            # inside the context the slot is exposed and the list is hidden
            assert "adapter" in m._modules
            assert "adapters" not in m._modules
            raise RuntimeError("boom during export")
    # ...and it is fully restored despite the exception
    assert "adapters" in m._modules
    assert "adapter" not in m._modules
    assert m.adapters[0].alpha == 2.0


def test_hide_adapters_restores_on_exception():
    m = _FakeMultiLoRALinear()
    with pytest.raises(RuntimeError):
        with hide_adapters(m):
            assert "adapters" not in m._modules
            raise RuntimeError("boom during base load")
    assert "adapters" in m._modules


def test_expose_adapter_slot_restores_on_success():
    m = _FakeMultiLoRALinear()
    with expose_adapter_slot(m, 1):
        assert "adapter" in m._modules
    assert "adapters" in m._modules
    assert "adapter" not in m._modules


# --------------------------------------------------------------------------- #
# B8: expert linears are skipped, but with a one-time warning (not silently).
# --------------------------------------------------------------------------- #
def test_expert_skip_warns_once():
    multi_lora_mod._EXPERT_SKIP_WARNED = False
    mlora = MultiLoRA(target_modules=["linear_fc1"], n_adapters=2, dim=8, alpha=16)
    module = nn.Linear(4, 4)
    full = "decoder.layers.0.mlp.experts.linear_fc1"
    with (
        patch.object(multi_lora_mod, "is_expert_linear", return_value=True),
        patch.object(mlora, "match", return_value=(MagicMock(), full)),
        patch.object(multi_lora_mod, "logger") as logmock,
    ):
        out1 = mlora.transform(module, name="linear_fc1", prefix="decoder.layers.0.mlp.experts.")
        out2 = mlora.transform(module, name="linear_fc1", prefix="decoder.layers.0.mlp.experts.")

    # expert modules are returned unwrapped...
    assert out1 is module and out2 is module
    # ...and the warning fires exactly once across both skips
    assert logmock.warning.call_count == 1


# --------------------------------------------------------------------------- #
# B9: load_adapter raises on a checkpoint/model mismatch in either direction.
# --------------------------------------------------------------------------- #
class _AdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.adapter = nn.Module()
        self.layer.adapter.linear_in = nn.Linear(4, 8, bias=False)
        self.layer.adapter.linear_out = nn.Linear(8, 4, bias=False)


def test_load_adapter_partial_raises():
    m = _AdapterModel()
    partial = {"layer.adapter.linear_in.weight": torch.zeros(8, 4)}  # linear_out missing
    with pytest.raises(KeyError, match="absent from the checkpoint"):
        load_adapter(m, 0, partial)


def test_load_adapter_full_ok():
    m = _AdapterModel()
    full = {
        "layer.adapter.linear_in.weight": torch.ones(8, 4),
        "layer.adapter.linear_out.weight": torch.ones(4, 8),
    }
    assert load_adapter(m, 0, full) == 2
    assert torch.allclose(m.layer.adapter.linear_in.weight, torch.ones(8, 4))


def test_load_adapter_unused_keys_raises():
    m = _AdapterModel()
    over_full = {
        "layer.adapter.linear_in.weight": torch.ones(8, 4),
        "layer.adapter.linear_out.weight": torch.ones(4, 8),
        # e.g. saved with a larger target_modules set than the resuming model
        "other_layer.adapter.linear_in.weight": torch.ones(8, 4),
    }
    with pytest.raises(KeyError, match="matched no"):
        load_adapter(m, 0, over_full)


# --------------------------------------------------------------------------- #
# Forward smoke / B4 reset: single-GPU integration through a real
# ColumnParallelLinear.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU + model-parallel init")
class TestMultiLoRALinearGPU:
    @pytest.fixture(autouse=True)
    def _mp(self):
        import megatron.core.parallel_state as parallel_state
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29555")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("LOCAL_RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl", world_size=1, rank=0)
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        yield
        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass

    def _build(self, dim=8, n_adapters=2, alpha=16, column_init_method="xavier"):
        from megatron.core.tensor_parallel import ColumnParallelLinear
        from megatron.core.transformer.transformer_config import TransformerConfig

        from megatron.bridge.peft.utils import init_method_normal

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        base = ColumnParallelLinear(
            16,
            16,
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            gather_output=False,
        ).cuda()
        mlora = MultiLoRALinear(
            to_wrap=base,
            n_adapters=n_adapters,
            dim=dim,
            alpha=alpha,
            full_name="linear_qkv",
            column_init_method=column_init_method,
            row_init_method="zero",
            dropout=0.0,
        )
        # Mirror model setup: adapter weights are cast to the compute dtype
        # (the base is bf16).
        mlora.adapters.to(device="cuda", dtype=torch.bfloat16)
        return mlora

    def test_forward_grouped_gemm_smoke(self):
        from megatron.bridge.peft.multi_lora_layers import (
            init_adapter_slot,
            set_tokens_per_adapter_slot,
        )

        mlora = self._build(dim=8, n_adapters=2, alpha=12)
        init_adapter_slot([mlora], 0, rank=6, alpha=12)
        tokens = 4
        set_tokens_per_adapter_slot([mlora], torch.tensor([tokens, 0], dtype=torch.int32, device="cuda"))
        x = torch.randn(tokens, 16, dtype=torch.bfloat16, device="cuda")
        out, _ = mlora(x)
        assert out.shape[0] == tokens
        # scaling must not promote the activation dtype (bf16 * fp32 -> fp32)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_reset_adapter_through_rng_tracker(self):
        mlora = self._build()
        idx = 0
        # perturb so we can see the re-init take effect
        with torch.no_grad():
            mlora.adapters[idx].linear_in.weight.fill_(7.0)
            mlora.adapters[idx].linear_out.weight.fill_(7.0)
        mlora.clear_adapter_slot(idx)  # -> reset_adapter under get_cuda_rng_tracker().fork()
        a = mlora.adapters[idx].linear_in.weight
        b = mlora.adapters[idx].linear_out.weight
        assert not torch.allclose(a, torch.full_like(a, 7.0))  # A re-initialized (xavier)
        assert torch.count_nonzero(b) == 0  # B zero-initialized

    def test_reset_adapter_deterministic_via_rng_tracker(self):
        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        def reseed():
            _set_random_seed(
                seed_=1234,
                data_parallel_random_init=False,
                te_rng_tracker=True,
                inference_rng_tracker=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )

        mlora = self._build()
        idx = 0

        reseed()
        torch.cuda.manual_seed(111)  # a bare nn.init would draw from here...
        mlora.clear_adapter_slot(idx)
        first = mlora.adapters[idx].linear_in.weight.clone()

        reseed()
        torch.cuda.manual_seed(222)  # ...and this seed change would alter its result
        mlora.clear_adapter_slot(idx)
        second = mlora.adapters[idx].linear_in.weight

        # An identically re-seeded tracker must give an identical re-init
        # regardless of the default generator — the draw has to come from the
        # tracker stream (this is what keeps DP replicas equal on slot reuse).
        assert torch.equal(first, second)

    def test_reset_adapter_mirrors_construction_init_methods(self):
        mlora = self._build(column_init_method="zero")
        idx = 0
        with torch.no_grad():
            mlora.adapters[idx].linear_in.weight.fill_(7.0)
        mlora.clear_adapter_slot(idx)
        # Construction used column_init_method="zero"; reset must reuse it (a
        # hardcoded xavier re-init would leave nonzero values here).
        assert torch.count_nonzero(mlora.adapters[idx].linear_in.weight) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
