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

"""Unit tests for the multi-adapter LoRA implementation.

Mirrors the structure of ``test_lora.py``/``test_lora_layers.py`` but covers the
multi-adapter specific behaviour:

* :class:`MultiLoRA` transform / module-matching logic.
* :class:`MultiLoRALinear` per-slot rank/alpha bookkeeping and rank masking.
* The standalone slot-management helpers (routing, init/clear, expose/hide, load).

The transform and slot tests run on CPU by patching the heavy
``ParallelLinearAdapter`` / ``MultiLoRALinear`` dependencies with light fakes
that share the same weight layout. A single Megatron integration test exercises
the real construction path and is gated on a GPU being available.
"""

import datetime
import os
from contextlib import ExitStack
from unittest.mock import patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft import multi_lora as multi_lora_module
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


class FakeMultiLoRALinear(nn.Module):
    """Stand-in for ``MultiLoRALinear`` that records the constructor kwargs.

    Used to test :meth:`MultiLoRA.transform` matching/wiring on CPU without
    constructing real parallel adapters.
    """

    def __init__(self, to_wrap: nn.Module, **kwargs) -> None:
        super().__init__()
        self.to_wrap = to_wrap
        self.init_kwargs = kwargs


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
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha if alpha is not None else dim
        self.base_linear_name = base_linear_name
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


def multi_lora_linear_patch():
    """Patch ``MultiLoRALinear`` in the transform module with a recording fake."""
    return patch.object(multi_lora_module, "MultiLoRALinear", FakeMultiLoRALinear)


def multi_lora_topk_router_patch(router_cls: type):
    """Patch ``TopKRouter`` in the transform module with a dummy router type."""
    return patch.object(multi_lora_module, "TopKRouter", router_cls)


def adapter_deps_patch() -> ExitStack:
    """Patch the layer module's adapter construction dependencies for CPU use."""
    stack = ExitStack()
    stack.enter_context(patch.object(multi_lora_layers_module, "ParallelLinearAdapter", _FakeParallelLinearAdapter))
    stack.enter_context(patch.object(multi_lora_layers_module, "get_adapter_attributes_from_linear", _fake_get_attrs))
    return stack


# ======================================================================
# Test models (plain nn.Linear; MultiLoRALinear is patched out for matching)
# ======================================================================


class SimpleModel(nn.Module):
    """Simple model with the canonical target/non-target linear names."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.linear_qkv = nn.Linear(32, 96)
        self.linear_proj = nn.Linear(32, 32)
        self.linear_fc1 = nn.Linear(32, 64)
        self.linear_fc2 = nn.Linear(64, 32)
        self.output_projection = nn.Linear(32, 100)  # not a target
        self.layernorm = nn.LayerNorm(32)


class NestedModel(nn.Module):
    """Two-layer model with attention/mlp sub-blocks for pattern matching."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": nn.ModuleDict(
                            {"linear_qkv": nn.Linear(32, 96), "linear_proj": nn.Linear(32, 32)}
                        ),
                        "mlp": nn.ModuleDict({"linear_fc1": nn.Linear(32, 64), "linear_fc2": nn.Linear(64, 32)}),
                    }
                )
                for _ in range(2)
            ]
        )


class MoEModel(nn.Module):
    """Model with a dense MLP linear and an expert linear of the same name."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        layer = self.decoder.layers[0]
        layer.mlp = nn.Module()
        layer.mlp.linear_fc1 = nn.Linear(32, 64)  # dense -> should be wrapped
        layer.mlp.experts = nn.Module()
        layer.mlp.experts.linear_fc1 = nn.Linear(32, 64)  # expert -> should be skipped


class _DummyTopKRouter(nn.Module):
    """Minimal router placeholder used as the patched ``TopKRouter`` type."""

    def __init__(self, hidden_size: int = 32, num_experts: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))


class RouterModel(nn.Module):
    def __init__(self, router: nn.Module) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.router = router


# ======================================================================
# MultiLoRA: configuration + checkpoint key filtering
# ======================================================================


class TestMultiLoRAConfig:
    """Configuration defaults, overrides, and adapter key filtering."""

    def test_default_initialization(self) -> None:
        peft = MultiLoRA()
        assert peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
        assert peft.n_adapters == 2
        assert peft.dim == 32
        assert peft.alpha == 32
        assert peft.dropout == 0.0
        assert peft.dropout_position == "pre"
        assert peft.lora_A_init_method == "xavier"
        assert peft.lora_B_init_method == "zero"
        assert peft.a2a_experimental is False
        assert peft.lora_dtype is None

    def test_custom_initialization(self) -> None:
        peft = MultiLoRA(
            target_modules=["linear_qkv"],
            n_adapters=8,
            dim=16,
            alpha=8,
            dropout=0.1,
            dropout_position="post",
            lora_A_init_method="uniform",
            lora_B_init_method="kaiming",
            a2a_experimental=True,
        )
        assert peft.target_modules == ["linear_qkv"]
        assert peft.n_adapters == 8
        assert peft.dim == 16
        assert peft.alpha == 8
        assert peft.dropout == 0.1
        assert peft.dropout_position == "post"
        assert peft.lora_A_init_method == "uniform"
        assert peft.lora_B_init_method == "kaiming"
        assert peft.a2a_experimental is True

    def test_adapter_key_filter_string_keys(self) -> None:
        peft = MultiLoRA()
        assert peft.adapter_key_filter("decoder.layers.0.linear_qkv.adapters.0.linear_in.weight")
        assert peft.adapter_key_filter("decoder.layers.0.linear_qkv.weight_A.0")
        assert peft.adapter_key_filter("decoder.layers.0.linear_qkv.weight_B.0")
        assert not peft.adapter_key_filter("decoder.layers.0.linear_qkv.weight")
        assert not peft.adapter_key_filter("decoder.embedding.word_embeddings.weight")

    def test_adapter_key_filter_tuple_keys(self) -> None:
        peft = MultiLoRA()
        trainable = nn.Parameter(torch.zeros(1))
        frozen = nn.Parameter(torch.zeros(1))
        frozen.requires_grad = False
        assert peft.adapter_key_filter(("adapters.0.linear_in.weight", trainable))
        assert not peft.adapter_key_filter(("to_wrap.weight", frozen))


# ======================================================================
# MultiLoRA.transform: matching / wiring (MultiLoRALinear patched out)
# ======================================================================


class TestMultiLoRATransform:
    """Module matching and constructor wiring of :meth:`MultiLoRA.transform`."""

    @pytest.fixture(autouse=True)
    def _patch_multi_lora_linear(self):
        with multi_lora_linear_patch():
            yield

    def test_transform_simple_model(self) -> None:
        model = SimpleModel()
        peft = MultiLoRA(target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])

        transformed = peft(model, training=True)

        assert isinstance(transformed.linear_qkv, FakeMultiLoRALinear)
        assert isinstance(transformed.linear_proj, FakeMultiLoRALinear)
        assert isinstance(transformed.linear_fc1, FakeMultiLoRALinear)
        assert isinstance(transformed.linear_fc2, FakeMultiLoRALinear)
        # Non-target / non-linear modules are untouched.
        assert isinstance(transformed.output_projection, nn.Linear)
        assert isinstance(transformed.embedding, nn.Embedding)
        assert isinstance(transformed.layernorm, nn.LayerNorm)

    def test_transform_forwards_constructor_arguments(self) -> None:
        model = SimpleModel()
        peft = MultiLoRA(
            target_modules=["linear_qkv"],
            n_adapters=4,
            dim=16,
            alpha=8,
            dropout=0.1,
            dropout_position="post",
            lora_A_init_method="uniform",
            lora_B_init_method="kaiming",
            a2a_experimental=True,
        )

        transformed = peft(model, training=True)

        kwargs = transformed.linear_qkv.init_kwargs
        assert kwargs["n_adapters"] == 4
        assert kwargs["dim"] == 16
        assert kwargs["alpha"] == 8
        assert kwargs["dropout"] == 0.1
        assert kwargs["dropout_position"] == "post"
        assert kwargs["column_init_method"] == "uniform"
        assert kwargs["row_init_method"] == "kaiming"
        assert kwargs["a2a_experimental"] is True
        assert kwargs["full_name"] == "linear_qkv"
        assert transformed.linear_qkv.to_wrap is not None

    def test_transform_nested_model(self) -> None:
        model = NestedModel()
        peft = MultiLoRA(target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"])

        transformed = peft(model, training=True)

        for layer in transformed.layers:
            assert isinstance(layer["attention"]["linear_qkv"], FakeMultiLoRALinear)
            assert isinstance(layer["attention"]["linear_proj"], FakeMultiLoRALinear)
            assert isinstance(layer["mlp"]["linear_fc1"], FakeMultiLoRALinear)
            assert isinstance(layer["mlp"]["linear_fc2"], FakeMultiLoRALinear)

    def test_transform_wildcard_matching(self) -> None:
        model = NestedModel()
        peft = MultiLoRA(target_modules=["layers.0.attention.*"])

        transformed = peft(model, training=True)

        assert isinstance(transformed.layers[0]["attention"]["linear_qkv"], FakeMultiLoRALinear)
        assert isinstance(transformed.layers[0]["attention"]["linear_proj"], FakeMultiLoRALinear)
        # MLP of layer 0 and everything in layer 1 stay as plain linears.
        assert isinstance(transformed.layers[0]["mlp"]["linear_fc1"], nn.Linear)
        assert isinstance(transformed.layers[1]["attention"]["linear_qkv"], nn.Linear)
        assert isinstance(transformed.layers[1]["mlp"]["linear_fc2"], nn.Linear)

    def test_transform_skips_expert_linear(self) -> None:
        model = MoEModel()
        peft = MultiLoRA(target_modules=["linear_fc1"])

        transformed = peft(model, training=True)

        layer = transformed.decoder.layers[0]
        # Dense MLP linear is wrapped; the routed-expert linear of the same name is skipped.
        assert isinstance(layer.mlp.linear_fc1, FakeMultiLoRALinear)
        assert isinstance(layer.mlp.experts.linear_fc1, nn.Linear)

    def test_transform_skips_topk_router(self) -> None:
        router = _DummyTopKRouter()
        model = RouterModel(router)
        peft = MultiLoRA(target_modules=["router"])

        with multi_lora_topk_router_patch(_DummyTopKRouter):
            transformed = peft(model, training=True)

        # The router matches by name but the explicit TopKRouter guard skips it.
        assert transformed.mlp.router is router
        assert not isinstance(transformed.mlp.router, FakeMultiLoRALinear)

    def test_transform_idempotent(self) -> None:
        model = SimpleModel()
        peft = MultiLoRA(target_modules=["linear_qkv", "linear_proj"])

        first = peft(model, training=True)
        first_qkv = first.linear_qkv
        first_proj = first.linear_proj

        second = peft(first, training=True)

        # Already-wrapped modules are returned as-is, not re-wrapped.
        assert second.linear_qkv is first_qkv
        assert second.linear_proj is first_proj

    def test_transform_list_of_chunks(self) -> None:
        chunks = [SimpleModel() for _ in range(3)]
        peft = MultiLoRA(target_modules=["linear_qkv"])

        transformed = peft(chunks, training=True)

        assert isinstance(transformed, list)
        assert len(transformed) == 3
        for chunk in transformed:
            assert isinstance(chunk.linear_qkv, FakeMultiLoRALinear)


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
# Megatron integration (GPU only)
# ======================================================================


@pytest.mark.run_only_on("gpu")
class TestMultiLoRAMegatronIntegration:
    """Apply MultiLoRA to a real GPT model via the provider pre-wrap hook."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            dist.init_process_group(
                backend="nccl" if device_count > 0 else "gloo",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=30),
            )

        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                context_parallel_size=1,
            )

        from megatron.core.process_groups_config import ProcessGroupCollection

        from megatron.bridge.training.initialize import _set_random_seed

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        _set_random_seed(
            seed_=1234,
            data_parallel_random_init=False,
            te_rng_tracker=True,
            inference_rng_tracker=False,
            pg_collection=pg_collection,
        )

        yield

        try:
            if parallel_state.model_parallel_is_initialized():
                parallel_state.destroy_model_parallel()
            if dist.is_initialized():
                dist.destroy_process_group()
                for key in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    os.environ.pop(key, None)
        except (NameError, AttributeError, RuntimeError):
            pass

    def test_multi_lora_with_gpt_model(self) -> None:
        model_provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=2,
            vocab_size=1000,
            ffn_hidden_size=256,
        )

        from megatron.core.process_groups_config import ProcessGroupCollection

        model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        peft = MultiLoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            n_adapters=2,
            dim=8,
            alpha=16,
        )

        model_provider.register_pre_wrap_hook(lambda model: peft(model, training=True))
        model_provider.finalize()

        adapted_model = model_provider.provide_distributed_model(ddp_config=None, wrap_with_ddp=False)
        assert isinstance(adapted_model, list)
        assert all(isinstance(chunk, MegatronModule) for chunk in adapted_model)

        adapted_model = [chunk.cuda() for chunk in adapted_model]

        found = [
            name
            for chunk in adapted_model
            for name, module in chunk.named_modules()
            if isinstance(module, MultiLoRALinear)
        ]
        assert len(found) > 0, "No MultiLoRALinear modules found in adapted model"

        total = sum(p.numel() for chunk in adapted_model for p in chunk.parameters())
        trainable = sum(p.numel() for chunk in adapted_model for p in chunk.parameters() if p.requires_grad)
        assert 0 < trainable < total
        assert trainable / total < 0.3
