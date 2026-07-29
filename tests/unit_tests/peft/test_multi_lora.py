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

"""Unit tests for the :class:`MultiLoRA` PEFT object.

Mirrors ``test_lora.py``: covers the multi-adapter LoRA configuration, the
``transform`` module-matching / constructor wiring, and a GPU-gated end-to-end
application to a real Megatron GPT model.

``MultiLoRALinear`` is patched out with a lightweight recording fake so the
matching logic runs on CPU; the layer module itself (slot bookkeeping, helpers,
export seam) is exercised in ``test_multi_lora_layers.py``.
"""

import datetime
import os
from unittest.mock import patch

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft import multi_lora as multi_lora_module
from megatron.bridge.peft.multi_lora import MultiLoRA
from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear


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


def multi_lora_linear_patch():
    """Patch ``MultiLoRALinear`` in the transform module with a recording fake."""
    return patch.object(multi_lora_module, "MultiLoRALinear", FakeMultiLoRALinear)


def multi_lora_topk_router_patch(router_cls: type):
    """Patch ``TopKRouter`` in the transform module with a dummy router type."""
    return patch.object(multi_lora_module, "TopKRouter", router_cls)


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
