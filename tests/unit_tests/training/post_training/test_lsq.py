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
"""Unit tests for megatron.bridge.training.post_training.lsq."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import modelopt.torch.opt as mto
import pytest
import torch
from megatron.core.optimizer import OptimizerConfig, ParamKey, _get_param_groups
from megatron.core.optimizer.optimizer import param_group_identifier_keys
from megatron.core.transformer.transformer_block import TransformerBlock

from megatron.bridge.training.config import OptimizerConfigOverrideProvider, OptimizerConfigOverrideProviderContext
from megatron.bridge.training.post_training.lsq import (
    LSQ_SCALE_PARAM_PATTERNS,
    LSQOptimizerConfigOverrideProvider,
    freeze_all_but_lsq_scales,
    propagate_expert_allreduce_to_lsq_params,
)


LSQ_KEY = ParamKey(name=LSQ_SCALE_PARAM_PATTERNS)


class _Quantizer(torch.nn.Module):
    """Stand-in for a ModelOpt static-block weight quantizer after ``enable_lsq``."""

    def __init__(self, *, pre: bool, scale_shape: tuple[int, ...] = (4, 1)):
        super().__init__()
        self._amax_post = torch.nn.Parameter(torch.ones(scale_shape))
        if pre:
            self._amax_pre = torch.nn.Parameter(torch.ones(scale_shape))
        self.register_buffer("_global_amax", torch.tensor(1.0))


class _QuantLinear(torch.nn.Module):
    def __init__(self, allreduce: bool | None, *, pre: bool = False, scale_shape: tuple[int, ...] = (4, 1)):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 16))
        if allreduce is not None:
            self.weight.allreduce = allreduce
        self.bias = torch.nn.Parameter(torch.zeros(4))
        self.weight_quantizer = _Quantizer(pre=pre, scale_shape=scale_shape)


class _QuantGroupedLinear(torch.nn.Module):
    """Grouped-GEMM experts: ``weight0..N`` behind one quantizer container."""

    def __init__(self, allreduce_flags: list[bool]):
        super().__init__()
        for i, allreduce in enumerate(allreduce_flags):
            weight = torch.nn.Parameter(torch.ones(4, 16))
            weight.allreduce = allreduce
            self.register_parameter(f"weight{i}", weight)
        self.weight_quantizer = torch.nn.ModuleList([_Quantizer(pre=False) for _ in allreduce_flags])


class _ToyMoEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = _QuantLinear(allreduce=True, pre=True)
        self.experts = _QuantGroupedLinear([False, False])
        self.local_experts = torch.nn.ModuleList([_QuantLinear(allreduce=False)])
        self.norm = torch.nn.LayerNorm(16)
        self.router = torch.nn.Module()
        self.router.register_buffer("expert_bias", torch.zeros(2))


class _PlainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 4)


def _transformer_block(*, recompute_granularity: str | None, quantized: bool) -> TransformerBlock:
    """A TransformerBlock built without its constructor, which needs a full model config."""
    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(recompute_granularity=recompute_granularity)
    block.linear = _QuantLinear(allreduce=None) if quantized else torch.nn.Linear(16, 4)
    block.forward = lambda hidden_states, **kwargs: hidden_states
    return block


class _ToyDistillationModel(torch.nn.Module):
    """A quantized student block next to a frozen BF16 teacher block."""

    def __init__(self, recompute_granularity: str | None):
        super().__init__()
        self.decoder = _transformer_block(recompute_granularity=recompute_granularity, quantized=True)
        self.teacher = _transformer_block(recompute_granularity=recompute_granularity, quantized=False)
        self.teacher.requires_grad_(False)


@pytest.fixture
def single_rank_process_group(tmp_path):
    """Initialize a single-rank process group; MCore aligns param groups across ranks."""
    initialized_here = not torch.distributed.is_initialized()
    if initialized_here:
        torch.distributed.init_process_group(
            "gloo",
            init_method=(tmp_path / "process_group_init").as_uri(),
            rank=0,
            world_size=1,
        )

    yield

    if initialized_here:
        torch.distributed.destroy_process_group()


def _scale_params(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if LSQ_KEY.matches(param, name)}


def _context(model: torch.nn.Module, *, lr: float, min_lr: float) -> OptimizerConfigOverrideProviderContext:
    return OptimizerConfigOverrideProviderContext(
        scheduler_config=Mock(),
        optimizer_config=OptimizerConfig(optimizer="adam", lr=lr, min_lr=min_lr, bf16=True),
        model=[model],
    )


def _group_identifier(group: dict) -> tuple:
    return tuple(group.get(key) for key in param_group_identifier_keys)


@pytest.mark.unit
class TestPropagateExpertAllreduceToLSQParams:
    def test_copies_flag_from_owning_weight(self):
        model = _ToyMoEModel()

        assert propagate_expert_allreduce_to_lsq_params(model) == 5

        scales = _scale_params(model)
        assert scales["linear_qkv.weight_quantizer._amax_pre"].allreduce is True
        assert scales["linear_qkv.weight_quantizer._amax_post"].allreduce is True
        assert scales["experts.weight_quantizer.0._amax_post"].allreduce is False
        assert scales["experts.weight_quantizer.1._amax_post"].allreduce is False
        assert scales["local_experts.0.weight_quantizer._amax_post"].allreduce is False
        assert not hasattr(model.local_experts[0].bias, "allreduce")
        assert not hasattr(model.norm.weight, "allreduce")

    @pytest.mark.parametrize("make_model", [lambda: _QuantLinear(allreduce=None), _PlainModel])
    def test_noop_without_flag_or_scales(self, make_model):
        model = make_model()

        assert propagate_expert_allreduce_to_lsq_params(model) == 0
        assert not any(hasattr(param, "allreduce") for param in model.parameters())

    def test_rejects_weights_with_mixed_flags(self):
        with pytest.raises(ValueError, match="disagree on the 'allreduce' attribute"):
            propagate_expert_allreduce_to_lsq_params(_QuantGroupedLinear([False, True]))


@pytest.mark.unit
class TestFreezeAllButLSQScales:
    def test_only_scales_stay_trainable(self):
        model = _ToyMoEModel()
        scale_names = set(_scale_params(model))

        assert freeze_all_but_lsq_scales([model]) == [model]

        assert {name for name, param in model.named_parameters() if param.requires_grad} == scale_names
        assert model.router.frozen_expert_bias

    def test_rejects_restored_model_without_scales(self):
        with patch.object(mto.ModeloptStateManager, "is_converted", return_value=True):
            with pytest.raises(ValueError, match="No LSQ scale parameters"):
                freeze_all_but_lsq_scales([_PlainModel()])

    def test_runs_before_modelopt_restore(self):
        """Run before the restore, the hook freezes the model and the scales created later train."""
        model = _ToyDistillationModel("full")
        model.decoder.linear = torch.nn.Linear(16, 4)

        freeze_all_but_lsq_scales([model])
        assert not any(param.requires_grad for param in model.parameters())
        assert not model.decoder.forward(hidden_states=torch.ones(2, 16)).requires_grad

        # What restoring the ModelOpt state does: add a quantizer with learnable scales.
        model.decoder.linear.weight_quantizer = _Quantizer(pre=False)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        assert trainable == {"decoder.linear.weight_quantizer._amax_post"}
        assert model.decoder.forward(hidden_states=torch.ones(2, 16)).requires_grad

    @pytest.mark.parametrize(
        ("recompute_granularity", "patched"), [("full", True), ("selective", True), (None, False)]
    )
    def test_recomputed_student_block_input_requires_grad(self, recompute_granularity, patched):
        """With the embedding frozen, recompute needs a grad-requiring block input to run backward."""
        model = _ToyDistillationModel(recompute_granularity)
        freeze_all_but_lsq_scales([model])
        decoder_forward = model.decoder.forward
        freeze_all_but_lsq_scales([model])
        assert model.decoder.forward is decoder_forward  # patched once

        hidden_states = torch.ones(2, 16)
        assert model.decoder.forward(hidden_states=hidden_states).requires_grad == patched
        assert not model.teacher.forward(hidden_states=hidden_states).requires_grad
        with torch.no_grad():
            assert not model.decoder.forward(hidden_states=hidden_states).requires_grad
        assert model.decoder.forward(hidden_states=None) is None


@pytest.mark.unit
class TestLSQOptimizerConfigOverrideProvider:
    def test_adds_scale_group_and_keeps_standard_overrides(self):
        context = _context(_ToyMoEModel(), lr=1e-5, min_lr=1e-6)
        standard = OptimizerConfigOverrideProvider().build_config_overrides(context)

        overrides = LSQOptimizerConfigOverrideProvider(
            scale_lr=1e-4, scale_min_lr=None, scale_wd_mult=0.0
        ).build_config_overrides(context)

        assert overrides[LSQ_KEY] == {"max_lr": 1e-4, "wd_mult": 0.0}
        for key, value in standard.items():
            assert overrides[key] == value

    @pytest.mark.parametrize(
        ("lr", "min_lr", "scale_lr", "scale_min_lr", "scale_wd_mult", "expected"),
        [
            (0.0, 0.0, 3e-5, None, 0.0, {"max_lr": 3e-5, "wd_mult": 0.0}),
            (0.0, 0.0, 3e-5, 3e-6, 1.0, {"max_lr": 3e-5, "min_lr": 3e-6}),
            (1e-5, 1e-6, 1e-5, 1e-6, 1.0, None),  # equal to the base: no scale group
        ],
    )
    def test_overrides_only_differing_settings(self, lr, min_lr, scale_lr, scale_min_lr, scale_wd_mult, expected):
        overrides = LSQOptimizerConfigOverrideProvider(
            scale_lr=scale_lr, scale_min_lr=scale_min_lr, scale_wd_mult=scale_wd_mult
        ).build_config_overrides(_context(_ToyMoEModel(), lr=lr, min_lr=min_lr))

        assert (overrides or {}).get(LSQ_KEY) == expected

    def test_warns_without_scale_params(self):
        with patch("megatron.bridge.training.post_training.lsq.warn_rank_0") as mock_warn:
            LSQOptimizerConfigOverrideProvider(
                scale_lr=1e-4, scale_min_lr=None, scale_wd_mult=0.0
            ).build_config_overrides(_context(_PlainModel(), lr=1e-5, min_lr=1e-6))

        mock_warn.assert_called_once()

    @pytest.mark.parametrize(
        ("lr", "scale_lr", "scale_wd_mult", "expected_scale_max_lr", "expected_scale_wd_mult"),
        [
            (1e-5, 1e-4, 0.0, 1e-4, 0.0),  # full QAD, scales at their own LR
            (0.0, 3e-5, 0.0, 3e-5, 0.0),  # scale-only
            (1e-5, 1e-5, 0.0, 1e-5, 0.0),  # same LR: scales share the no-weight-decay group
            (1e-5, 1e-5, 1.0, 1e-5, 1.0),  # same settings: scales share the default group
        ],
    )
    def test_param_groups_have_unique_identifiers(
        self,
        single_rank_process_group,
        lr,
        scale_lr,
        scale_wd_mult,
        expected_scale_max_lr,
        expected_scale_wd_mult,
    ):
        """Distinct groups must differ in the fields MCore uses to match groups on checkpoint load."""
        model = _ToyMoEModel()
        context = _context(model, lr=lr, min_lr=0.0)
        overrides = LSQOptimizerConfigOverrideProvider(
            scale_lr=scale_lr, scale_min_lr=None, scale_wd_mult=scale_wd_mult
        ).build_config_overrides(context)

        param_groups = _get_param_groups([model], context.optimizer_config, overrides)

        identifiers = [_group_identifier(group) for group in param_groups]
        assert len(identifiers) == len(set(identifiers))
        scale_param_ids = {id(param) for param in _scale_params(model).values()}
        scale_groups = [group for group in param_groups if scale_param_ids & {id(p) for p in group["params"]}]
        assert scale_groups
        for group in scale_groups:
            assert group["max_lr"] == expected_scale_max_lr
            assert group["wd_mult"] == expected_scale_wd_mult

    def test_one_dimensional_scales_combine_with_standard_overrides(self, single_rank_process_group):
        """1-D scales also match the standard no-weight-decay key; the merged overrides must agree."""
        model = _QuantLinear(allreduce=True, pre=True, scale_shape=(4,))
        context = _context(model, lr=1e-5, min_lr=0.0)
        overrides = LSQOptimizerConfigOverrideProvider(
            scale_lr=1e-4, scale_min_lr=None, scale_wd_mult=0.0
        ).build_config_overrides(context)

        param_groups = _get_param_groups([model], context.optimizer_config, overrides)

        scale_param_ids = {id(param) for param in _scale_params(model).values()}
        (scale_group,) = [group for group in param_groups if scale_param_ids & {id(p) for p in group["params"]}]
        assert {id(p) for p in scale_group["params"]} == scale_param_ids
        assert scale_group["max_lr"] == 1e-4
        assert scale_group["wd_mult"] == 0.0
