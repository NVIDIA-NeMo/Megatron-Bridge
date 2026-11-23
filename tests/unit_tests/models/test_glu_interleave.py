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

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.glu import (
    classify_glu_fc1,
    deinterleave_glu_tensor,
    get_glu_interleave_size,
    interleave_glu_tensor,
    logical_to_physical_glu_tensor,
    physical_to_logical_glu_tensor,
)


ROUTED_WEIGHT = "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
SHARED_BIAS = "decoder.layers.0.mlp.shared_experts.linear_fc1.bias"
DENSE_WEIGHT = "decoder.layers.0.mlp.linear_fc1.weight"
GROUPED_WEIGHT = "decoder.layers.0.mlp.experts.linear_fc1.weight0"
GROUPED_BIAS = "decoder.layers.0.mlp.experts.linear_fc1.bias0"


def test_classify_glu_fc1_matches_checkpoint_selection(monkeypatch):
    monkeypatch.delenv("USE_ACT_FUSION_FOR_DENSE", raising=False)

    assert classify_glu_fc1(ROUTED_WEIGHT) == "routed_expert"
    assert classify_glu_fc1(ROUTED_WEIGHT.replace(".weight", ".to_wrap.weight")) == "routed_expert"
    assert classify_glu_fc1(SHARED_BIAS) == "shared_expert"
    assert classify_glu_fc1(DENSE_WEIGHT) is None
    assert classify_glu_fc1("decoder.layers.0.mlp.experts.local_experts.0.linear_fc2.weight") is None

    monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
    assert classify_glu_fc1(DENSE_WEIGHT) == "dense"


def test_get_glu_interleave_size_selects_routed_and_shared_config(monkeypatch):
    monkeypatch.setenv("USE_ACT_FUSION_FOR_DENSE", "1")
    config = SimpleNamespace(
        gated_linear_unit=True,
        moe_mlp_glu_interleave_size=2,
        moe_shared_expert_glu_interleave_size=4,
        use_grouped_gemm_for_shared_expert=True,
    )

    assert get_glu_interleave_size(config, ROUTED_WEIGHT) == 2
    assert get_glu_interleave_size(config, DENSE_WEIGHT) == 2
    assert get_glu_interleave_size(config, SHARED_BIAS) == 4


@pytest.mark.parametrize("shape", [(16,), (16, 3)])
def test_glu_layout_round_trip(shape):
    logical = torch.arange(torch.Size(shape).numel()).reshape(shape)

    physical = interleave_glu_tensor(logical, interleave_size=2)

    assert physical.is_contiguous()
    assert torch.equal(deinterleave_glu_tensor(physical, interleave_size=2), logical)


def test_configured_glu_layout_conversion_is_zero_copy_when_disabled():
    logical = torch.arange(32).reshape(8, 4)
    disabled_config = SimpleNamespace(moe_mlp_glu_interleave_size=None)

    physical = logical_to_physical_glu_tensor(
        logical,
        model_config=disabled_config,
        param_name=ROUTED_WEIGHT,
    )

    assert physical is logical
    assert (
        physical_to_logical_glu_tensor(
            physical,
            model_config=disabled_config,
            param_name=ROUTED_WEIGHT,
        )
        is logical
    )


def test_configured_glu_layout_conversion_restores_logical_rows():
    config = SimpleNamespace(gated_linear_unit=True, moe_mlp_glu_interleave_size=2)
    logical = torch.arange(32).reshape(8, 4)

    physical = logical_to_physical_glu_tensor(
        logical,
        model_config=config,
        param_name=ROUTED_WEIGHT,
    )

    assert not torch.equal(physical, logical)
    assert torch.equal(
        physical_to_logical_glu_tensor(
            physical,
            model_config=config,
            param_name=ROUTED_WEIGHT,
        ),
        logical,
    )


@pytest.mark.parametrize(
    ("param_name", "shape"),
    [
        (GROUPED_WEIGHT, (3, 8, 4)),
        (GROUPED_BIAS, (3, 8)),
    ],
)
def test_configured_glu_layout_conversion_supports_grouped_storage(param_name, shape):
    config = SimpleNamespace(gated_linear_unit=True, moe_mlp_glu_interleave_size=2)
    logical = torch.arange(torch.Size(shape).numel()).reshape(shape)

    physical = logical_to_physical_glu_tensor(
        logical,
        model_config=config,
        param_name=param_name,
    )

    assert not torch.equal(physical, logical)
    assert torch.equal(
        physical_to_logical_glu_tensor(
            physical,
            model_config=config,
            param_name=param_name,
        ),
        logical,
    )


@pytest.mark.parametrize("interleave_size", [True, 0, -1, 1.5])
def test_get_glu_interleave_size_rejects_invalid_config(interleave_size):
    config = SimpleNamespace(gated_linear_unit=True, moe_mlp_glu_interleave_size=interleave_size)

    with pytest.raises(ValueError, match="must be a positive integer"):
        get_glu_interleave_size(config, ROUTED_WEIGHT)


def test_glu_layout_conversion_rejects_incompatible_rows():
    with pytest.raises(ValueError, match=r"rows \(10\) must be divisible"):
        interleave_glu_tensor(torch.zeros(10, 4), interleave_size=4)


@pytest.mark.parametrize("shape,row_dim", [((12,), 0), ((3, 12, 5), 1)])
def test_glu_layout_matches_explicit_three_block_row_order(shape, row_dim):
    logical = torch.arange(torch.Size(shape).numel()).reshape(shape)
    # Independent oracle: three blocks distinguish the forward permutation
    # from its inverse, unlike the commonly used two-block fixtures.
    indices = torch.tensor([0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11])
    physical = logical.index_select(row_dim, indices)
    assert torch.equal(interleave_glu_tensor(logical, 2, row_dim=row_dim), physical)
    assert torch.equal(deinterleave_glu_tensor(physical, 2, row_dim=row_dim), logical)


@pytest.mark.parametrize("suffix", ["_scale_inv", "_extra_state", ".lora_A.weight"])
def test_glu_selection_excludes_metadata_and_adapters(suffix):
    assert classify_glu_fc1(GROUPED_WEIGHT + suffix) is None


def test_non_glu_config_leaves_fc1_unchanged_like_mcore():
    config = SimpleNamespace(gated_linear_unit=False, moe_mlp_glu_interleave_size=2)
    weight = torch.arange(60).reshape(12, 5)
    assert get_glu_interleave_size(config, ROUTED_WEIGHT) is None
    assert physical_to_logical_glu_tensor(weight, model_config=config, param_name=ROUTED_WEIGHT) is weight


def test_shared_expert_requires_grouped_gemm_like_mcore():
    config = SimpleNamespace(
        gated_linear_unit=True,
        moe_shared_expert_glu_interleave_size=2,
        use_grouped_gemm_for_shared_expert=False,
    )
    assert get_glu_interleave_size(config, SHARED_BIAS) is None
