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

import pytest

from megatron.bridge.models.falcon_h1.modeling_falconh1.mamba_hybrid_layer_allocation import (
    Symbols,
    allocate_layers,
)


pytestmark = pytest.mark.unit


def test_parallel_hybrid_ratio_allocates_falcon_h1_layers():
    layers = allocate_layers(
        total_layers_count=4,
        target_attention_ratio=0.0,
        target_mlp_ratio=0.0,
        target_parallel_hybrid_ratio=1.0,
    )

    assert layers == [Symbols.PARALLEL] * 4


def test_auto_allocation_matches_requested_layer_counts():
    layers = allocate_layers(
        total_layers_count=10,
        target_attention_ratio=0.2,
        target_mlp_ratio=0.1,
        target_parallel_hybrid_ratio=0.2,
    )

    assert layers.count(Symbols.ATTENTION) == 2
    assert layers.count(Symbols.MLP) == 1
    assert layers.count(Symbols.PARALLEL) == 2
    assert layers.count(Symbols.MAMBA) == 5


def test_override_pattern_can_define_all_layers_when_ratios_are_zero():
    layers = allocate_layers(
        total_layers_count=4,
        target_attention_ratio=0.0,
        target_mlp_ratio=0.0,
        target_parallel_hybrid_ratio=0.0,
        override_pattern="M*P-",
    )

    assert layers == [Symbols.MAMBA, Symbols.ATTENTION, Symbols.PARALLEL, Symbols.MLP]


def test_override_pattern_length_is_validated():
    with pytest.raises(ValueError, match="wrong length"):
        allocate_layers(
            total_layers_count=4,
            target_attention_ratio=0.0,
            target_mlp_ratio=0.0,
            target_parallel_hybrid_ratio=0.0,
            override_pattern="M*P",
        )


def test_override_pattern_symbols_are_validated():
    with pytest.raises(ValueError, match="not one of"):
        allocate_layers(
            total_layers_count=4,
            target_attention_ratio=0.0,
            target_mlp_ratio=0.0,
            target_parallel_hybrid_ratio=0.0,
            override_pattern="M*PX",
        )


def test_override_pattern_counts_must_match_nonzero_ratios():
    with pytest.raises(ValueError, match="number of each type of layer"):
        allocate_layers(
            total_layers_count=4,
            target_attention_ratio=0.5,
            target_mlp_ratio=0.0,
            target_parallel_hybrid_ratio=0.0,
            override_pattern="MMMM",
        )
