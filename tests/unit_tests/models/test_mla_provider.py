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

"""Unit tests for shared MLA model providers."""

from unittest.mock import patch

import pytest

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.mla_provider import HybridMLAModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


pytestmark = pytest.mark.unit


def test_hybrid_mla_provider_inherits_hybrid_and_mla_configs() -> None:
    """Hybrid MLA exposes both the Hybrid provider and MLA config fields."""
    assert issubclass(HybridMLAModelProvider, HybridModelProvider)
    assert issubclass(HybridMLAModelProvider, MLATransformerConfig)

    provider = HybridMLAModelProvider(q_lora_rank=16, kv_lora_rank=8)
    assert provider.q_lora_rank == 16
    assert provider.kv_lora_rank == 8


def test_hybrid_mla_provider_adds_balanced_indexshare_safe_pipeline_segments() -> None:
    """Pipeline boundaries keep logical pairs and IndexShare groups together."""
    provider = HybridMLAModelProvider(
        num_layers=8,
        hybrid_layer_pattern="D-DEDEDE",
        pipeline_model_parallel_size=2,
        dsa_indexer_topk_freq=8,
        dsa_indexer_skip_topk_offset=5,
    )

    with patch.object(HybridModelProvider, "finalize") as base_finalize:
        provider.finalize()

    assert provider.hybrid_layer_pattern == "D-DE|DEDE"
    base_finalize.assert_called_once_with()


def test_hybrid_mla_provider_rejects_cross_stage_indexshare() -> None:
    """An explicit boundary cannot place a shared indexer on a new stage."""
    provider = HybridMLAModelProvider(
        num_layers=8,
        hybrid_layer_pattern="D-DEDE|DE",
        pipeline_model_parallel_size=2,
        dsa_indexer_topk_freq=8,
        dsa_indexer_skip_topk_offset=5,
    )

    with patch.object(HybridModelProvider, "finalize"), pytest.raises(
        ValueError, match="cannot split DSA IndexShare groups"
    ):
        provider.finalize()


def test_hybrid_mla_provider_rejects_split_logical_pair() -> None:
    """An explicit boundary cannot separate a DSA layer from its FFN."""
    provider = HybridMLAModelProvider(
        num_layers=8,
        hybrid_layer_pattern="D|-DEDEDE",
        pipeline_model_parallel_size=2,
        dsa_indexer_topk_freq=8,
        dsa_indexer_skip_topk_offset=5,
    )

    with patch.object(HybridModelProvider, "finalize"), pytest.raises(ValueError, match="before DSA layers"):
        provider.finalize()


def test_hybrid_mla_provider_rejects_pipeline_layout_without_enough_full_indexers() -> None:
    """Every pipeline stage after the first must begin with a full indexer."""
    provider = HybridMLAModelProvider(
        num_layers=8,
        hybrid_layer_pattern="D-DEDEDE",
        pipeline_model_parallel_size=4,
        dsa_indexer_topk_freq=8,
        dsa_indexer_skip_topk_offset=5,
    )

    with patch.object(HybridModelProvider, "finalize"), pytest.raises(ValueError, match="valid boundaries"):
        provider.finalize()
