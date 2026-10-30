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

"""Tests for DSA-safe pipeline segmentation in the Hybrid MLA provider."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.hybrid_mla_provider import _add_dsa_pipeline_separators


pytestmark = pytest.mark.unit

# Sixteen physical layers. With topk_freq=4, the DSA layers numbered 3, 7, 11 and 15
# reuse the indexer output of layers 1, 5, 9 and 13, so only 5, 9 and 13 can start a segment.
PATTERN = "D-" * 8
FOUR_SEGMENTS = "D-D-|D-D-|D-D-|D-D-"


def _segment(pattern: str, num_pipeline_segments: int, topk_freq: int = 4) -> str:
    return _add_dsa_pipeline_separators(
        pattern,
        num_pipeline_segments=num_pipeline_segments,
        topk_freq=topk_freq,
        skip_topk_offset=0,
    )


def test_unsegmented_pattern_gets_one_segment_per_virtual_chunk():
    assert _segment(PATTERN, 4) == FOUR_SEGMENTS


def test_explicit_segments_for_every_virtual_chunk_are_kept():
    assert _segment(FOUR_SEGMENTS, 4) == FOUR_SEGMENTS


def test_explicit_segments_must_match_the_chunk_count():
    with pytest.raises(ValueError, match="needs 2"):
        _segment(FOUR_SEGMENTS, 2)


def test_segment_boundary_inside_an_index_share_group_is_rejected():
    with pytest.raises(ValueError, match="IndexShare"):
        _segment("D-|D-D-D-|D-D-|D-D-", 4)


def test_too_few_full_indexer_boundaries_are_rejected():
    with pytest.raises(ValueError, match="valid boundaries"):
        _segment(PATTERN, 5)


def test_pipeline_without_virtual_stages_keeps_two_segments():
    assert _segment(PATTERN, 2, topk_freq=1) == "D-D-D-D-|D-D-D-D-"


def test_single_segment_and_mtp_suffix_are_preserved():
    assert _segment(PATTERN, 1) == PATTERN
    assert _segment(PATTERN + Symbols.MTP_SEPARATOR + "DE", 2, topk_freq=1).endswith(Symbols.MTP_SEPARATOR + "DE")


@pytest.mark.parametrize(("pp_size", "vp_size", "expected_segments"), [(2, None, 2), (2, 2, 4), (1, None, 1)])
def test_finalize_counts_virtual_pipeline_chunks(pp_size, vp_size, expected_segments):
    config_module = pytest.importorskip("transformers.models.glm5_next.configuration_glm5_next")
    from megatron.bridge.models.glm5next.glm53flash_bridge import GLM53FlashBridge

    config = config_module.Glm5NextConfig(
        text_config={
            "scoring_func": "sigmoid",
            "linear_attn_config": {"num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4},
        }
    )
    provider = GLM53FlashBridge().provider_bridge(SimpleNamespace(config=config))
    provider.pipeline_model_parallel_size = pp_size
    provider.virtual_pipeline_model_parallel_size = vp_size
    with patch.object(HybridModelProvider, "finalize"):
        provider.finalize()

    main_pattern = provider.hybrid_layer_pattern.split(Symbols.MTP_SEPARATOR)[0]
    assert main_pattern.count(Symbols.PIPE) + 1 == expected_segments
