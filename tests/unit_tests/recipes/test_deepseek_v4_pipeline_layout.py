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

"""Native pipeline allocation for DeepSeek V4 attention/MoE pairs."""

from unittest.mock import patch

import pytest
from megatron.core.models.hybrid import hybrid_layer_allocation as allocation
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider


pytestmark = pytest.mark.unit


def _provider(logical_layers, pp, vp=None):
    pattern = "WEWE" + "".join("CE" if i % 2 == 0 else "HE" for i in range(logical_layers - 2))
    return DeepSeekV4HybridModelProvider(
        num_layers=2 * logical_layers,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_ffn_hidden_size=256,
        ffn_hidden_size=256,
        vocab_size=1280,
        hybrid_layer_pattern=pattern,
        mtp_num_layers=1,
        mtp_hybrid_override_pattern="WE",
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
    )


@pytest.mark.parametrize(
    ("logical_layers", "pp", "vp", "counts"),
    [(43, 8, None, [6] * 3 + [5] * 5), (43, 4, 4, [3] * 11 + [2] * 5), (61, 4, None, [16, 15, 15, 15])],
)
def test_native_pipeline_segments_match_mcore_allocation(logical_layers, pp, vp, counts):
    cfg = _provider(logical_layers, pp, vp)
    original_pattern = cfg.hybrid_layer_pattern
    set_deepseek_v4_pipeline_model_parallel_layout(cfg)

    segments = cfg.hybrid_layer_pattern.split("|")
    assert "".join(segments) == original_pattern
    assert [len(segment) for segment in segments] == [2 * count for count in counts]
    assert all(segment.endswith("E") for segment in segments)
    assert segments[0].count("E") >= 3
    layout = PipelineParallelLayerLayout(cfg.pipeline_model_parallel_layout, pipeline_model_parallel_size=pp)
    mtp_standalone = layout.validate_layer_layout(cfg.num_layers, cfg.mtp_num_layers)
    assert not mtp_standalone
    assert layout.layout[0][0][0] is LayerType.embedding
    assert layout.layout[-1][-1][-2:] == [LayerType.mtp, LayerType.loss]

    for index, segment in enumerate(segments):
        pp_rank, vp_rank = index % pp, index // pp
        with (
            patch.object(allocation.torch.distributed, "get_rank", return_value=pp_rank),
            patch.object(allocation.torch.distributed, "get_world_size", return_value=pp),
            patch.object(allocation, "log_on_each_pipeline_stage"),
        ):
            configs, offset = allocation.select_pipeline_segment(
                cfg.hybrid_layer_pattern, cfg, object(), vp_rank if vp else None
            )
        assert offset == 2 * sum(counts[:index])
        assert "".join(allocation.get_layer_type_list_from_layer_config_list(configs)) == segment
        assert layout.layout[pp_rank][vp_rank].count(LayerType.decoder) == len(configs)


def test_pro_perf_preserves_measured_last_stage_room_for_mtp():
    cfg = _provider(61, 4, 4)
    set_deepseek_v4_pipeline_model_parallel_layout(cfg, logical_layers_per_stage=[4] * 15 + [1])
    assert [len(segment) for segment in cfg.hybrid_layer_pattern.split("|")] == [8] * 15 + [2]
    assert cfg.pipeline_model_parallel_layout[-1] == ["decoder", "decoder", "mtp", "loss"]


def test_return_to_pp_one_removes_inherited_stage_boundaries_and_keeps_mtp():
    cfg = _provider(43, 8)
    original_pattern = cfg.hybrid_layer_pattern
    cfg.hybrid_layer_pattern += "/WE"
    set_deepseek_v4_pipeline_model_parallel_layout(cfg)
    cfg.pipeline_model_parallel_size = 1
    set_deepseek_v4_pipeline_model_parallel_layout(cfg)
    assert cfg.pipeline_model_parallel_layout is None
    assert cfg.hybrid_layer_pattern == original_pattern + "/WE"


def test_reapplying_layout_is_idempotent():
    cfg = _provider(43, 4, 4)
    set_deepseek_v4_pipeline_model_parallel_layout(cfg)
    expected = cfg.hybrid_layer_pattern, cfg.pipeline_model_parallel_layout
    set_deepseek_v4_pipeline_model_parallel_layout(cfg)
    assert (cfg.hybrid_layer_pattern, cfg.pipeline_model_parallel_layout) == expected


@pytest.mark.parametrize("counts", [[4] * 16, [4] * 14 + [5], [4] * 15 + [-1]])
def test_invalid_stage_counts_fail_before_model_construction(counts):
    cfg = _provider(61, 4, 4)
    with pytest.raises(ValueError, match="stage counts"):
        set_deepseek_v4_pipeline_model_parallel_layout(cfg, logical_layers_per_stage=counts)
