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

"""Shensi runs on MCore's HybridModel: stack spec, pipeline pattern and MTP template."""

from types import SimpleNamespace

from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig
from megatron.bridge.models.shensi.shensi_hybrid import (
    ShensiDecoderStack,
    get_shensi_hybrid_stack_spec,
    shensi_hybrid_layer_pattern,
)
from megatron.bridge.models.shensi.shensi_provider import (
    SHENSI_HARD_DEFAULTS,
    shensi_derived_field_values,
)


def _transformer_config(*, num_layers: int = 4, n_hash: int = 1, mtp: int = 1, pp: int = 1):
    hf_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=["sliding_attention"] * num_layers,
        mlp_layer_types=["hash_moe"] * n_hash + ["moe"] * (num_layers - n_hash),
        n_routed_experts=8,
        hc_conv_kernels=(4, 8, 12),
        hidden_size=128,
        head_dim=64,
        qk_rope_head_dim=8,
        num_attention_heads=4,
        q_lora_rank=32,
        o_groups=4,
        o_lora_rank=32,
        sliding_window=128,
        compress_rope_theta=160000.0,
        index_n_heads=8,
        index_head_dim=16,
        index_topk=32,
        hc_mult=16,
        hc_active_streams=4,
        hc_fixed_streams=2,
        attn_res_block_size=4,
        moe_intermediate_size=32,
        routed_expert_hidden_size=32,
        erc_loss_alpha=0.5,
        erc_loss_coef=1.0,
        rope_theta=10000.0,
        vocab_size=128,
        scoring_func="sqrtsoftplus",
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    fields = dict(shensi_derived_field_values(hf_config, mtp_num_layers=mtp))
    fields.update(SHENSI_HARD_DEFAULTS)
    fields.update(
        num_layers=num_layers,
        mtp_num_layers=mtp if mtp else None,
        hidden_size=128,
        pipeline_model_parallel_size=pp,
    )
    return ShensiTransformerConfig(**fields)


def test_pattern_marks_every_layer_and_each_mtp_head() -> None:
    """One attention symbol per decoder layer, with the MTP heads after the '/' separator."""
    config = _transformer_config(num_layers=4, mtp=1)
    assert shensi_hybrid_layer_pattern(config) == Symbols.ATTENTION * 4 + Symbols.MTP_SEPARATOR + Symbols.ATTENTION
    assert config.hybrid_layer_pattern == Symbols.ATTENTION * 4 + Symbols.MTP_SEPARATOR + Symbols.ATTENTION


def test_pattern_has_no_mtp_section_without_mtp_heads() -> None:
    config = _transformer_config(num_layers=2, n_hash=1, mtp=0)
    assert shensi_hybrid_layer_pattern(config) == Symbols.ATTENTION * 2


def test_pattern_splits_pipeline_stages_and_writes_the_layout() -> None:
    """The stage split travels in ``pipeline_model_parallel_layout``, which the AttnRes plan reads."""
    config = _transformer_config(num_layers=4, mtp=1, pp=2)
    assert shensi_hybrid_layer_pattern(config) == "**|**" + Symbols.MTP_SEPARATOR + Symbols.ATTENTION
    assert config.pipeline_model_parallel_layout == [
        ["embedding", "decoder", "decoder"],
        ["decoder", "decoder", "mtp", "loss"],
    ]


def test_stack_spec_hosts_the_shensi_stack_and_the_mtp_template() -> None:
    """The decoder is the bridge-side stack; the MTP inner layer is built from the pattern."""
    config = _transformer_config(num_layers=4, mtp=1)
    spec = get_shensi_hybrid_stack_spec(config, vp_stage=None, pp_rank=0)

    assert spec.module is ShensiDecoderStack
    assert isinstance(spec.submodules, HybridStackSubmodules)
    assert spec.submodules.attention_layer.module.__name__ == "ShensiTransformerLayer"

    mtp_layers = spec.submodules.mtp_block_spec.layer_specs
    assert len(mtp_layers) == 1
    assert mtp_layers[0].module.__name__ == "ShensiMultiTokenPredictionLayer"
    # The hybrid path builds the MTP inner model layer from the pattern, so the recipe must not pin one.
    assert mtp_layers[0].submodules.mtp_model_layer is None


def test_stack_spec_has_no_mtp_template_without_mtp_heads() -> None:
    config = _transformer_config(num_layers=2, n_hash=1, mtp=0)
    spec = get_shensi_hybrid_stack_spec(config, vp_stage=None, pp_rank=0)
    assert spec.submodules.mtp_block_spec is None
