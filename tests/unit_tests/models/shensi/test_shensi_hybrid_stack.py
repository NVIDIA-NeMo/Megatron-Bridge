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
from megatron.bridge.models.shensi.modeling_shensi import shensi_layer_plan
from megatron.bridge.models.shensi.shensi_hybrid import (
    ShensiDecoderStack,
    get_shensi_hybrid_stack_spec,
    shensi_hybrid_layer_pattern,
)
from megatron.bridge.models.shensi.shensi_provider import (
    SHENSI_HARD_DEFAULTS,
    shensi_derived_field_values,
)


def _transformer_config(*, num_layers: int = 4, n_dense: int = 1, mtp: int = 1, pp: int = 1):
    hf_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        mlp_layer_types=["dense"] * n_dense + ["sparse"] * (num_layers - n_dense),
        first_k_dense_replace=n_dense,
        n_routed_experts=8,
        hidden_size=128,
        head_dim=64,
        qk_rope_head_dim=0,
        v_head_dim=64,
        num_attention_heads=4,
        q_lora_rank=32,
        latent_size=32,
        deepemb_kv_rank=16,
        kv_emb_rank=16,
        deepemb_rank=8,
        index_n_heads=8,
        index_head_dim=16,
        index_topk=32,
        index_topk_blocks=4,
        index_block_size=4,
        index_topk_freq=2,
        index_latent_rank=4,
        linear_num_heads=4,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        hc_mult=2,
        hc_active_streams=2,
        hc_adapter_rank=4,
        hc_fixed_streams=1,
        hc_conv_kernels=[4],
        hc_stream_rank=32,
        hc_eps=1e-6,
        attn_res_rank=8,
        attn_res_block_size=2,
        attn_res_block_summary_rank=32,
        rosa_embed_dim=128,
        rosa_layer_ids=[],
        rosa_reset_token_id=None,
        moe_intermediate_size=32,
        erc_loss_alpha=0.5,
        erc_loss_coef=1.0,
        vocab_size=128,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        attention_dropout=0.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
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
    config = _transformer_config(num_layers=2, n_dense=1, mtp=0)
    assert shensi_hybrid_layer_pattern(config) == Symbols.ATTENTION * 2


def test_pattern_splits_pipeline_stages_and_writes_the_layout() -> None:
    """The stage split travels in ``pipeline_model_parallel_layout``, which the AttnRes plan reads."""
    config = _transformer_config(num_layers=4, mtp=1, pp=2)
    assert shensi_hybrid_layer_pattern(config) == "**|**" + Symbols.MTP_SEPARATOR + Symbols.ATTENTION
    assert config.pipeline_model_parallel_layout == [
        ["embedding", "decoder", "decoder"],
        ["decoder", "decoder", "mtp", "loss"],
    ]


def test_layer_plan_uses_the_global_layer_number() -> None:
    """PP stages after the first pass local layer numbers; the plan must stay global."""
    config = SimpleNamespace(
        num_layers=4,
        layer_types=["indexed_attention", "linear_attention", "indexed_attention", "linear_attention"],
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        attn_res_block_layer_types=[
            "block_write_layer",
            "block_read_layer",
            "block_write_layer",
            "block_read_layer",
        ],
    )
    assert shensi_layer_plan(config, 1, False) == (0, True, True, 0)
    assert shensi_layer_plan(config, 2, False) == (1, False, True, 0)
    assert shensi_layer_plan(config, 3, False) == (2, True, False, 1)
    assert shensi_layer_plan(config, 4, False) == (3, False, False, 1)
    # MTP mirrors the last decoder layer regardless of the local construction number: same layer
    # plan, including the attention-residual block index it routes over.
    assert shensi_layer_plan(config, 1, True) == (3, False, False, 1)


def test_stack_spec_hosts_the_shensi_stack_and_the_mtp_template() -> None:
    """The decoder is the bridge-side stack; the MTP inner layer is built from the pattern."""
    config = _transformer_config(num_layers=4, mtp=1)
    spec = get_shensi_hybrid_stack_spec(config, vp_stage=None, pp_rank=0)

    assert spec.module is ShensiDecoderStack
    assert isinstance(spec.submodules, HybridStackSubmodules)
    assert spec.submodules.attention_layer.module.__name__ == "ShensiTransformerLayer"

    mtp_layers = spec.submodules.mtp_block_spec.layer_specs
    assert len(mtp_layers) == 1
    assert mtp_layers[0].module.__name__ == "MultiTokenPredictionLayer"
    # The hybrid path builds the MTP inner model layer from the pattern, so the recipe must not pin one.
    assert mtp_layers[0].submodules.mtp_model_layer is None


def test_stack_spec_has_no_mtp_template_without_mtp_heads() -> None:
    config = _transformer_config(num_layers=2, n_dense=1, mtp=0)
    spec = get_shensi_hybrid_stack_spec(config, vp_stage=None, pp_rank=0)
    assert spec.submodules.mtp_block_spec is None
