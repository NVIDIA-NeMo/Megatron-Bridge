# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import Mock

import pytest
import torch
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.models.hybrid.hybrid_layer_allocation import select_pipeline_segment
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import Qwen3_5MoeTextConfig, Qwen3_5TextConfig

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.hybrid.gpt_checkpoint_compat import gpt_checkpoint_key, read_gpt_layout_keys
from megatron.bridge.models.qwen.modeling_qwen35.hybrid import qwen35_hybrid_mappings, qwen35_pipeline_pattern
from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
from megatron.bridge.models.qwen_vl.modeling_qwen35.hybrid_model import Qwen35VLHybridLanguageModel


@pytest.mark.parametrize("moe", [False, True])
def test_split_layer_mapping_preserves_hf_block_and_expert_indices(moe):
    templates = (
        Qwen35MoEBridge._get_moe_lm_mappings(experts_packed=False) if moe else Qwen35Bridge._get_dense_lm_mappings()
    )
    registry = MegatronMappingRegistry(*qwen35_hybrid_mappings(templates, num_layers=4))
    qkv = registry.megatron_to_hf_lookup("decoder.layers.6.self_attention.linear_qkv.weight")
    assert qkv.hf_param["q"] == "model.layers.3.self_attn.q_proj.weight"
    assert registry.hf_to_megatron_lookup(qkv.hf_param["q"]).megatron_param == qkv.megatron_param

    if moe:
        mlp_name = "decoder.layers.7.mlp.experts.linear_fc1.weight12"
        hf_gate = "model.layers.3.mlp.experts.12.gate_proj.weight"
    else:
        mlp_name = "decoder.layers.7.mlp.linear_fc1.weight"
        hf_gate = "model.layers.3.mlp.gate_proj.weight"
    mlp = registry.megatron_to_hf_lookup(mlp_name)
    assert mlp.hf_param["gate"] == hf_gate
    assert registry.hf_to_megatron_lookup(hf_gate).megatron_param == mlp_name
    assert registry.megatron_to_hf_lookup("decoder.final_norm.weight").hf_param == "model.norm.weight"


@pytest.mark.parametrize(
    "bridge_class,config_class", [(Qwen35Bridge, Qwen3_5TextConfig), (Qwen35MoEBridge, Qwen3_5MoeTextConfig)]
)
def test_hybrid_config_export_preserves_nonperiodic_attention_schedule(bridge_class, config_class):
    hf_config = config_class(
        num_hidden_layers=4, layer_types=["full_attention", "linear_attention", "linear_attention", "full_attention"]
    )
    bridge = bridge_class()
    config = bridge.hf_config_to_model_config(hf_config)
    restored = config_class(**bridge.megatron_to_hf_config(config))
    assert restored.num_hidden_layers == hf_config.num_hidden_layers
    assert restored.layer_types == hf_config.layer_types
    assert config.num_layers == 2 * hf_config.num_hidden_layers


def test_hybrid_mtp_mapping_preserves_zero_centered_gdn_conversion():
    mappings = qwen35_hybrid_mappings(Qwen35MoEBridge._get_moe_lm_mappings(), num_layers=4)
    registry = MegatronMappingRegistry(*mappings)
    norm = registry.megatron_to_hf_lookup("decoder.layers.4.self_attention.out_norm.weight")
    assert type(norm).__name__ == "RMSNorm2ZeroCenteredRMSNormMapping"
    assert norm.hf_param == "model.layers.2.linear_attn.norm.weight"

    registry = MegatronMappingRegistry(*qwen35_hybrid_mappings(Qwen35MoEBridge._get_moe_mtp_mappings(), num_layers=4))
    mapping = registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.layers.1.mlp.router.weight")
    assert mapping.hf_param == "mtp.layers.0.mlp.gate.weight"


def test_vl_mapping_splits_only_language_blocks():
    from transformers import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge, _get_vision_mappings

    bridge = Qwen35VLMoEBridge()
    bridge.hf_config = Qwen3_5MoeConfig(text_config={"num_hidden_layers": 4})
    registry = bridge.mapping_registry()
    language = registry.megatron_to_hf_lookup("language_model.decoder.layers.7.mlp.router.weight")
    assert language.hf_param == "model.language_model.layers.3.mlp.gate.weight"
    vision = MegatronMappingRegistry(*_get_vision_mappings())
    for template in vision.mappings:
        name = template.megatron_param.replace("*", "0")
        assert registry.megatron_to_hf_lookup(name).hf_param == vision.megatron_to_hf_lookup(name).hf_param


@pytest.mark.parametrize("moe", [False, True])
def test_gpt_checkpoint_keys_invert_the_block_split(moe):
    """A GPTModel checkpoint stays loadable only if every Hybrid key maps back to its GPT layer."""
    if moe:
        templates = Qwen35MoEBridge._get_moe_lm_mappings() + Qwen35MoEBridge._get_moe_mtp_mappings()
    else:
        templates = Qwen35Bridge._get_dense_lm_mappings() + Qwen35Bridge._get_dense_mtp_mappings()

    for template in templates:
        hybrid = qwen35_hybrid_mappings([template], num_layers=4)
        if "decoder.layers.*." in template.megatron_param:
            gpt = [template.megatron_param.replace("decoder.layers.*.", f"decoder.layers.{i}.") for i in range(4)]
        else:
            gpt = [template.megatron_param]
        # GPT MTP layers rename their inner layer to ``transformer_layer`` in checkpoints.
        gpt = [name.replace(".mtp_model_layer.", ".transformer_layer.") for name in gpt]
        assert [gpt_checkpoint_key(mapping.megatron_param) for mapping in hybrid] == gpt


@pytest.mark.parametrize(
    "pipeline",
    [
        dict(
            pipeline_model_parallel_size=2, num_layers_in_first_pipeline_stage=17, num_layers_in_last_pipeline_stage=23
        ),
        dict(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2),
        dict(
            pipeline_model_parallel_size=6,
            account_for_embedding_in_pipeline_split=True,
            account_for_loss_in_pipeline_split=True,
        ),
    ],
)
def test_pipeline_pattern_keeps_gpt_stage_assignment(pipeline):
    config = TransformerConfig(
        num_layers=40, hidden_size=64, num_attention_heads=4, pipeline_dtype=torch.bfloat16, **pipeline
    )
    pattern = "".join(("*" if (i + 1) % 4 == 0 else "G") + "E" for i in range(40))
    segments = qwen35_pipeline_pattern(config, pattern).split("|")
    assert "".join(segments) == pattern

    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size
    for index, segment in enumerate(segments):
        vp_stage, pp_rank = divmod(index, pp_size)
        offset = get_transformer_layer_offset(config, vp_stage if vp_size else None, pp_rank)
        assert sum(map(len, segments[:index])) == 2 * offset


@pytest.mark.parametrize("multimodal", [False, True])
def test_gpt_checkpoint_rewrite_preserves_other_decoders(multimodal):
    language = HybridModel.__new__(HybridModel)
    torch.nn.Module.__init__(language)
    language.hybrid_layer_pattern = "G-G-*-/-"
    model = torch.nn.Module() if multimodal else language
    prefix = "language_model." if multimodal else ""
    if multimodal:
        model.add_module("language_model", language)
        model.add_module("vision_model", torch.nn.Module())

    keys = [
        prefix + "decoder.layers.4.self_attention.linear_qkv.weight",
        prefix + "decoder.layers.5.mlp.linear_fc1.weight",
        prefix + "decoder.final_norm.weight",
        prefix + "mtp.layers.0.transformer_layer.layers.1.mlp.linear_fc1.weight",
    ]
    vision_keys = [f"vision_model.decoder.layers.{i}.self_attention.linear_qkv.weight" for i in (1, 2, 3)]
    if multimodal:
        keys += vision_keys
    entries = {key: ShardedTensor.from_rank_offsets(key, torch.ones(2)) for key in keys}
    extra_key = prefix + "output_layer._extra_state"
    entries[extra_key] = ShardedObject(extra_key, None, (1,), (0,))
    if multimodal:
        entries["vision_model.output_layer._extra_state"] = None
    state = {"model": entries}

    read_gpt_layout_keys(state, [model])

    for key in keys:
        expected = key if multimodal and key in vision_keys else gpt_checkpoint_key(key)
        assert entries[key].key == expected
    assert extra_key not in entries
    if multimodal:
        assert "vision_model.output_layer._extra_state" in entries
        assert len({entries[key].key for key in vision_keys}) == len(vision_keys)


@pytest.mark.parametrize("pre_process", [False, True])
def test_vl_tied_mtp_checkpoint_uses_embedding_on_last_pipeline_stage(pre_process):
    model = Qwen35VLHybridLanguageModel.__new__(Qwen35VLHybridLanguageModel)
    torch.nn.Module.__init__(model)
    model.mtp_process = True
    model.pre_process = pre_process
    model.post_process = True
    embedding_key = "language_model.embedding.word_embeddings.weight"
    output_key = "language_model.output_layer.weight"
    embedding = ShardedTensor.from_rank_offsets(embedding_key, torch.ones(2, 2))
    state = {
        embedding_key: embedding,
        output_key: ShardedTensor.from_rank_offsets(output_key, torch.ones(2, 2)),
    }

    model.tie_embeddings_and_output_weights_state_dict(state, output_key, embedding_key)

    assert state[embedding_key] is embedding
    assert (output_key in state) is pre_process


@pytest.mark.parametrize("config_class", [Qwen3_5TextConfig, Qwen3_5MoeTextConfig])
@pytest.mark.parametrize("explicit_pattern", [False, True])
def test_text_builder_assigns_all_virtual_pipeline_stages(monkeypatch, config_class, explicit_pattern):
    config = AutoBridge.from_hf_config(config_class(num_hidden_layers=8)).get_model_config()
    config.pipeline_model_parallel_size = 2
    config.virtual_pipeline_model_parallel_size = 2
    config.pipeline_dtype = torch.bfloat16
    config.mtp_num_layers = 1
    if explicit_pattern:
        pattern = config.hybrid_layer_pattern
        config.hybrid_layer_pattern = "|".join(pattern[i : i + 4] for i in range(0, len(pattern), 4))
    config.finalize()
    original_pattern = config.hybrid_layer_pattern
    builder = config.get_builder_cls()(config)
    constructor = Mock()
    monkeypatch.setattr("megatron.training.models.hybrid.HybridModel", constructor)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    group = Mock()
    group.size.return_value = 2
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    pg_collection = SimpleNamespace(pp=group)

    for vp_stage in range(2):
        for pp_rank in range(2):
            group.rank.return_value = pp_rank
            monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: pp_rank)
            builder.build_model(pg_collection, vp_stage=vp_stage)
            kwargs = constructor.call_args.kwargs
            main, mtp = kwargs["hybrid_layer_pattern"].split("/", 1)
            assert mtp == config.mtp_hybrid_override_pattern
            layers, offset = select_pipeline_segment(
                main, kwargs["config"], pg_collection.pp, vp_stage, tp_group=group, dp_cp_group=group
            )
            assert len(layers) == 4
            assert offset == 4 * (vp_stage * 2 + pp_rank)
            assert kwargs["pre_process"] is (vp_stage == 0 and pp_rank == 0)
            assert kwargs["post_process"] is (vp_stage == 1 and pp_rank == 1)
    assert config.hybrid_layer_pattern == original_pattern
