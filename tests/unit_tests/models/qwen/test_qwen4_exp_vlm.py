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

"""VLM/text-only conversion and per-document QSA multimodal rotary contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from safetensors.torch import save_file
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpModelConfig
from megatron.bridge.models.qwen.modeling_qwen4_exp.qsa import QSAIndexer
from megatron.bridge.models.qwen.modeling_qwen4_exp.vl_model import Qwen4ExpVLModel, Qwen4ExpVLModelConfig
from megatron.bridge.models.qwen.qwen4_exp_bridge import PLENGramEmbeddingMapping, Qwen4ExpBridge, Qwen4ExpTextBridge
from megatron.bridge.training.vlm_step import _filter_visual_kwargs_for_model
from tests.unit_tests.models.qwen.test_qwen4_exp_bridge import _text_config_dict


pytestmark = pytest.mark.unit


@pytest.fixture
def full_hf_config():
    text = PretrainedConfig(**_text_config_dict())
    text.model_type = "qwen4_exp_text"
    return PretrainedConfig(
        architectures=["Qwen4ExpForConditionalGeneration"],
        text_config=text,
        vision_config=PretrainedConfig(
            depth=2,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            out_hidden_size=text.hidden_size,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_channels=3,
            num_position_embeddings=16,
            hidden_act="gelu_pytorch_tanh",
            deepstack_visual_indexes=[],
        ),
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
    )


@pytest.mark.parametrize("top_level_tied", [False, True])
def test_full_vlm_uses_parent_tie_setting_and_multimodal_config(full_hf_config, top_level_tied):
    full_hf_config.tie_word_embeddings = top_level_tied
    full_hf_config.text_config.tie_word_embeddings = not top_level_tied
    config = Qwen4ExpBridge().hf_config_to_model_config(full_hf_config)

    assert isinstance(config, Qwen4ExpVLModelConfig)
    assert not config.hf_model_text_only
    assert config.share_embeddings_and_output_weights is top_level_tied
    assert config.vision_config["depth"] == 2
    assert config.mrope_section == [11, 11, 10]
    config.finalize()
    assert config.qwen4_mrope
    assert not config.apply_rope_fusion
    assert not config.scatter_embedding_sequence_parallel
    restored = Qwen4ExpVLModelConfig.from_dict(config.as_dict())
    assert type(restored) is Qwen4ExpVLModelConfig
    assert restored.vision_config == config.vision_config
    assert not restored.hf_model_text_only
    assert restored.get_builder_cls() is config.get_builder_cls()


def test_native_text_checkpoint_has_no_vision_construction(full_hf_config):
    config = Qwen4ExpTextBridge().hf_config_to_model_config(full_hf_config.text_config)
    assert type(config) is Qwen4ExpModelConfig
    assert config.hf_model_text_only
    assert Qwen4ExpModelConfig.from_dict(config.as_dict()).hf_model_text_only
    assert not hasattr(config, "vision_config")
    assert config.mrope_section is None
    config.finalize()
    assert not config.qwen4_mrope


def test_text_projection_is_lazy_filters_vision_and_preserves_export_namespace(full_hf_config, tmp_path, monkeypatch):
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(12.0).reshape(3, 4),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.arange(4.0),
        "lm_head.weight": torch.arange(12.0).reshape(3, 4) + 20,
        "model.visual.patch_embed.proj.weight": torch.ones(2, 2),
        "mtp.layers.0.weight": torch.ones(2, 2),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    pretrained = PreTrainedCausalLM(tmp_path, device="cpu", torch_dtype=torch.float32)
    pretrained.config = full_hf_config
    full_hf_config.tie_word_embeddings = True
    full_hf_config.text_config.tie_word_embeddings = False
    full_hf_config.text_config.auto_map = {"AutoModel": "custom.MultimodalModel"}
    loaded = []
    original_load = SafeTensorsStateSource.load_tensors

    def record_load(source, keys):
        loaded.extend(keys)
        return original_load(source, keys)

    monkeypatch.setattr(SafeTensorsStateSource, "load_tensors", record_load)
    projected = Qwen4ExpBridge().text_only_pretrained(pretrained)
    assert loaded == []
    assert projected.config.architectures == ["Qwen4ExpForCausalLM"]
    assert projected.config.tie_word_embeddings
    assert not hasattr(projected.config, "auto_map")
    assert projected._text_only
    projected_config = Qwen4ExpTextBridge().hf_config_to_model_config(projected.config)
    assert projected_config.hf_model_text_only
    assert Qwen4ExpModelConfig.from_dict(projected_config.as_dict()).hf_model_text_only
    source = projected.state.source
    assert set(source.get_all_keys()) == {
        "model.embed_tokens.weight",
        "model.hyper_connection_mixer.hc_norm.weight",
        "lm_head.weight",
    }
    assert loaded == []
    weights = source.load_tensors(["model.embed_tokens.weight", "lm_head.weight"])
    assert loaded == ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    torch.testing.assert_close(weights["model.embed_tokens.weight"], tensors[loaded[0]])
    torch.testing.assert_close(weights["lm_head.weight"], tensors["lm_head.weight"])
    assert hasattr(full_hf_config.text_config, "auto_map")
    assert full_hf_config.text_config.tie_word_embeddings is False

    text_bridge = Qwen4ExpTextBridge()
    text_bridge.hf_pretrained = projected
    text_mapping = text_bridge.mapping_registry()
    assert (
        text_mapping.megatron_to_hf_lookup("embedding.word_embeddings.weight").hf_param == "model.embed_tokens.weight"
    )
    assert text_mapping.megatron_to_hf_lookup("output_layer.weight").hf_param == "lm_head.weight"
    assert text_mapping.megatron_to_hf_lookup("vision_model.pos_embed.weight") is None
    full_bridge = Qwen4ExpBridge()
    full_bridge.hf_pretrained = pretrained
    full_mapping = full_bridge.mapping_registry()
    assert full_mapping.megatron_to_hf_lookup("language_model.embedding.word_embeddings.weight").hf_param == (
        "model.language_model.embed_tokens.weight"
    )
    assert (
        full_mapping.megatron_to_hf_lookup("vision_model.pos_embed.weight").hf_param == "model.visual.pos_embed.weight"
    )


def test_text_projection_rejects_subfolder(full_hf_config):
    pretrained = PreTrainedCausalLM("unused-model", device="cpu", subfolder="weights")
    pretrained.config = full_hf_config
    with pytest.raises(ValueError, match="subfolder"):
        Qwen4ExpBridge().text_only_pretrained(pretrained)


@pytest.mark.parametrize("packed", [False, True])
def test_qsa_mrope_uses_each_documents_actual_block_start_frequencies(packed):
    # Exercise the actual pooling and rotary operations without allocating TE projections.
    indexer = QSAIndexer.__new__(QSAIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.config = SimpleNamespace(
        sequence_parallel=False,
        qwen4_mrope=True,
        apply_rope_fusion=False,
        apply_rotary_pos_emb_in_fp32=False,
        rotary_interleaved=False,
    )
    indexer.n_heads = indexer.kv_heads = 1
    indexer.head_dim = 4
    indexer.compress_ratio = 2
    indexer.rotary_interleaved = False
    indexer.q_layernorm = torch.nn.Identity()
    indexer.k_layernorm = torch.nn.Identity()
    indexer.index_qk_proj = Mock(side_effect=lambda hidden: (torch.cat((hidden, hidden), dim=-1), None))
    indexer._select_blocks = Mock(side_effect=lambda q, *args: (torch.zeros(q.shape[0], 1, dtype=torch.uint8), True))
    # Two documents with identical hidden values and different multimodal positions.
    angles = torch.tensor([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]])
    seq, batch = (8, 1) if packed else (4, 2)
    hidden = torch.zeros(seq, batch, 4)
    hidden[..., 0] = 1.0
    token_angles = angles.reshape(seq, batch) if packed else angles.transpose(0, 1)
    frequencies = token_angles[:, :, None, None].expand(seq, batch, 1, 4).contiguous()
    packed_params = None
    if packed:
        boundaries = torch.tensor([0, 4, 8], dtype=torch.int32)
        packed_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=boundaries,
            cu_seqlens_kv=boundaries,
            max_seqlen_q=4,
            max_seqlen_kv=4,
        )

    indexer.forward(hidden, frequencies, packed_params)

    queries, pooled, valid, _, _ = indexer._select_blocks.call_args.args
    expected_queries = torch.zeros(8, 1, 4)
    expected_queries[:, 0, 0] = angles.flatten().cos()
    expected_queries[:, 0, 2] = angles.flatten().sin()
    torch.testing.assert_close(queries, expected_queries)
    expected_pooled = torch.zeros(2, 2, 4)
    block_angles = torch.tensor([[0.0, 0.2], [1.0, 1.2]])
    expected_pooled[:, :, 0] = block_angles.cos()
    expected_pooled[:, :, 2] = block_angles.sin()
    torch.testing.assert_close(pooled, expected_pooled)
    assert valid.all()


@pytest.mark.parametrize("sharded", [False, True])
def test_actual_checkpoint_keys_select_unsplit_or_sharded_ple_mapping(full_hf_config, sharded):
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    keys = [prefix + f"shard_{i}.weight" for i in range(2)] if sharded else [prefix + "weight"]
    source = Mock()
    source.get_all_keys.return_value = keys
    bridge = Qwen4ExpBridge()
    bridge.hf_pretrained = SimpleNamespace(config=full_hf_config, state=SimpleNamespace(source=source))

    assert bridge._num_ple_shards(full_hf_config.text_config) == (2 if sharded else 0)
    mapping = bridge.mapping_registry().megatron_to_hf_lookup(
        "language_model.decoder.layers.1.per_layer_embedding.ple_embedding.ngram_embedding.weight"
    )
    if sharded:
        assert isinstance(mapping, PLENGramEmbeddingMapping)
        assert mapping.num_shards == 2
        assert mapping.hf_param == {f"shard_{i}": prefix + f"shard_{i}.weight" for i in range(2)}
    else:
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == prefix + "weight"


def test_vlm_step_filters_unconsumed_multimodal_token_types():
    model = Qwen4ExpVLModel.__new__(Qwen4ExpVLModel)
    torch.nn.Module.__init__(model)
    pixels = torch.ones(1, 8)
    grid = torch.tensor([[1, 2, 2]])
    kwargs = _filter_visual_kwargs_for_model(
        model,
        {"pixel_values": pixels, "image_grid_thw": grid, "mm_token_type_ids": torch.zeros(1, 4)},
    )
    assert set(kwargs) == {"pixel_values", "image_grid_thw"}
    assert kwargs["pixel_values"] is pixels
    assert kwargs["image_grid_thw"] is grid


@pytest.mark.parametrize("full_vlm", [False, True])
def test_finalization_mirrors_authoritative_tie_setting_to_transformer(full_hf_config, full_vlm):
    bridge = Qwen4ExpBridge() if full_vlm else Qwen4ExpTextBridge()
    hf_config = full_hf_config if full_vlm else full_hf_config.text_config
    config = bridge.hf_config_to_model_config(hf_config)
    config.share_embeddings_and_output_weights = True
    config.transformer.share_embeddings_and_output_weights = False
    config.finalize()
    assert config.transformer.share_embeddings_and_output_weights
    assert bridge._share_embeddings_and_output_weights(config.transformer)

    config.share_embeddings_and_output_weights = False
    config.finalize()
    assert not config.transformer.share_embeddings_and_output_weights
    assert not bridge._share_embeddings_and_output_weights(config.transformer)
