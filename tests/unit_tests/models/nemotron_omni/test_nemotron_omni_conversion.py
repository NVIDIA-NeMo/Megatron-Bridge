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
from unittest.mock import Mock

import torch
from torch import nn

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import get_model_bridge
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import NemotronOmniBridge
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider
from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel


class _DictConfig(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


def _mapping_names(registry: MegatronMappingRegistry) -> list[str]:
    names = []
    for mapping in registry.mappings:
        megatron_param = getattr(mapping, "megatron_param", None)
        if megatron_param is not None:
            names.append(str(megatron_param))
        hf_param = getattr(mapping, "hf_param", None)
        if isinstance(hf_param, dict):
            names.extend(str(v) for v in hf_param.values())
        elif hf_param is not None:
            names.append(str(hf_param))
    return names


def _mock_omni_hf_config():
    llm_config = _DictConfig(
        torch_dtype="bfloat16",
        hidden_act="silu",
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        initializer_range=0.02,
        layer_norm_epsilon=1e-6,
        vocab_size=131072,
        max_position_embeddings=4096,
        hybrid_override_pattern="MEME",
        mamba_head_dim=64,
        mamba_num_heads=4,
        n_groups=2,
        ssm_state_size=16,
        residual_in_fp32=False,
        moe_intermediate_size=384,
        moe_shared_expert_intermediate_size=768,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        rope_theta=10000.0,
    )
    sound_config = _DictConfig(
        model_type="parakeet",
        hidden_size=128,
        projection_hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        subsampling_factor=8,
        num_mel_bins=128,
        conv_kernel_size=9,
        convolution_bias=False,
    )
    vision_config = _DictConfig(
        separate_video_embedder=True,
        video_temporal_patch_size=2,
    )
    return _DictConfig(
        architectures=["NemotronH_Nano_Omni_Reasoning_V3"],
        auto_map={"AutoModelForCausalLM": "modeling.NemotronH_Nano_Omni_Reasoning_V3"},
        llm_config=llm_config,
        sound_config=sound_config,
        vision_config=vision_config,
        projector_hidden_size=1024,
        img_context_token_id=18,
        sound_context_token_id=27,
    )


def test_public_nemotron_omni_architecture_is_registered():
    hf_config = _mock_omni_hf_config()

    assert AutoBridge.supports(hf_config)
    assert isinstance(get_model_bridge("NemotronH_Nano_Omni_Reasoning_V3", hf_config=hf_config), NemotronOmniBridge)


def test_nemotron_omni_provider_bridge_maps_public_config_fields():
    hf_config = _mock_omni_hf_config()
    hf_pretrained = Mock(spec=PreTrainedVLM)
    hf_pretrained.config = hf_config

    provider = NemotronOmniBridge().provider_bridge(hf_pretrained)

    assert isinstance(provider, NemotronOmniModelProvider)
    assert provider.has_sound is True
    assert provider.language_model_type == "nemotron6-moe"
    assert provider.hidden_size == 256
    assert provider.ffn_hidden_size == 512
    assert provider.num_attention_heads == 8
    assert provider.num_query_groups == 2
    assert provider.kv_channels == 32
    assert provider.layernorm_epsilon == 1e-6
    assert provider.num_moe_experts == 8
    assert provider.moe_router_topk == 2
    assert provider.moe_ffn_hidden_size == 384
    assert provider.moe_shared_expert_intermediate_size == 768
    assert provider.vision_proj_ffn_hidden_size == 1024
    assert provider.image_token_index == 18
    assert provider.sound_context_token_id == 27
    assert provider.sound_hidden_size == 128
    assert provider.sound_projection_hidden_size == 256
    assert provider.sound_config["num_mel_bins"] == 128
    assert provider.dynamic_resolution is True
    assert provider.separate_video_embedder is True
    assert provider.temporal_patch_dim == 2
    assert provider.temporal_ckpt_compat is True


def test_nemotron_omni_mapping_registry_includes_sound_mappings():
    registry = NemotronOmniBridge().mapping_registry()
    names = _mapping_names(registry)

    assert any("llava_model.sound_projection" in name for name in names)
    assert any("sound_projection.linear1.weight" in name for name in names)
    assert any("llava_model.sound_model.encoder.**" in name for name in names)
    assert any("sound_encoder.encoder.**" in name for name in names)


def test_nemotron_omni_encode_batch_preserves_packed_sequence_metadata():
    from megatron.bridge.data.energon.metadata import batch_metadata_kwargs
    from megatron.bridge.data.energon.nemotron_omni_task_encoder import (
        NemotronOmniTaskBatch,
        NemotronOmniTaskEncoder,
    )

    tokens = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    position_ids = torch.tensor([[0, 1, 2]])
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_seqlens_argmin = torch.tensor(3, dtype=torch.int32)
    max_seqlen = torch.tensor(2, dtype=torch.int32)
    pixel_values = torch.ones(1, 4, 8)

    batch = NemotronOmniTaskBatch(
        **batch_metadata_kwargs(keys=["sample"]),
        input_ids=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=None,
        position_ids=position_ids,
        visual_tensors={"pixel_values": pixel_values},
        cu_seqlens=cu_seqlens,
        cu_seqlens_unpadded=cu_seqlens.clone(),
        cu_seqlens_argmin=cu_seqlens_argmin,
        max_seqlen=max_seqlen,
    )

    raw = NemotronOmniTaskEncoder.__new__(NemotronOmniTaskEncoder).encode_batch(batch)

    assert raw["tokens"] is tokens
    assert raw["cu_seqlens"] is cu_seqlens
    assert torch.equal(raw["cu_seqlens_unpadded"], cu_seqlens)
    assert raw["cu_seqlens_argmin"] is cu_seqlens_argmin
    assert raw["max_seqlen"] is max_seqlen
    assert torch.equal(raw["visual_inputs"].pixel_values, pixel_values)


def test_nemotron_omni_freeze_sound_modules_without_stdout(monkeypatch, capsys):
    monkeypatch.setattr(NemotronVLModel, "freeze", lambda self, **_: None)

    model = NemotronOmniModel.__new__(NemotronOmniModel)
    model.llava_model = SimpleNamespace(
        sound_model=nn.Linear(4, 4),
        sound_projection=nn.Linear(4, 4),
    )

    model.freeze(freeze_sound_model=True, freeze_sound_projection=True)

    assert all(not param.requires_grad for param in model.llava_model.sound_model.parameters())
    assert all(not param.requires_grad for param in model.llava_model.sound_projection.parameters())
    assert capsys.readouterr().out == ""
