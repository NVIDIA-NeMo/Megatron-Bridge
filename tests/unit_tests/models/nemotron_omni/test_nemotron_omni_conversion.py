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
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from megatron.core.activations import squared_relu
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, get_model_bridge
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotron_omni import nemotron_omni_provider as provider_module
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni_llava import NemotronOmniLlavaModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_bridge import (
    NemotronOmniBridge,
    NemotronOmniLlavaBridge,
)
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    NEMOTRON_OMNI_LLAVA_CONTRACT,
    NemotronOmniLlavaModelProvider,
    NemotronOmniModelProvider,
)
from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel
from megatron.bridge.models.nemotron_vl.nemotron_vl_bridge import NemotronVLBridge
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronVLModelProvider
from megatron.bridge.training.config import ConfigContainer


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
        moe_latent_size=128,
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


def _mock_legacy_v2_omni_hf_config():
    """Represent Nano Omni weights exported before the V3 architecture name."""

    hf_config = _mock_omni_hf_config()
    hf_config.architectures = ["NemotronH_Nano_VL_V2"]
    hf_config.model_type = "NemotronH_Nano_VL_V2"
    del hf_config.sound_config
    del hf_config.sound_context_token_id
    hf_config.vision_config.args = {"register_multiple": 10}
    return hf_config


def test_public_nemotron_omni_architecture_is_registered():
    hf_config = _mock_omni_hf_config()

    assert AutoBridge.supports(hf_config)
    assert isinstance(get_model_bridge("NemotronH_Nano_Omni_Reasoning_V3", hf_config=hf_config), NemotronOmniBridge)

    hf_config.architectures = ["NemotronH_Super_Omni_Reasoning_V3"]
    assert AutoBridge.supports(hf_config)
    assert isinstance(get_model_bridge("NemotronH_Super_Omni_Reasoning_V3", hf_config=hf_config), NemotronOmniBridge)


def test_legacy_v2_moe_checkpoint_routes_to_canonical_nemotron_omni():
    hf_config = _mock_legacy_v2_omni_hf_config()
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = hf_config

    bridge = get_model_bridge("NemotronH_Nano_VL_V2", hf_config=hf_config)
    provider = bridge.provider_bridge(hf_pretrained)
    registry = bridge.mapping_registry()

    assert isinstance(bridge, NemotronVLBridge)
    assert isinstance(provider, NemotronOmniModelProvider)
    assert provider.nemotron_omni_contract == NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT
    assert provider.image_token_index == 18
    assert provider.img_start_token_id == 19
    assert provider.img_end_token_id == 20
    assert provider.has_sound is False
    assert provider.separate_video_embedder is True
    assert provider.temporal_patch_dim == 2
    assert provider.temporal_ckpt_compat is True
    video_mapping = registry.hf_to_megatron_lookup(
        "vision_model.radio_model.model.patch_generator.video_embedder.weight"
    )
    assert video_mapping.megatron_param == "vision_model.video_embedder.weight"
    assert all(not mapping.megatron_param.startswith("llava_model.") for mapping in registry.mappings)


def test_dense_legacy_v2_checkpoint_keeps_nemotron_vl_path():
    hf_config = _mock_legacy_v2_omni_hf_config()
    del hf_config.llm_config.n_routed_experts
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = hf_config

    bridge = get_model_bridge("NemotronH_Nano_VL_V2", hf_config=hf_config)
    provider = bridge.provider_bridge(hf_pretrained)
    registry = bridge.mapping_registry()

    assert isinstance(provider, NemotronVLModelProvider)
    assert any(mapping.megatron_param.startswith("llava_model.") for mapping in registry.mappings)


def test_nemotron_omni_provider_bridge_maps_public_config_fields():
    hf_config = _mock_omni_hf_config()
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = hf_config

    provider = NemotronOmniBridge().provider_bridge(hf_pretrained)

    assert isinstance(provider, NemotronOmniModelProvider)
    assert provider.nemotron_omni_contract == NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT
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
    assert provider.radio_interpolate_only_cpe is False
    assert provider.separate_video_embedder is True
    assert provider.temporal_patch_dim == 2
    assert provider.temporal_ckpt_compat is True
    serialized = ConfigContainer._convert_value_to_dict(provider)
    assert serialized["nemotron_omni_contract"] == NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT
    assert serialized["has_sound"] is True
    assert "add_sound_encoder" not in serialized


def test_nemotron_omni_provider_bridge_omits_sound_when_config_is_absent():
    hf_config = _mock_omni_hf_config()
    del hf_config.sound_config
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = hf_config

    provider = NemotronOmniBridge().provider_bridge(hf_pretrained)

    assert provider.has_sound is False
    assert provider.sound_config is None
    assert provider.sound_context_token_id == 0


def test_nemotron_omni_hf_config_export_preserves_sound_capability():
    provider = NemotronOmniModelProvider(
        has_sound=True,
        sound_context_token_id=27,
        sound_config={"hidden_size": 128},
    )

    hf_config = NemotronOmniBridge.megatron_to_hf_config(provider)

    assert hf_config["sound_config"] == {"hidden_size": 128}
    assert hf_config["sound_context_token_id"] == 27


def test_nemotron_omni_hf_config_export_omits_disabled_sound_capability():
    provider = NemotronOmniModelProvider(
        has_sound=False,
        sound_context_token_id=27,
        sound_config={"hidden_size": 128},
    )

    hf_config = NemotronOmniBridge.megatron_to_hf_config(provider)

    assert hf_config["sound_config"] is None
    assert hf_config["sound_context_token_id"] is None


def test_nemotron_omni_provider_rejects_static_resolution():
    provider = NemotronOmniModelProvider()
    provider.dynamic_resolution = False

    with pytest.raises(ValueError, match="only supports dynamic_resolution=True"):
        provider.finalize()


@pytest.mark.parametrize("image_token_index", [0, -1])
def test_nemotron_omni_provider_rejects_nonpositive_image_token_index(image_token_index):
    provider = NemotronOmniModelProvider(image_token_index=image_token_index)

    with pytest.raises(ValueError, match="requires a positive image_token_index"):
        provider.finalize()


def test_nemotron_omni_provider_rejects_nonpositive_sound_token_index():
    provider = NemotronOmniModelProvider(
        image_token_index=18,
        has_sound=True,
        sound_context_token_id=0,
        sound_config={},
    )

    with pytest.raises(ValueError, match="requires a positive sound_context_token_id"):
        provider.finalize()


def test_nemotron_omni_provider_requires_sound_config_when_enabled():
    provider = NemotronOmniModelProvider(image_token_index=18, has_sound=True, sound_context_token_id=27)

    with pytest.raises(ValueError, match="requires sound_config"):
        provider.finalize()


def test_canonical_provider_builds_dedicated_model(monkeypatch):
    provider = NemotronOmniModelProvider(
        image_token_index=18,
        nemotron_omni_contract=NEMOTRON_OMNI_EXPANDED_SEQUENCE_CONTRACT,
    )
    model = SimpleNamespace()
    model_factory = Mock(return_value=model)
    llava_factory = Mock()

    monkeypatch.setattr(provider_module, "LLaVAModel", llava_factory)
    monkeypatch.setattr(provider_module, "get_vit_layer_with_transformer_engine_spec", Mock(return_value=object()))
    monkeypatch.setattr(provider_module, "get_language_mlp_submodules", Mock(return_value=object()))
    monkeypatch.setattr(provider_module, "NemotronOmniModel", model_factory)

    assert provider.provide() is model
    model_factory.assert_called_once()
    llava_factory.assert_not_called()


def test_nemotron_omni_provider_can_omit_sound_modules():
    provider = NemotronOmniModelProvider(has_sound=False)

    sound_model, sound_projection = provider._build_sound_modules(None, None, add_encoder=True)

    assert provider.has_sound is False
    assert sound_model is None
    assert sound_projection is None


def test_nemotron_omni_provider_builds_sound_modules_when_enabled(monkeypatch):
    provider = NemotronOmniModelProvider(has_sound=True)
    expected_sound_model = object()
    expected_sound_projection = object()
    monkeypatch.setattr(provider, "_build_sound_encoder", lambda: expected_sound_model)
    monkeypatch.setattr(provider, "_build_sound_projection_config", lambda _: object())
    monkeypatch.setattr(provider_module, "get_language_mlp_submodules", lambda _: object())
    monkeypatch.setattr(provider_module, "MultimodalProjector", lambda **_: expected_sound_projection)

    sound_model, sound_projection = provider._build_sound_modules(None, None, add_encoder=True)

    assert sound_model is expected_sound_model
    assert sound_projection is expected_sound_projection


def test_nemotron_omni_vision_projection_uses_squared_relu():
    provider = NemotronOmniModelProvider()

    vision_projection_config = provider._build_vision_projection_config(provider)
    values = torch.tensor([-2.0, 0.0, 3.0])

    assert vision_projection_config.activation_func is squared_relu
    assert torch.equal(vision_projection_config.activation_func(values), torch.tensor([0.0, 0.0, 9.0]))


def test_nemotron_omni_providers_use_contract_specific_cpe_defaults():
    assert NemotronOmniModelProvider().radio_interpolate_only_cpe is False
    assert NemotronOmniLlavaModelProvider().radio_interpolate_only_cpe is True


def test_nemotron_omni_mapping_registry_includes_sound_mappings():
    registry = NemotronOmniBridge().mapping_registry()
    names = _mapping_names(registry)

    assert any("sound_projection" in name for name in names)
    assert any("sound_projection.linear1.weight" in name for name in names)
    assert any("sound_model.encoder.**" in name for name in names)
    assert any("sound_encoder.encoder.**" in name for name in names)
    assert all(not name.startswith("llava_model.") for name in names)


def test_nemotron_omni_export_preserves_source_only_buffers():
    bridge = NemotronOmniBridge()
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    source_tensors = {
        name: torch.full((2,), index, dtype=torch.float32) for index, name in enumerate(bridge._HF_PASSTHROUGH_KEYS)
    }
    hf_pretrained.state = MagicMock()
    hf_pretrained.state.source.get_all_keys.return_value = [
        "language_model.weight",
        *source_tensors,
    ]
    hf_pretrained.state.__getitem__ = Mock(side_effect=source_tensors.__getitem__)
    converted = HFWeightTuple("language_model.weight", torch.ones(1))

    with patch.object(NemotronVLBridge, "stream_weights_megatron_to_hf", return_value=iter([converted])):
        exported = list(bridge.stream_weights_megatron_to_hf([], hf_pretrained))

    assert exported[0] == converted
    exported_buffers = {item.param_name: item.weight for item in exported[1:]}
    assert exported_buffers.keys() == source_tensors.keys()
    for name, source_tensor in source_tensors.items():
        assert torch.equal(exported_buffers[name], source_tensor)


def test_nemotron_omni_export_exposes_transitive_dynamic_modules(tmp_path):
    modeling_path = tmp_path / "modeling.py"
    modeling_path.write_text("from .configuration import NemotronOmniConfig\n")
    bridge = NemotronOmniBridge()

    bridge.postprocess_hf_export_artifacts(tmp_path)
    bridge.postprocess_hf_export_artifacts(tmp_path)

    modeling_source = modeling_path.read_text()
    assert modeling_source.count("from .configuration_nemotron_h import NemotronHConfig") == 1
    assert modeling_source.count("from .configuration_radio import RADIOConfig") == 1


def test_nemotron_omni_export_requires_modeling_entrypoint(tmp_path):
    bridge = NemotronOmniBridge()

    with pytest.raises(FileNotFoundError, match="missing required artifact.*modeling.py"):
        bridge.postprocess_hf_export_artifacts(tmp_path)


def test_nemotron_omni_config_only_export_preserves_source_only_buffers(tmp_path):
    bridge = NemotronOmniBridge()
    source_tensors = {
        name: torch.full((2,), index, dtype=torch.float32) for index, name in enumerate(bridge._HF_PASSTHROUGH_KEYS)
    }
    save_file(source_tensors, tmp_path / "model.safetensors")
    hf_config = PretrainedConfig()
    hf_config.name_or_path = str(tmp_path)
    converted = HFWeightTuple("language_model.weight", torch.ones(1))

    with patch.object(NemotronVLBridge, "stream_weights_megatron_to_hf", return_value=iter([converted])):
        exported = list(bridge.stream_weights_megatron_to_hf([], hf_config))

    exported_buffers = {item.param_name: item.weight for item in exported[1:]}
    assert exported_buffers.keys() == source_tensors.keys()
    for name, source_tensor in source_tensors.items():
        assert torch.equal(exported_buffers[name], source_tensor)


def test_canonical_bridge_maps_super_mtp_config():
    hf_config = _mock_omni_hf_config()
    hf_config.architectures = ["NemotronH_Super_Omni_Reasoning_V3"]
    hf_config.llm_config.mtp_hybrid_override_pattern = "*E"
    hf_config.llm_config.num_nextn_predict_layers = 1
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = hf_config

    provider = NemotronOmniBridge().provider_bridge(hf_pretrained)

    assert isinstance(provider, NemotronOmniModelProvider)
    assert provider.hybrid_layer_pattern == "MEME"
    assert provider.moe_latent_size == 128
    assert provider.mtp_hybrid_override_pattern == "*E"
    assert provider.mtp_num_layers == 1
    assert isinstance(
        get_model_bridge("NemotronH_Super_Omni_Reasoning_V3", hf_config=hf_config),
        NemotronOmniBridge,
    )


def test_canonical_mapping_registry_uses_top_level_model_names():
    bridge = NemotronOmniBridge()
    bridge.hf_config = _mock_omni_hf_config()
    bridge.hf_config.llm_config.mtp_hybrid_override_pattern = "*E"
    bridge.hf_config.llm_config.num_nextn_predict_layers = 1
    registry = bridge.mapping_registry()

    embedding = registry.megatron_to_hf_lookup("language_model.embedding.word_embeddings.weight")
    vision = registry.megatron_to_hf_lookup("vision_model.embedder.weight")
    projector = registry.megatron_to_hf_lookup("vision_projection.encoder.linear_fc1.weight")
    mtp_projection = registry.megatron_to_hf_lookup("language_model.mtp.layers.0.eh_proj.weight")
    mtp_expert = registry.megatron_to_hf_lookup(
        "language_model.mtp.layers.0.mtp_model_layer.layers.1.mlp.experts.linear_fc1.weight3"
    )
    reverse_mtp_qkv = registry.hf_to_megatron_lookup("mtp.layers.1.mixer.q_proj.weight")

    assert embedding.hf_param == "language_model.backbone.embeddings.weight"
    assert vision.hf_param == ("vision_model.radio_model.model.patch_generator.embedder.weight")
    assert projector.hf_param == "mlp1.1.weight"
    assert mtp_projection.hf_param == "mtp.layers.0.eh_proj.weight"
    assert mtp_expert.hf_param == "mtp.layers.1.mixer.experts.3.up_proj.weight"
    assert (
        reverse_mtp_qkv.megatron_param
        == "language_model.mtp.layers.0.mtp_model_layer.layers.1.self_attention.linear_qkv.weight"
    )
    assert all(not mapping.megatron_param.startswith("llava_model.") for mapping in registry.mappings)


def test_llava_bridge_retains_legacy_wrapper_namespace():
    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = _mock_omni_hf_config()

    with pytest.warns(FutureWarning, match="NemotronOmniLlavaBridge is deprecated"):
        provider = NemotronOmniLlavaBridge().provider_bridge(hf_pretrained)
    registry = NemotronOmniLlavaBridge().mapping_registry()

    assert isinstance(provider, NemotronOmniLlavaModelProvider)
    assert provider.nemotron_omni_contract == NEMOTRON_OMNI_LLAVA_CONTRACT
    serialized = ConfigContainer._convert_value_to_dict(provider)
    assert serialized["nemotron_omni_contract"] == NEMOTRON_OMNI_LLAVA_CONTRACT
    assert any(mapping.megatron_param.startswith("llava_model.") for mapping in registry.mappings)


def test_nemotron_omni_encode_batch_preserves_packed_sequence_metadata():
    from megatron.bridge.data.energon.metadata import batch_metadata_kwargs
    from megatron.bridge.data.energon.nemotron_omni_task_encoder import (
        NemotronOmniTaskBatch,
        NemotronOmniTaskEncoder,
    )
    from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs

    tokens = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, -100]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    position_ids = torch.tensor([[0, 1, 2]])
    cu_seqlens_q = torch.tensor([0, 1, 3], dtype=torch.int32)
    max_seqlen_q = torch.tensor(2, dtype=torch.int32)
    pixel_values = torch.ones(1, 4, 8)

    batch = NemotronOmniTaskBatch(
        **batch_metadata_kwargs(keys=["sample"]),
        input_ids=tokens,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=None,
        position_ids=position_ids,
        visual_inputs=GenericVisualInputs(pixel_values=pixel_values),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_q,
    )

    raw = NemotronOmniTaskEncoder.__new__(NemotronOmniTaskEncoder).encode_batch(batch)

    assert raw["input_ids"] is tokens
    assert raw["tokens"] is tokens
    assert raw["cu_seqlens_q"] is cu_seqlens_q
    assert raw["cu_seqlens_kv"] is cu_seqlens_q
    assert raw["max_seqlen_q"] is max_seqlen_q
    assert raw["max_seqlen_kv"] is max_seqlen_q
    assert "cu_seqlens" not in raw
    assert "cu_seqlens_unpadded" not in raw
    assert "cu_seqlens_argmin" not in raw
    assert torch.equal(raw["visual_inputs"].pixel_values, pixel_values)


def test_nemotron_omni_freeze_sound_modules_without_stdout(monkeypatch, capsys):
    monkeypatch.setattr(NemotronVLModel, "freeze", lambda self, **_: None)

    model = NemotronOmniLlavaModel.__new__(NemotronOmniLlavaModel)
    model.llava_model = SimpleNamespace(
        sound_model=nn.Linear(4, 4),
        sound_projection=nn.Linear(4, 4),
    )

    model.freeze(freeze_sound_model=True, freeze_sound_projection=True)

    assert all(not param.requires_grad for param in model.llava_model.sound_model.parameters())
    assert all(not param.requires_grad for param in model.llava_model.sound_projection.parameters())
    assert capsys.readouterr().out == ""


def test_nemotron_omni_freeze_skips_modules_absent_from_pipeline_stage():
    model = NemotronOmniModel.__new__(NemotronOmniModel)
    nn.Module.__init__(model)
    model.language_model = nn.Linear(4, 4)
    model.vision_model = None
    model.vision_projection = None
    model.sound_model = None
    model.sound_projection = None

    model.freeze(
        freeze_language_model=True,
        freeze_vision_model=True,
        freeze_vision_projection=True,
        freeze_sound_model=True,
        freeze_sound_projection=True,
    )

    assert all(not param.requires_grad for param in model.language_model.parameters())
