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

"""Unit tests for the builder-backed MiniMax-M3 configs and builders."""

import dataclasses
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.moe.router import TopKRouter
from transformers import PretrainedConfig

from megatron.bridge.models.minimax_m3.minimax_m3_bridge import (
    MiniMaxM3Bridge,
    MiniMaxM3ModelProvider,
    MiniMaxM3TopKRouter,
    MiniMaxM3VLModelProvider,
    _promote_router_weights_to_float32,
    minimax_m3_block_spec,
    quick_gelu,
)
from megatron.bridge.models.minimax_m3.minimax_m3_builder import MiniMaxM3ModelBuilder, MiniMaxM3VLModelBuilder
from megatron.bridge.models.minimax_m3.minimax_m3_config import (
    MiniMaxM3ModelConfig,
    MiniMaxM3VLModelConfig,
    MiniMaxM3VLTransformerConfig,
)
from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VisionTower
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.recipes.gpt import vanilla_gpt_pretrain_config
from megatron.bridge.training.model_load_save import load_model_config


pytestmark = pytest.mark.unit

_BUILDER_MODULE = "megatron.bridge.models.minimax_m3.minimax_m3_builder"

_TEXT_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 32,
    "dense_intermediate_size": 128,
    "shared_intermediate_size": 32,
    "n_shared_experts": 1,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "hidden_act": "swigluoai",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000.0,
    "rotary_dim": 8,
    "vocab_size": 1024,
    "tie_word_embeddings": False,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "scoring_func": "sigmoid",
    "use_routing_bias": True,
    "use_qk_norm": True,
    "use_gemma_norm": True,
    "routed_scaling_factor": 2.0,
    "router_aux_loss_coef": 0.001,
    "moe_layer_freq": [0, 1, 1, 1],
    "sparse_attention_config": {
        "sparse_num_index_heads": 2,
        "sparse_index_dim": 8,
        "sparse_attention_freq": [0, 1, 1, 1],
    },
    "num_nextn_predict_layers": 1,
    "swiglu_limit": 7.0,
}

_VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 2,
    "image_size": 4,
    "num_channels": 3,
    "model_type": "clip_vision_model",
}


def _hf_config(**top_level) -> PretrainedConfig:
    """Native Transformers MiniMax-M3 config with toy sizes."""
    configuration = pytest.importorskip("transformers.models.minimax_m3_vl.configuration_minimax_m3_vl")
    values = {
        "text_config": dict(_TEXT_CONFIG),
        "vision_config": dict(_VISION_CONFIG),
        "image_token_index": 200025,
        "video_token_index": 200026,
        "projector_hidden_size": 48,
        "multimodal_projector_bias": True,
        "img_token_compression_config": {
            "image_token_compression_method": "patch_merge",
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "torch_dtype": "bfloat16",
    }
    values.update(top_level)
    return configuration.MiniMaxM3VLConfig(**values)


def _remote_vision_config() -> PretrainedConfig:
    """A generic config holding the vision shape as undeclared attributes, as remote code loads it."""
    return PretrainedConfig(**{**_VISION_CONFIG, "rope_theta": 10000.0, "vision_segment_max_frames": 4})


def _provider(hf_config: PretrainedConfig) -> MiniMaxM3VLModelProvider:
    with pytest.warns(FutureWarning):
        return MiniMaxM3Bridge().provider_bridge(SimpleNamespace(config=hf_config))


class TestMiniMaxM3ModelConfigFromHF:
    def test_bridge_opts_into_builder_conversion(self):
        assert MiniMaxM3Bridge.MODEL_CONFIG_CLASS is MiniMaxM3VLModelConfig
        assert MiniMaxM3Bridge.USE_MODEL_CONFIG_FOR_CONVERSION is True
        assert "provider_bridge" not in MiniMaxM3Bridge.__dict__

    def test_config_types(self):
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        assert type(model_config) is MiniMaxM3VLModelConfig
        assert type(model_config.transformer) is MiniMaxM3VLTransformerConfig
        assert model_config.get_builder_cls() is MiniMaxM3VLModelBuilder

    def test_builder_path_does_not_use_provider(self):
        bridge = MiniMaxM3Bridge()
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(_hf_config())

        assert model_config.num_layers == 4

    def test_language_settings(self):
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        assert model_config.transformer_layer_spec is minimax_m3_block_spec
        assert model_config.hidden_size == 64
        assert model_config.ffn_hidden_size == 128
        assert model_config.moe_ffn_hidden_size == 32
        assert model_config.moe_shared_expert_intermediate_size == 32
        assert model_config.moe_layer_freq == [0, 1, 1, 1]
        assert model_config.num_moe_experts == 8
        assert model_config.activation_func is quick_gelu
        assert model_config.activation_func_clamp_value == 7.0
        assert model_config.glu_linear_offset == 1.0
        assert model_config.layernorm_zero_centered_gamma is True
        assert model_config.rotary_percent == 0.5
        assert model_config.position_embedding_type == "rope"
        assert model_config.masked_softmax_fusion is True
        assert model_config.rope_scaling_factor == 1.0
        assert model_config.moe_router_dtype == "fp32"
        assert model_config.moe_router_enable_expert_bias is True
        assert model_config.moe_token_dispatcher_type == "flex"
        assert model_config.moe_flex_dispatcher_backend == "hybridep"
        assert model_config.mtp_num_layers is None
        assert model_config.params_dtype == torch.bfloat16
        assert model_config.seq_length == 4096
        assert model_config.share_embeddings_and_output_weights is False

    def test_vision_language_fields(self):
        hf_config = _hf_config()
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(hf_config)

        assert model_config.transformer.vision_config is hf_config.vision_config
        assert model_config.transformer.image_token_id == 200025
        assert model_config.transformer.video_token_id == 200026
        assert model_config.transformer.projector_hidden_size == 48
        assert model_config.transformer.spatial_merge_size == 2
        assert model_config.transformer.lightning_indexer_layers == [1, 2, 3]
        assert model_config.transformer.index_n_heads == 2
        assert model_config.transformer.index_head_dim == 8
        assert model_config.temporal_patch_size == 2
        assert model_config.hf_config_dict["model_type"] == "minimax_m3_vl"
        assert model_config.scatter_embedding_sequence_parallel is False
        assert model_config.special_token_ids == {"images": 200025}

    def test_builder_config_matches_legacy_provider(self):
        """Every field the provider and the builder config share has the same value."""
        provider = _provider(_hf_config())
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        assert type(provider) is MiniMaxM3VLModelProvider
        # The provider keeps the layer spec it has always used; the builder config names the function.
        assert provider.transformer_layer_spec.func is minimax_m3_block_spec
        mismatches = {}
        for owner in (model_config, model_config.transformer):
            for field in dataclasses.fields(owner):
                if field.name in {"transformer", "transformer_layer_spec"} or not hasattr(provider, field.name):
                    continue
                expected, actual = getattr(provider, field.name), getattr(owner, field.name)
                # The Hugging Face vision config is compared by content.
                if isinstance(expected, PretrainedConfig):
                    expected, actual = expected.to_dict(), actual.to_dict()
                if actual != expected:
                    mismatches[field.name] = (expected, actual)
        assert mismatches == {}

    def test_export_config_matches_legacy_provider(self):
        provider = _provider(_hf_config())
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        exported = MiniMaxM3Bridge.megatron_to_hf_config(model_config)

        assert exported == MiniMaxM3Bridge.megatron_to_hf_config(provider)
        assert exported["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"]
        assert exported["text_config"]["layer_types"][1] == "minimax_m3_sparse"

    def test_provider_kwargs_are_the_builder_kwargs(self):
        bridge = MiniMaxM3Bridge()
        provider_kwargs = bridge.hf_config_to_provider_kwargs(_hf_config())
        model_config_kwargs = bridge.hf_config_to_model_config_kwargs(_hf_config())

        layer_spec = provider_kwargs.pop("transformer_layer_spec")
        assert model_config_kwargs.pop("transformer_layer_spec") is minimax_m3_block_spec
        assert isinstance(layer_spec, partial) and layer_spec.func is minimax_m3_block_spec
        provider_vision = provider_kwargs.pop("vision_config")
        assert provider_vision.to_dict() == model_config_kwargs.pop("vision_config").to_dict()
        assert provider_kwargs == model_config_kwargs

    def test_text_config_matches_text_provider(self):
        text_provider = _provider(_hf_config()).to_text_provider()
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        text_config = model_config.to_text_config()

        assert type(text_config) is MiniMaxM3ModelConfig
        assert type(text_config.transformer) is TransformerConfig
        assert text_config.get_builder_cls() is MiniMaxM3ModelBuilder
        assert not hasattr(text_config, "vision_config")
        mismatches = {}
        for owner in (text_config, text_config.transformer):
            for field in dataclasses.fields(owner):
                if field.name in {"transformer", "transformer_layer_spec"} or not hasattr(text_provider, field.name):
                    continue
                if getattr(owner, field.name) != getattr(text_provider, field.name):
                    mismatches[field.name] = (getattr(text_provider, field.name), getattr(owner, field.name))
        assert mismatches == {}
        assert MiniMaxM3Bridge.megatron_to_hf_config(text_config) == MiniMaxM3Bridge.megatron_to_hf_config(
            text_provider
        )

    def test_flat_overrides_route_to_their_owner(self):
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        model_config.freeze_vision_model = True
        model_config.image_token_id = 7
        model_config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(model_config.transformer)
        assert model_config.freeze_vision_model is True
        assert model_config.transformer.image_token_id == 7
        assert model_config.transformer.tensor_model_parallel_size == 2
        with pytest.raises(AttributeError):
            model_config.vision_tokens = 1

    @pytest.mark.parametrize("remote_vision_config", [False, True], ids=["native", "remote"])
    def test_checkpoint_run_config_reload(self, tmp_path, remote_vision_config):
        """A config saved to run_config.yaml reloads as the same config and exports identically."""
        hf_config = _hf_config()
        if remote_vision_config:
            hf_config.vision_config = _remote_vision_config()
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(hf_config)
        model_config.freeze_vision_model = True
        vision_config = model_config.transformer.vision_config
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = model_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is MiniMaxM3VLModelConfig
        assert type(loaded.transformer) is MiniMaxM3VLTransformerConfig
        assert loaded.get_builder_cls() is MiniMaxM3VLModelBuilder
        assert loaded.transformer_layer_spec is minimax_m3_block_spec
        assert loaded.freeze_vision_model is True
        assert loaded.lightning_indexer_layers == [1, 2, 3]
        # The vision config keeps its class and its undeclared attributes.
        assert type(loaded.transformer.vision_config) is type(vision_config)
        assert loaded.transformer.vision_config.to_dict() == vision_config.to_dict()
        assert MiniMaxM3Bridge.megatron_to_hf_config(loaded) == MiniMaxM3Bridge.megatron_to_hf_config(model_config)
        # Serialization leaves the live config untouched.
        assert model_config.transformer.vision_config is vision_config

    def test_text_config_run_config_reload(self, tmp_path):
        text_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config()).to_text_config()
        run_config = vanilla_gpt_pretrain_config()
        run_config.model = text_config
        run_config.to_yaml(str(tmp_path / "run_config.yaml"))

        loaded, _ = load_model_config(str(tmp_path))

        assert type(loaded) is MiniMaxM3ModelConfig
        assert loaded.get_builder_cls() is MiniMaxM3ModelBuilder
        assert loaded.transformer_layer_spec is minimax_m3_block_spec
        assert loaded.activation_func is quick_gelu
        assert loaded.moe_layer_freq == [0, 1, 1, 1]


class TestMiniMaxM3BlockSpec:
    def test_builder_config_resolves_to_transformer_config(self, monkeypatch):
        """A builder passes its model config; the spec reads the nested transformer config and backend."""
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config()).to_text_config()
        model_config.transformer_impl = "local"
        calls = []

        def fake_get_gpt_decoder_block_spec(config, use_transformer_engine, **kwargs):
            calls.append((config, use_transformer_engine, kwargs))
            return SimpleNamespace(layer_specs=[])

        monkeypatch.setattr(
            "megatron.bridge.models.minimax_m3.minimax_m3_bridge.get_gpt_decoder_block_spec",
            fake_get_gpt_decoder_block_spec,
        )

        minimax_m3_block_spec(model_config, vp_stage=1)

        assert calls == [
            (
                model_config.transformer,
                False,
                {"normalization": None, "qk_l2_norm": False, "vp_stage": 1, "pp_rank": None},
            )
        ]


def _router(num_experts: int = 4, hidden_size: int = 8) -> TopKRouter:
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.weight = torch.nn.Parameter(torch.randn(num_experts, hidden_size, dtype=torch.float32).bfloat16())
    router.register_buffer("expert_bias", torch.randn(num_experts, dtype=torch.float32) / 3)
    return router


class TestFloat32RouterTensors:
    def test_float16_conversion_keeps_router_tensors_in_float32(self):
        """Builder models are wrapped by Megatron Core directly; the marks keep the router tensors FP32."""
        model = torch.nn.Module()
        model.router = _router()
        _promote_router_weights_to_float32([model])
        source_weight = torch.randn(4, 8, dtype=torch.float32)
        source_bias = torch.randn(4, dtype=torch.float32) / 3
        model.load_state_dict({"router.weight": source_weight, "router.expert_bias": source_bias})

        convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)

        assert model.router.weight.dtype == torch.float32
        assert model.router.expert_bias.dtype == torch.float32
        torch.testing.assert_close(model.router.weight, source_weight, rtol=0, atol=0)
        torch.testing.assert_close(model.router.expert_bias, source_bias, rtol=0, atol=0)

    @pytest.mark.parametrize("entry_point", ["bridge", "megatron_core"])
    def test_distributed_build_keeps_float32_tensors_exact(self, monkeypatch, tmp_path, entry_point):
        """FP32 tensors loaded before the wrap stay exact through the distributed build.

        ``bridge`` is ``build_distributed_models()``, which ``AutoBridge.get_model()``, the training
        build and checkpoint loading use; ``megatron_core`` is the builder's own method. Only the
        stand-in for the GPT stage is fake: the MiniMax-M3 builder, Megatron Core's distributed build
        and ``Float16Module`` run as in a real build.
        """
        from megatron.core import parallel_state
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.module import MegatronModule

        from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder
        from megatron.bridge.models.minimax_m3.modeling_minimax_m3_vl import MiniMaxM3VisionPatchEmbeddings
        from megatron.bridge.models.model_provider import build_distributed_models

        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config()).to_text_config()
        model_config.use_cpu_initialization = True

        class Stage(MegatronModule):
            def __init__(self):
                super().__init__(config=model_config.transformer)
                self.router = _router()
                self.patch_embedding = MiniMaxM3VisionPatchEmbeddings(
                    {"hidden_size": 8, "patch_size": 2, "temporal_patch_size": 2, "num_channels": 3}
                )
                self.linear = torch.nn.Linear(8, 8)

        monkeypatch.setattr(GPTModelBuilder, "build_model", lambda *args, **kwargs: Stage())
        source = {
            "router.weight": torch.randn(4, 8, dtype=torch.float32),
            "router.expert_bias": torch.randn(4, dtype=torch.float32) / 3,
            "patch_embedding.patch_embedding.weight": torch.randn(8, 3, 2, 2, 2, dtype=torch.float32),
        }

        def load_source(stages):
            # As AutoBridge.get_model() loads Hugging Face weights, or PEFT its base weights, before the wrap.
            stages[0].load_state_dict(source, strict=False)
            return stages

        model_config.pre_wrap_hooks.append(load_source)
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{tmp_path / 'dist_init'}", rank=0, world_size=1
        )
        parallel_state.initialize_model_parallel()
        try:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            if entry_point == "bridge":
                stages = build_distributed_models(model_config, pg_collection, wrap_with_ddp=False)
            else:
                stages = MiniMaxM3ModelBuilder(model_config).build_distributed_models(
                    pg_collection, wrap_with_ddp=False
                )
        finally:
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

        stage = stages[0].module
        state = stage.state_dict()
        for name, expected in source.items():
            assert state[name].dtype == torch.float32, name
            torch.testing.assert_close(state[name], expected, rtol=0, atol=0)
        assert stage.linear.weight.dtype == torch.bfloat16


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _pg_collection(pp_rank: int = 0, pp_size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pp=_FakeGroup(pp_rank, pp_size))


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList()
        self.router = _router()
        self.share_embeddings_and_output_weights = False
        self.shared_embedding_or_output_weight = None


class TestMiniMaxM3ModelBuilder:
    def test_text_builder_promotes_router_weights(self, monkeypatch):
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config()).to_text_config()
        language_model = _LanguageModel()
        gpt_build_model = Mock(return_value=language_model)
        monkeypatch.setattr("megatron.bridge.models.gpt.gpt_builder.GPTModelBuilder.build_model", gpt_build_model)
        pg_collection = _pg_collection()

        model = MiniMaxM3ModelBuilder(model_config).build_model(pg_collection, pre_process=True, vp_stage=None)

        assert model is language_model
        gpt_build_model.assert_called_once_with(pg_collection, True, None, None)
        assert model.router.weight.dtype == torch.float32
        assert model.router.weight.keep_in_fp32 is True
        assert model.router.expert_bias.keep_in_fp32 is True

    def test_vl_builder_constructs_the_vision_language_model(self, monkeypatch):
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())
        text_build_model = Mock(return_value=_LanguageModel())
        monkeypatch.setattr(MiniMaxM3ModelBuilder, "build_model", text_build_model)
        pg_collection = _pg_collection()

        model = MiniMaxM3VLModelBuilder(model_config).build_model(pg_collection, pre_process=True, post_process=True)

        assert model.config is model_config.transformer
        assert model.pg_collection is pg_collection
        assert model.language_model is text_build_model.return_value
        text_build_model.assert_called_once_with(pg_collection, pre_process=True, post_process=True, vp_stage=None)
        assert isinstance(model.vision_tower, MiniMaxM3VisionTower)
        assert sorted(model.lightning_indexers) == []
        # Checkpoint conversion has no Float16Module: the builder casts the vision modules itself,
        # keeping the FP32 patch embedding.
        patch_embedding = model.vision_tower.vision_model.embeddings.patch_embedding
        assert patch_embedding.weight.dtype == torch.float32
        for module in (model.vision_tower, model.multi_modal_projector, model.patch_merge_mlp):
            for name, parameter in module.named_parameters():
                if parameter is not patch_embedding.weight:
                    assert parameter.dtype == torch.bfloat16, name

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "pre_process", "post_process"),
        [(0, 1, True, True), (0, 2, True, False), (1, 2, False, True)],
    )
    def test_build_model_derives_stage_flags(self, monkeypatch, pp_rank, pp_size, pre_process, post_process):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        model_class = Mock(return_value=Mock())
        monkeypatch.setattr(f"{_BUILDER_MODULE}.MiniMaxM3VLModel", model_class)
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())

        MiniMaxM3VLModelBuilder(model_config).build_model(_pg_collection(pp_rank, pp_size))

        kwargs = model_class.call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == (pre_process, post_process)

    def test_build_model_freezes_requested_modules(self, monkeypatch):
        model_class = Mock(return_value=Mock())
        monkeypatch.setattr(f"{_BUILDER_MODULE}.MiniMaxM3VLModel", model_class)
        monkeypatch.setattr(f"{_BUILDER_MODULE}.convert_module_to_dtype_except_fp32_marked", Mock())
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())
        model_config.freeze_language_model = True
        model_config.freeze_vision_projection = True

        model = MiniMaxM3VLModelBuilder(model_config).build_model(
            _pg_collection(), pre_process=True, post_process=True
        )

        model.freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_builders_reject_other_configs(self):
        vl_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())
        text_config = vl_config.to_text_config()
        other_config = SimpleNamespace(transformer=text_config.transformer)

        with pytest.raises(TypeError, match="MiniMaxM3VLModelConfig"):
            MiniMaxM3VLModelBuilder(text_config).build_model(_pg_collection())
        with pytest.raises(TypeError, match="MiniMaxM3ModelConfig"):
            MiniMaxM3ModelBuilder(other_config).build_model(_pg_collection())

    def test_build_model_requires_vision_config(self, monkeypatch):
        model_class = Mock()
        monkeypatch.setattr(f"{_BUILDER_MODULE}.MiniMaxM3VLModel", model_class)
        model_config = MiniMaxM3Bridge().hf_config_to_model_config(_hf_config())
        model_config.vision_config = None

        with pytest.raises(ValueError, match="vision_config"):
            MiniMaxM3VLModelBuilder(model_config).build_model(_pg_collection())
        model_class.assert_not_called()


def test_text_provider_class_is_unchanged():
    """The deprecated text provider still builds through the MiniMax-M3 block spec."""
    text_provider = _provider(_hf_config()).to_text_provider()

    assert type(text_provider) is MiniMaxM3ModelProvider
    assert text_provider.transformer_layer_spec.func is minimax_m3_block_spec
    assert MiniMaxM3TopKRouter.__mro__[1] is TopKRouter
