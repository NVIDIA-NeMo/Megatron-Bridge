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

"""Unit tests for the Qwen3.5-VL builder-backed model config and builder."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.qwen35_vl_builder import Qwen35VLModelBuilder
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig, Qwen35VLTransformerConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import _TRANSFORMERS_HAS_QWEN3_5_MOE
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.model_load_save import load_model_config
from megatron.bridge.utils.yaml_utils import dump_dataclass_to_yaml


pytestmark = pytest.mark.skipif(
    not _TRANSFORMERS_HAS_QWEN3_5_MOE, reason="transformers does not have qwen3_5_moe support"
)

_BUILDER_MODULE = "megatron.bridge.models.qwen_vl.qwen35_vl_builder"
_PROVIDER_MODULE = "megatron.bridge.models.qwen_vl.qwen35_vl_provider"


def _model_config(**overrides) -> Qwen35VLModelConfig:
    """Map a small Qwen3.5 MoE vision-language HF config to its builder config."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge

    hf_config = Qwen3_5MoeConfig(
        text_config=dict(
            hidden_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            vocab_size=128,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            mtp_num_hidden_layers=1,
            dtype="bfloat16",
        ),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )
    model_config = Qwen35VLMoEBridge().hf_config_to_model_config(hf_config)
    for name, value in overrides.items():
        setattr(model_config, name, value)
    return model_config


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


def _attention_spec() -> ModuleSpec:
    return ModuleSpec(module=SelfAttention, submodules=SimpleNamespace())


@pytest.fixture
def build_calls(monkeypatch):
    """Replace the model and spec factories, recording the arguments the builder passes."""
    calls = {}
    gdn_attention = ModuleSpec(module=object, submodules=SimpleNamespace())
    block_spec = SimpleNamespace(
        layer_specs=[
            ModuleSpec(module=object, submodules=SimpleNamespace(self_attention=gdn_attention)),
            ModuleSpec(module=object, submodules=SimpleNamespace(self_attention=_attention_spec())),
        ]
    )
    mtp_spec = SimpleNamespace(
        layer_specs=[
            ModuleSpec(
                module=object,
                submodules=SimpleNamespace(
                    mtp_model_layer=ModuleSpec(
                        module=object, submodules=SimpleNamespace(self_attention=_attention_spec())
                    )
                ),
            )
        ]
    )

    def block_spec_factory(config, vp_stage=None, pp_rank=None):
        calls["block_spec"] = dict(config=config, vp_stage=vp_stage, pp_rank=pp_rank)
        return block_spec

    def mtp_spec_factory(config, transformer_layer_spec, vp_stage=None, pp_rank=None):
        calls["mtp_spec"] = dict(
            config=config, transformer_layer_spec=transformer_layer_spec, vp_stage=vp_stage, pp_rank=pp_rank
        )
        return mtp_spec

    model = Mock()
    model_class = Mock(return_value=model)
    monkeypatch.setattr(
        f"{_PROVIDER_MODULE}.get_transformer_block_with_experimental_attention_variant_spec", block_spec_factory
    )
    monkeypatch.setattr(f"{_BUILDER_MODULE}.mtp_block_spec", mtp_spec_factory)
    monkeypatch.setattr(f"{_BUILDER_MODULE}.Qwen3VLModel", model_class)
    calls.update(model=model, model_class=model_class, gdn_attention=gdn_attention)
    return calls


class TestQwen35VLModelConfig:
    def test_defaults(self):
        config = Qwen35VLModelConfig(
            transformer=Qwen35VLTransformerConfig(num_layers=4, hidden_size=64, num_attention_heads=4)
        )

        assert config.position_embedding_type == "mrope"
        assert config.scatter_embedding_sequence_parallel is False
        assert config.vision_config is None
        assert config.special_token_ids == {"images": 248056}
        assert Qwen35VLModelConfig.modality_keys == {"images": "qwen_visual"}
        assert config.vision_recompute_granularity == "inherit"
        assert config.vision_cuda_graph_impl == "none"
        assert config.transformer.apply_rotary_pos_emb_in_fp32 is False
        assert config.transformer.spatial_merge_size == 2
        assert config.transformer.vision_dp_when_cp is False
        assert config.get_builder_cls() is Qwen35VLModelBuilder

    def test_flat_overrides_route_to_their_owner(self):
        config = _model_config()

        config.freeze_vision_model = True
        config.vision_dp_when_cp = True
        config.tensor_model_parallel_size = 2

        assert "freeze_vision_model" not in vars(config.transformer)
        assert config.freeze_vision_model is True
        assert config.transformer.vision_dp_when_cp is True
        assert config.transformer.tensor_model_parallel_size == 2
        # The providers carried these as undeclared attributes; the builder config rejects them.
        for name in ("head_dim", "hf_text_config", "audio_token_id"):
            with pytest.raises(AttributeError):
                setattr(config, name, None)

    @pytest.mark.parametrize(("context_parallel_size", "per_token_loss"), [(1, False), (2, True)])
    def test_finalize_computes_per_token_loss_for_context_parallelism(self, context_parallel_size, per_token_loss):
        config = _model_config(context_parallel_size=context_parallel_size)

        config.finalize()

        assert config.calculate_per_token_loss is per_token_loss

    def test_serialization_round_trip(self, tmp_path):
        config = _model_config(
            freeze_vision_model=True,
            vision_recompute_granularity="full",
            vision_recompute_method="block",
            vision_recompute_num_layers=2,
            vision_dp_when_cp=True,
        )
        serialized = ConfigContainer._convert_value_to_dict(config)
        (tmp_path / "run_config.yaml").write_text(dump_dataclass_to_yaml({"model": serialized}))

        restored, _ = load_model_config(str(tmp_path))

        assert isinstance(restored, Qwen35VLModelConfig)
        assert isinstance(restored.transformer, Qwen35VLTransformerConfig)
        assert restored.get_builder_cls() is Qwen35VLModelBuilder
        assert type(restored.vision_config) is type(config.vision_config)
        assert restored.vision_config.to_dict() == config.vision_config.to_dict()
        for name in (
            "freeze_vision_model",
            "vision_recompute_granularity",
            "vision_recompute_method",
            "vision_recompute_num_layers",
            "vision_dp_when_cp",
            "image_token_id",
            "mrope_section",
            "experimental_attention_variant",
            "linear_attention_freq",
            "num_moe_experts",
            "mtp_num_layers",
            "attention_softmax_in_fp32",
            "activation_func",
            "params_dtype",
        ):
            assert getattr(restored, name) == getattr(config, name), name
        assert yaml.safe_load((tmp_path / "run_config.yaml").read_text())["model"]["_builder_"] == config.builder


class TestQwen35VLModelBuilder:
    def test_build_model_combines_language_and_vision(self, build_calls):
        config = _model_config()

        model = Qwen35VLModelBuilder(config).build_model(
            _pg_collection(pp_rank=1, pp_size=2), pre_process=False, post_process=True, vp_stage=None
        )

        assert model is build_calls["model"]
        kwargs = build_calls["model_class"].call_args.kwargs
        assert kwargs["language_transformer_config"] is config.transformer
        assert kwargs["model_config"] is config
        assert kwargs["vision_transformer_config"] is config.vision_config
        assert config.vision_config.dtype == config.params_dtype
        assert (kwargs["pre_process"], kwargs["post_process"], kwargs["vp_stage"]) == (False, True, None)
        assert build_calls["block_spec"] == dict(config=config.transformer, vp_stage=None, pp_rank=1)
        assert build_calls["mtp_spec"]["config"] is config
        assert build_calls["mtp_spec"]["transformer_layer_spec"] is kwargs["language_transformer_layer_spec"]
        assert build_calls["mtp_spec"]["pp_rank"] == 1
        # Standard attention layers, including MTP's, take mRoPE; GDN layers are untouched.
        language_layers = kwargs["language_transformer_layer_spec"].layer_specs
        assert language_layers[0].submodules.self_attention is build_calls["gdn_attention"]
        assert build_calls["gdn_attention"].module is object
        assert language_layers[1].submodules.self_attention.module is Qwen3VLSelfAttention
        mtp_layer = kwargs["mtp_block_spec"].layer_specs[0].submodules.mtp_model_layer
        assert mtp_layer.submodules.self_attention.module is Qwen3VLSelfAttention
        build_calls["model"].freeze.assert_not_called()
        # Without a Float16Module wrapper the vision encoder would stay in fp32.
        build_calls["model"].vision_model.to.assert_called_once_with(dtype=config.params_dtype)

    def test_build_model_skips_cast_without_vision_encoder(self, build_calls):
        build_calls["model"].vision_model = None

        model = Qwen35VLModelBuilder(_model_config()).build_model(
            _pg_collection(pp_rank=1, pp_size=2), pre_process=False, post_process=True
        )

        assert model.vision_model is None

    @pytest.mark.parametrize(
        ("pp_rank", "pp_size", "vp_stage", "vp_size", "pre_process", "post_process"),
        [
            (0, 1, None, None, True, True),
            (0, 2, None, None, True, False),
            (1, 2, None, None, False, True),
            (0, 2, 1, 2, False, False),
            (1, 2, 1, 2, False, True),
        ],
    )
    def test_build_model_derives_stage_flags(
        self, build_calls, monkeypatch, pp_rank, pp_size, vp_stage, vp_size, pre_process, post_process
    ):
        # Megatron-Core reads stage ranks from the process group only once distributed is initialized.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        config = _model_config(virtual_pipeline_model_parallel_size=vp_size)

        model = Qwen35VLModelBuilder(config).build_model(_pg_collection(pp_rank, pp_size), vp_stage=vp_stage)

        kwargs = build_calls["model_class"].call_args.kwargs
        assert (kwargs["pre_process"], kwargs["post_process"]) == (pre_process, post_process)
        if vp_stage is not None:
            assert model.vp_stage == vp_stage

    def test_build_model_freezes_requested_modules(self, build_calls):
        config = _model_config(freeze_language_model=True, freeze_vision_projection=True)

        Qwen35VLModelBuilder(config).build_model(_pg_collection(), pre_process=True, post_process=True)

        build_calls["model"].freeze.assert_called_once_with(
            freeze_language_model=True, freeze_vision_model=False, freeze_vision_projection=True
        )

    def test_build_model_rejects_other_configs(self, build_calls):
        from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
        from megatron.bridge.models.transformer_config import TransformerConfig

        text_config = BridgeGPTModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4), vocab_size=128
        )
        with pytest.raises(TypeError, match="Qwen35VLModelConfig"):
            Qwen35VLModelBuilder(text_config).build_model(_pg_collection())

    def test_build_model_requires_vision_config(self, build_calls):
        with pytest.raises(ValueError, match="vision_config"):
            Qwen35VLModelBuilder(_model_config(vision_config=None)).build_model(_pg_collection())

    def test_build_model_rejects_custom_layer_spec(self, build_calls):
        config = _model_config()
        config.transformer_layer_spec = ModuleSpec(module=torch.nn.Identity)

        with pytest.raises(ValueError, match="transformer_layer_spec"):
            Qwen35VLModelBuilder(config).build_model(_pg_collection())
