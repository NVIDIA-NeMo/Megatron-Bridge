# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Focused constructor-boundary tests for custom output-logit dtype paths."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NEMOTRON_OMNI_LLAVA_CONTRACT,
    NemotronOmniLlavaModelProvider,
)
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import NemotronVLModelProvider
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.thinker_model import Qwen3ASRThinkerModel
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import Qwen3OmniThinkerModel
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.thinker_model import Qwen25OmniThinkerModel
from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model


def _fake_megatron_init(self, config) -> None:
    nn.Module.__init__(self)
    self.config = config


def _language_config() -> SimpleNamespace:
    return SimpleNamespace(
        virtual_pipeline_model_parallel_size=None,
        vocab_size=32,
        language_max_sequence_length=16,
        hidden_size=8,
        rotary_percent=1.0,
        rotary_base=10_000,
        fp16_lm_cross_entropy=False,
        logit_dtype=torch.float32,
        share_embeddings_and_output_weights=False,
        image_token_id=1,
        video_token_id=2,
        audio_token_id=3,
        vision_start_token_id=4,
        audio_start_token_id=5,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        sequence_parallel=False,
        context_parallel_size=1,
        position_embedding_type="mrope",
        rope_scaling=False,
        rope_scaling_factor=1.0,
        mtp_num_layers=0,
        projector_bias=False,
        hf_text_config=SimpleNamespace(max_position_embeddings=16),
    )


def _layer_spec() -> SimpleNamespace:
    return SimpleNamespace(submodules=SimpleNamespace(self_attention=SimpleNamespace(module=None)))


def _pg_collection() -> SimpleNamespace:
    return SimpleNamespace(cp=None, tp=None, pp=None, embd=None)


def _fake_language_model(captured: dict[str, torch.dtype]):
    def constructor(*args, logit_dtype=None, **kwargs):
        del args, kwargs
        captured["logit_dtype"] = logit_dtype
        return SimpleNamespace(
            share_embeddings_and_output_weights=False,
            config=SimpleNamespace(cuda_graph_impl="none"),
            embedding=object(),
            register_load_state_dict_post_hook=lambda hook: None,
        )

    return constructor


@pytest.mark.parametrize(
    ("model_cls", "constructor_target", "thinker_config"),
    [
        (
            Qwen3ASRThinkerModel,
            "megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.thinker_model.Qwen3VLGPTModel",
            SimpleNamespace(audio_config=object()),
        ),
        (
            Qwen25OmniThinkerModel,
            "megatron.bridge.models.qwen_omni.modeling_qwen25_omni.thinker_model.Qwen3VLGPTModel",
            SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), audio_config=object()),
        ),
        (
            Qwen3OmniThinkerModel,
            "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model.Qwen3VLGPTModel",
            SimpleNamespace(vision_config=object(), audio_config=object()),
        ),
    ],
)
def test_qwen_thinker_forwards_logit_dtype(model_cls, constructor_target, thinker_config) -> None:
    captured: dict[str, torch.dtype] = {}
    with (
        patch("megatron.core.transformer.module.MegatronModule.__init__", new=_fake_megatron_init),
        patch(constructor_target, new=_fake_language_model(captured)),
    ):
        model_cls(
            language_transformer_config=_language_config(),
            language_transformer_layer_spec=_layer_spec(),
            thinker_transformer_config=thinker_config,
            pre_process=False,
            add_encoder=False,
            pg_collection=_pg_collection(),
        )

    assert captured["logit_dtype"] is torch.float32


def test_exaone45_wrapper_forwards_logit_dtype() -> None:
    captured: dict[str, torch.dtype] = {}
    with (
        patch("megatron.core.transformer.module.MegatronModule.__init__", new=_fake_megatron_init),
        patch(
            "megatron.bridge.models.exaone.exaone45.modelling_exaone45.model.Exaone45GPTModel",
            new=_fake_language_model(captured),
        ),
    ):
        Exaone45Model(
            language_transformer_config=_language_config(),
            language_transformer_layer_spec=object(),
            vision_transformer_config=SimpleNamespace(spatial_merge_size=2),
            pre_process=False,
            add_encoder=False,
            pg_collection=_pg_collection(),
        )

    assert captured["logit_dtype"] is torch.float32


def test_step37_wrapper_forwards_logit_dtype() -> None:
    captured: dict[str, torch.dtype] = {}
    with (
        patch("megatron.core.transformer.module.MegatronModule.__init__", new=_fake_megatron_init),
        patch(
            "megatron.bridge.models.stepfun.modelling_step37.model.Step37GPTModel",
            new=_fake_language_model(captured),
        ),
    ):
        Step37Model(
            language_transformer_config=_language_config(),
            language_transformer_layer_spec=object(),
            vision_transformer_config=SimpleNamespace(width=8),
            pre_process=False,
            add_encoder=False,
            pg_collection=_pg_collection(),
        )

    assert captured["logit_dtype"] is torch.float32


def test_nemotron_omni_wrapper_forwards_logit_dtype() -> None:
    captured: dict[str, torch.dtype] = {}
    with (
        patch("megatron.core.transformer.module.MegatronModule.__init__", new=_fake_megatron_init),
        patch(
            "megatron.bridge.models.nemotron_omni.modeling_nemotron_omni.HybridModel",
            new=_fake_language_model(captured),
        ),
    ):
        NemotronOmniModel(
            language_transformer_config=_language_config(),
            language_transformer_layer_spec=object(),
            language_vocab_size=32,
            language_max_sequence_length=16,
            vision_transformer_config=object(),
            vision_transformer_layer_spec=object(),
            vision_projection_config=object(),
            vision_projection_layer_spec=object(),
            image_token_index=1,
            pre_process=False,
            add_encoder=False,
            pg_collection=_pg_collection(),
        )

    assert captured["logit_dtype"] is torch.float32


def test_nemotron_vl_provider_rejects_requested_dtype_at_llava_boundary() -> None:
    provider = NemotronVLModelProvider(logit_dtype=torch.float32)

    with pytest.raises(RuntimeError, match="LLaVAModel does not support logit_dtype"):
        provider.provide()


def test_legacy_nemotron_omni_provider_rejects_requested_dtype_at_llava_boundary() -> None:
    provider = NemotronOmniLlavaModelProvider(
        nemotron_omni_contract=NEMOTRON_OMNI_LLAVA_CONTRACT,
        logit_dtype=torch.float32,
    )
    with (
        patch.object(provider, "_validate_omni_config"),
        patch.object(provider, "_build_vision_config", return_value=SimpleNamespace(class_token_len=0)),
        patch.object(provider, "_build_vision_projection_config", return_value=object()),
        patch.object(provider, "_build_sound_modules", return_value=(None, None)),
        patch(
            "megatron.bridge.models.nemotron_omni.nemotron_omni_provider.get_vit_layer_with_transformer_engine_spec",
            return_value=object(),
        ),
        patch(
            "megatron.bridge.models.nemotron_omni.nemotron_omni_provider.get_language_mlp_submodules",
            return_value=object(),
        ),
    ):
        with pytest.raises(RuntimeError, match="LLaVAModel does not support logit_dtype"):
            provider._provide_llava()


def test_gemma4_dense_forwards_logit_dtype() -> None:
    captured: dict[str, torch.dtype] = {}
    provider = Gemma4DenseProvider(logit_dtype=torch.float32)
    provider._gemma4_dense_finalized = True
    fake_model = SimpleNamespace()

    def fake_gpt(*args, logit_dtype=None, **kwargs):
        del args, kwargs
        captured["logit_dtype"] = logit_dtype
        return fake_model

    with (
        patch("megatron.core.models.gpt.GPTModel", new=fake_gpt),
        patch("megatron.bridge.models.gemma.gemma4_provider.get_gemma4_layer_spec", return_value=object()),
        patch("megatron.bridge.models.gemma.gemma4_provider.Gemma4DenseRotaryEmbedding", return_value=object()),
        patch("megatron.bridge.models.gemma.gemma4_provider._attach_ple_modules"),
        patch("megatron.bridge.models.gemma.gemma4_provider.wire_gemma4_kv_sharing"),
        patch("megatron.bridge.models.gemma.gemma4_provider._install_ple_forward"),
        patch("megatron.bridge.models.gemma.gemma4_provider._install_gemma4_dense_load_state_aliases"),
    ):
        provider.build()

    assert captured["logit_dtype"] is torch.float32


def test_falcon_model_owned_head_forwards_output_dtype() -> None:
    captured: dict[str, torch.dtype] = {}
    config = SimpleNamespace(
        hidden_size=8,
        init_method=None,
        params_dtype=torch.bfloat16,
        quant_recipe=None,
    )
    pg_collection = SimpleNamespace(tp=None, cp=None)

    def fake_language_module_init(self, config, pg_collection=None) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.pg_collection = pg_collection

    def fake_column_parallel_linear(*args, output_dtype=None, **kwargs):
        del args, kwargs
        captured["output_dtype"] = output_dtype
        return nn.Identity()

    with (
        patch(
            "megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model.LanguageModule.__init__",
            new=fake_language_module_init,
        ),
        patch(
            "megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model.tensor_parallel.ColumnParallelLinear",
            new=fake_column_parallel_linear,
        ),
        patch(
            "megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model.build_module",
            return_value=nn.Identity(),
        ),
        patch(
            "megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model.has_config_logger_enabled",
            return_value=False,
        ),
        patch.object(FalconH1Model, "setup_embeddings_and_output_layer"),
    ):
        FalconH1Model(
            config=config,
            falconh1_stack_spec=object(),
            vocab_size=32,
            max_sequence_length=16,
            pre_process=False,
            post_process=True,
            logit_dtype=torch.float32,
            pg_collection=pg_collection,
        )

    assert captured["output_dtype"] is torch.float32
