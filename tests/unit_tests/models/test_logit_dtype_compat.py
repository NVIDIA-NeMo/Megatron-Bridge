# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import inspect
from dataclasses import fields

import pytest
import torch

from megatron.bridge.models.exaone.exaone45.modelling_exaone45.text_model import Exaone45GPTModel
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import Exaone45TransformerConfig
from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg, output_dtype_kwarg
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.transformer_config import Qwen3ASRTransformerConfig
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.transformer_config import Qwen25OmniTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.stepfun.modelling_step37.transformer_config import Step37TransformerConfig


def test_custom_gpt_models_mirror_logit_dtype() -> None:
    """Bridge GPT subclasses must not narrow the corrected MCore constructor."""
    for model_cls in (Qwen3VLGPTModel, Exaone45GPTModel):
        parameter = inspect.signature(model_cls).parameters.get("logit_dtype")

        assert parameter is not None
        assert parameter.default is None


def test_custom_output_layer_provider_exposes_logit_dtype() -> None:
    """Falcon H1 owns its LM head and therefore needs the same provider contract."""
    provider = FalconH1ModelProvider(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=1,
        vocab_size=1000,
        logit_dtype=torch.float32,
    )

    assert provider.logit_dtype is torch.float32


def test_custom_wrapper_configs_declare_logit_dtype() -> None:
    for config_cls in (
        Qwen3VLTransformerConfig,
        Qwen25OmniTransformerConfig,
        Qwen3ASRTransformerConfig,
        Exaone45TransformerConfig,
        Step37TransformerConfig,
    ):
        field_names = {field.name for field in fields(config_cls)}

        assert "logit_dtype" in field_names


def test_default_dtype_omits_keyword_for_old_constructor() -> None:
    class OldConstructor:
        def __init__(self) -> None:
            pass

    assert logit_dtype_kwarg(OldConstructor, None) == {}
    assert output_dtype_kwarg(OldConstructor, None) == {}


def test_requested_dtype_fails_clearly_for_old_constructor() -> None:
    class OldConstructor:
        def __init__(self) -> None:
            pass

    with pytest.raises(RuntimeError, match="Megatron-LM PR #6252"):
        logit_dtype_kwarg(OldConstructor, torch.float32)


def test_llava_rejection_does_not_claim_base_pr_is_sufficient() -> None:
    from megatron.core.models.multimodal.llava_model import LLaVAModel

    with pytest.raises(RuntimeError, match="this constructor needs a compatible implementation"):
        logit_dtype_kwarg(LLaVAModel, torch.float32)


def test_unsupported_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="logit_dtype must be one of"):
        logit_dtype_kwarg(object, torch.float16)


def test_requested_dtype_is_forwarded_for_new_constructor() -> None:
    class NewConstructor:
        def __init__(
            self,
            logit_dtype: torch.dtype | None = None,
            output_dtype: torch.dtype | None = None,
        ) -> None:
            pass

    assert logit_dtype_kwarg(NewConstructor, torch.float32) == {"logit_dtype": torch.float32}
    assert output_dtype_kwarg(NewConstructor, torch.float32) == {"output_dtype": torch.float32}
