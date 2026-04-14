from types import SimpleNamespace

import pytest

import megatron.bridge.models.qwen_vl.qwen3_vl_provider as qwen3_vl_provider_module
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import (
    Qwen3VLModelProvider,
    Qwen3VLMoEModelProvider,
)


class _DummyQwen3VLModel:
    """Minimal stand-in for Qwen3VLModel used to inspect provider inputs."""

    def __init__(
        self,
        *,
        language_transformer_config,
        language_transformer_layer_spec,
        vision_transformer_config,
        pre_process,
        post_process,
        pg_collection,
    ):
        self.language_transformer_config = language_transformer_config
        self.language_transformer_layer_spec = language_transformer_layer_spec
        self.vision_transformer_config = vision_transformer_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = pg_collection
        self.freeze_calls = []

    def freeze(self, **kwargs):
        """Record freeze calls without touching parameters."""
        self.freeze_calls.append(kwargs)


@pytest.mark.parametrize("provider_cls", [Qwen3VLModelProvider, Qwen3VLMoEModelProvider])
def test_qwen3_vl_provide_disables_incompatible_rope_fusion(
    provider_cls,
    monkeypatch: pytest.MonkeyPatch,
):
    """Qwen3-VL providers should clear fused RoPE for mRoPE models before model build."""
    monkeypatch.setattr(
        qwen3_vl_provider_module,
        "get_gpt_layer_with_transformer_engine_spec",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(qwen3_vl_provider_module, "Qwen3VLModel", _DummyQwen3VLModel)

    provider = provider_cls(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        apply_rope_fusion=True,
    )

    model = provider.provide()

    assert provider.apply_rope_fusion is False
    assert model.language_transformer_config.apply_rope_fusion is False
