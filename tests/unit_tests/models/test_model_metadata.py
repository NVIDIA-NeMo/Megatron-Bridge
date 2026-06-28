from types import SimpleNamespace

from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.base import ModelConfig
from megatron.training.models.gpt import GPTModelConfig

from megatron.bridge.models.metadata import get_hf_model_id_from_model_config


def test_get_hf_model_id_prefers_legacy_field() -> None:
    model_config = SimpleNamespace(
        hf_model_id="legacy/model",
        extra_checkpoint_metadata={"hf_model_id": "metadata/model"},
    )

    assert get_hf_model_id_from_model_config(model_config) == "legacy/model"


def test_get_hf_model_id_reads_serializable_metadata() -> None:
    model_config = {"extra_checkpoint_metadata": {"hf_model_id": "metadata/model"}}

    assert get_hf_model_id_from_model_config(model_config) == "metadata/model"


def test_get_hf_model_id_returns_none_when_absent() -> None:
    assert get_hf_model_id_from_model_config(SimpleNamespace()) is None


def test_hf_model_id_survives_model_config_roundtrip() -> None:
    model_config = GPTModelConfig(
        transformer=TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            attention_backend=AttnBackend.local,
        ),
        vocab_size=128,
        extra_checkpoint_metadata={"hf_model_id": "metadata/model"},
    )

    restored = ModelConfig.from_dict(model_config.as_dict())

    assert get_hf_model_id_from_model_config(restored) == "metadata/model"
