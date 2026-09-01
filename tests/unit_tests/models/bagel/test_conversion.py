# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

from megatron.bridge import AutoBridge
from megatron.bridge.models.bagel.conversion import BagelBridge, BagelConfig
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


def _config() -> BagelConfig:
    return BagelConfig(
        architectures=["BagelForConditionalGeneration"],
        torch_dtype="bfloat16",
        llm_config={
            "num_hidden_layers": 28,
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "intermediate_size": 18944,
            "max_position_embeddings": 32768,
            "vocab_size": 152064,
            "rms_norm_eps": 1.0e-6,
            "rope_theta": 1_000_000.0,
            "attention_dropout": 0.0,
        },
        vit_config={"hidden_size": 1152},
        vae_config={"z_channels": 16},
    )


def test_auto_bridge_builds_bagel_provider_from_native_config():
    """Resolve BAGEL by architecture and preserve its nested official config."""
    bridge = AutoBridge.from_hf_config(_config())

    provider = bridge.to_megatron_provider(load_weights=False)

    assert isinstance(bridge._model_bridge, BagelBridge)
    assert provider.num_layers == 28
    assert provider.hidden_size == 3584
    assert provider.num_query_groups == 4
    assert provider.official_config_values["model_type"] == "bagel"


def test_bagel_bridge_loads_only_official_ema_file(tmp_path):
    """Route only ema.safetensors into the strict native BAGEL mapper."""
    config = _config()
    source = SafeTensorsStateSource(tmp_path)
    pretrained = SimpleNamespace(config=config, state=SimpleNamespace(source=source))
    model = object()
    report = SimpleNamespace(
        source_tensors_consumed=1,
        target_tensors_verified=1,
        fp32_main_tensors_preserved=1,
    )

    with patch(
        "megatron.bridge.models.bagel.conversion.initialize_bagel_from_native_checkpoint",
        return_value=report,
    ) as initialize:
        models = BagelBridge().load_weights_hf_to_megatron(pretrained, model)

    assert models == [model]
    args, kwargs = initialize.call_args
    assert args == (model, str(tmp_path / "ema.safetensors"))
    assert kwargs["validate_metadata"] is False
    assert kwargs["llm_config"].num_hidden_layers == 28
