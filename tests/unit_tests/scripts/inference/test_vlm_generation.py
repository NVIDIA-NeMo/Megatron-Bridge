# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for deterministic VLM generation configuration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


_SCRIPT_PATH = Path(__file__).parents[4] / "scripts" / "inference" / "vlm_generation.py"
_SPEC = importlib.util.spec_from_file_location("vlm_generation_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
vlm_generation = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(_SCRIPT_PATH.parent))
try:
    _SPEC.loader.exec_module(vlm_generation)
finally:
    sys.path.pop(0)


pytestmark = pytest.mark.unit

_INKLING2_FIXTURES = Path(__file__).parents[2] / "models" / "inkling" / "fixtures"


def test_model_builder_requires_explicit_conversion_migration() -> None:
    legacy_bridge = SimpleNamespace(_model_bridge=SimpleNamespace(MODEL_CONFIG_CLASS=object))
    migrated_bridge = SimpleNamespace(
        _model_bridge=SimpleNamespace(MODEL_CONFIG_CLASS=object, USE_MODEL_CONFIG_FOR_CONVERSION=True)
    )

    assert vlm_generation._uses_model_builder(legacy_bridge) is False
    assert vlm_generation._uses_model_builder(migrated_bridge) is True


def test_deterministic_execution_sets_required_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_ALGO",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    ):
        monkeypatch.delenv(name, raising=False)

    with (
        patch.object(vlm_generation.torch, "use_deterministic_algorithms"),
        patch.object(vlm_generation.torch, "manual_seed"),
        patch.object(vlm_generation.torch.cuda, "manual_seed_all"),
    ):
        vlm_generation._enable_deterministic_execution()

    assert vlm_generation.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert vlm_generation.os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert vlm_generation.os.environ["NCCL_ALGO"] == "Ring"
    assert vlm_generation.os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"


def test_deterministic_execution_rejects_conflicting_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        vlm_generation._enable_deterministic_execution()


def test_deterministic_execution_seeds_and_configures_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setattr(vlm_generation.torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(vlm_generation.torch.backends.cudnn, "deterministic", False)

    with (
        patch.object(vlm_generation.torch, "use_deterministic_algorithms") as use_deterministic,
        patch.object(vlm_generation.torch, "manual_seed") as manual_seed,
        patch.object(vlm_generation.torch.cuda, "manual_seed_all") as cuda_manual_seed_all,
    ):
        vlm_generation._enable_deterministic_execution()

    use_deterministic.assert_called_once_with(True)
    manual_seed.assert_called_once_with(0)
    cuda_manual_seed_all.assert_called_once_with(0)
    assert vlm_generation.torch.backends.cudnn.benchmark is False
    assert vlm_generation.torch.backends.cudnn.deterministic is True


def test_builder_configuration_propagates_deterministic_mode() -> None:
    transformer = SimpleNamespace()
    model_config = SimpleNamespace(transformer=transformer)
    bridge = SimpleNamespace(get_model_config=MagicMock(return_value=model_config))

    result = vlm_generation._configure_builder_model(
        bridge,
        tp=8,
        pp=1,
        ep=1,
        etp=1,
        deterministic=True,
    )

    assert result is model_config
    assert transformer.deterministic_mode is True
    assert transformer.tensor_model_parallel_size == 8


@pytest.mark.parametrize("uses_builder", [False, True])
def test_checkpoint_generation_disables_dropping_without_replacing_dispatcher(uses_builder: bool) -> None:
    class LoadReached(Exception):
        pass

    bridge = MagicMock()
    bridge._model_bridge = SimpleNamespace(USE_MODEL_CONFIG_FOR_CONVERSION=uses_builder)
    bridge.load_megatron_model.side_effect = LoadReached
    args = SimpleNamespace(
        deterministic=False,
        tp=1,
        pp=1,
        ep=8,
        etp=1,
        pp_layout=None,
        hf_model_path="test/model",
        hf_revision="test-revision",
        megatron_model_path="work/checkpoint",
        trust_remote_code=True,
    )
    with (
        patch.object(vlm_generation, "maybe_initialize_distributed"),
        patch.object(vlm_generation, "is_safe_repo", return_value=True),
        patch.object(vlm_generation, "print_rank_0"),
        patch.object(
            vlm_generation.AutoConfig,
            "from_pretrained",
            return_value=SimpleNamespace(model_type="test", architectures=[]),
        ),
        patch.object(vlm_generation.AutoBridge, "from_hf_pretrained", return_value=bridge),
        pytest.raises(LoadReached),
    ):
        vlm_generation.main(args)

    bridge.load_megatron_model.assert_called_once()
    assert bridge.load_megatron_model.call_args.args == ("work/checkpoint",)
    overrides = bridge.load_megatron_model.call_args.kwargs["mp_overrides"]
    assert overrides["moe_expert_capacity_factor"] is None
    assert overrides["moe_expert_rank_capacity_factor"] is None
    assert overrides["moe_paged_stash"] is False
    assert overrides["moe_ncclep_zero_copy"] is False
    assert overrides["moe_pad_expert_input_to_capacity"] is False
    assert overrides["moe_router_force_load_balancing"] is False
    assert overrides["moe_router_force_biased"] is None
    assert overrides["moe_hybridep_pad_uneven_dispatch_inputs"] is True
    assert "moe_token_dispatcher_type" not in overrides
    assert "moe_flex_dispatcher_backend" not in overrides


def test_inkling2_image_prompt_reaches_the_vision_encoder_in_bridge_layout(tmp_path: Path) -> None:
    pytest.importorskip("transformers", minversion="5.23")
    from megatron.bridge.models.inkling.data.tokenizer import INKLING2_BOS_TOKEN, INKLING2_EOS_TOKEN
    from megatron.bridge.models.inkling.modeling_inkling2.vision import Inkling2VisionEncoder

    class InputsReached(Exception):
        pass

    image_token = "<|unused_200054|>"
    special = [
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_image|>",
        "<|content_model_end_sampling|>",
        "<|content_thinking|>",
        "<|content_invoke_tool_json|>",
        "<|content_xml|>",
        "<|end_message|>",
        image_token,
        INKLING2_BOS_TOKEN,
        INKLING2_EOS_TOKEN,
    ]
    words = ["Describe", "this", "image", ".", "Thinking", "effort", "level", ":", "0", "9"]
    tokens = ["<unk>", "<pad>", *special, *words]
    image_token_id = tokens.index(image_token)
    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    fast.add_special_tokens({"additional_special_tokens": special})
    fast.chat_template = (_INKLING2_FIXTURES / "inkling2_small_78cb6c0e_chat_template.jinja").read_text()
    checkpoint = tmp_path / "checkpoint"
    fast.save_pretrained(checkpoint)
    config = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2",
        "image_token_id": image_token_id,
        "text_config": {"vocab_size": 128, "unpadded_vocab_size": 64},
        "vision_config": {"hidden_size": 32, "num_hidden_layers": 2, "native_patch_size": 20},
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    processor_config = _INKLING2_FIXTURES / "inkling2_small_78cb6c0e_processor_config.json"
    (checkpoint / "processor_config.json").write_text(processor_config.read_text())
    image_path = tmp_path / "image.png"
    Image.new("RGB", (81, 41), (200, 100, 50)).save(image_path)

    bridge = MagicMock()
    bridge._model_bridge = SimpleNamespace(USE_MODEL_CONFIG_FOR_CONVERSION=False)
    bridge.to_megatron_provider.return_value.provide_distributed_model.return_value = []
    captured = {}
    real_process_image_inputs = vlm_generation.process_image_inputs

    def capture_inputs(*args, **kwargs):
        captured["outputs"] = real_process_image_inputs(*args, **kwargs)
        raise InputsReached

    args = SimpleNamespace(
        deterministic=False,
        tp=1,
        pp=1,
        ep=1,
        etp=1,
        pp_layout=None,
        hf_model_path=str(checkpoint),
        hf_revision=None,
        megatron_model_path=None,
        trust_remote_code=False,
        image_path=str(image_path),
        image_paths=None,
        video_path=None,
        prompt="Describe this image .",
    )
    with (
        patch.object(vlm_generation, "maybe_initialize_distributed"),
        patch.object(vlm_generation, "is_safe_repo", return_value=False),
        patch.object(vlm_generation, "print_rank_0"),
        patch.object(vlm_generation.AutoBridge, "from_hf_pretrained", return_value=bridge),
        patch.object(vlm_generation, "process_image_inputs", side_effect=capture_inputs),
        pytest.raises(InputsReached),
    ):
        vlm_generation.main(args)

    input_ids, pixel_values, image_grid_thw, *_ = captured["outputs"]
    vision_config = {
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 16,
        "projector_bias": True,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "mlp_ratio": 2.0,
        "posemb_grid_size": 4,
        "layer_norm_eps": 1e-6,
    }
    encoder = Inkling2VisionEncoder(vision_config, text_hidden_size=8).float().eval()
    with torch.no_grad():
        embeddings = encoder(pixel_values, image_grid_thw)

    assert int((input_ids == image_token_id).sum()) == pixel_values.shape[0]
    assert embeddings.shape == (pixel_values.shape[0], 8)
