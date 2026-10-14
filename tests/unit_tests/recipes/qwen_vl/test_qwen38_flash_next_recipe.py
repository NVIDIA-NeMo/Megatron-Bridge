"""Recipe integration contracts; no model weights or GPUs are allocated."""

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.qwen.qwen4_exp_bridge import Qwen4ExpBridge
from megatron.bridge.recipes import qwen38_flash_next_sft_64gpu_gb200_bf16_config
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.models.qwen.test_qwen4_exp_vlm import full_hf_config  # noqa: F401


pytestmark = pytest.mark.unit


def test_full_vl_recipe_roundtrip_and_weight_initialization_contract(full_hf_config, tmp_path):
    # Mock only HF I/O. Exercise real model, dataset, optimizer and logger configs.
    model = Qwen4ExpBridge().hf_config_to_model_config(full_hf_config)
    bridge = Mock()
    bridge.get_model_config.return_value = model
    with patch(
        "megatron.bridge.recipes.qwen_vl.qwen38_flash_next.AutoBridge.from_hf_pretrained",
        return_value=bridge,
    ) as load:
        cfg = qwen38_flash_next_sft_64gpu_gb200_bf16_config(hf_path="work/cache/qwen3.8-flash-next")

    revision = cfg.dataset.task_encoder.hf_processor_revision
    load.assert_called_once_with("work/cache/qwen3.8-flash-next", revision=revision, text_only=False)
    assert len(revision) == 40
    assert cfg.tokenizer.hf_tokenizer_kwargs["revision"] == revision
    assert not cfg.model.hf_model_text_only
    assert cfg.model.vision_config["patch_size"] == cfg.dataset.task_encoder.patch_size
    # A caller must supply pretrained_checkpoint to standard finetune; no hidden
    # pre-wrap load or private execution object is serialized into the recipe.
    assert cfg.checkpoint.pretrained_checkpoint is None
    assert not cfg.model.pre_wrap_hooks
    assert not cfg.model.perform_initialization
    assert cfg.model.calculate_per_token_loss
    assert not cfg.ddp.average_in_collective
    assert cfg.ddp.grad_reduce_in_fp32 and cfg.mixed_precision.grad_reduce_in_fp32
    assert cfg.dataset.packing_buffer_size is None
    assert not cfg.dataset.enable_in_batch_packing
    assert cfg.logger.log_throughput and cfg.logger.log_interval == 1
    assert cfg.logger.tensorboard_log_interval == 1

    path = tmp_path / "recipe.yaml"
    cfg.to_yaml(str(path))
    restored = ConfigContainer.from_yaml(str(path))
    assert type(restored.model) is type(cfg.model)
    assert restored.model.vision_config == cfg.model.vision_config
    assert type(restored.dataset.task_encoder) is type(cfg.dataset.task_encoder)
    assert restored.dataset.task_encoder.hf_processor_revision == revision
    assert restored.logger.log_throughput
