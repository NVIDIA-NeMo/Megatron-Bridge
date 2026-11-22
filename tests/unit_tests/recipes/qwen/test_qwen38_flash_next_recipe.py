"""Text-only recipe and frozen-table configuration serialization contracts."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.qwen.qwen4_exp_bridge import Qwen4ExpTextBridge
from megatron.bridge.recipes import (
    qwen38_flash_next_text_frozen_ngram_pretrain_32gpu_gb200_bf16_muon_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.models.qwen.test_qwen4_exp_vlm import full_hf_config  # noqa: F401


pytestmark = pytest.mark.unit


def test_frozen_ngram_text_recipe_roundtrip(full_hf_config, tmp_path):
    text = full_hf_config.text_config
    text.architectures = ["Qwen4ExpForCausalLM"]
    model = Qwen4ExpTextBridge().hf_config_to_model_config(text)
    assert not model.ple_freeze_ngram_embeddings
    bridge = Mock()
    bridge.get_model_config.return_value = model
    with patch(
        "megatron.bridge.recipes.qwen.gb200.qwen38_flash_next.AutoBridge.from_hf_pretrained",
        return_value=bridge,
    ) as load:
        cfg = qwen38_flash_next_text_frozen_ngram_pretrain_32gpu_gb200_bf16_muon_config(hf_path="work/cache/model")
    load.assert_called_once_with(
        "work/cache/model", revision=cfg.tokenizer.hf_tokenizer_kwargs["revision"], text_only=True
    )
    path = tmp_path / "recipe.yaml"
    cfg.to_yaml(str(path))
    restored = ConfigContainer.from_yaml(str(path))
    for candidate in (cfg, restored):
        assert candidate.model.hf_model_text_only
        assert candidate.model.ple_freeze_ngram_embeddings
        assert candidate.model.tensor_model_parallel_size == 2
        assert candidate.model.sequence_parallel
        assert candidate.model.expert_model_parallel_size == 32
        assert candidate.model.expert_tensor_parallel_size == 1
        assert candidate.model.pipeline_model_parallel_size == candidate.model.context_parallel_size == 1
        assert candidate.model.seq_length == candidate.dataset.seq_length == 4096
        assert not candidate.model.pre_wrap_hooks
        assert candidate.model.moe_flex_dispatcher_backend == "hybridep"
        assert candidate.checkpoint.finetune
        assert candidate.checkpoint.load is None
        assert candidate.checkpoint.pretrained_checkpoint is None
        assert candidate.train.train_iters == 30
        assert candidate.train.global_batch_size == 32
        assert candidate.model.ple_embedding_sharding == "ep"
        assert not candidate.model.moe_router_force_load_balancing
        assert candidate.logger.log_throughput
        assert candidate.optimizer.optimizer == "dist_muon"
        assert candidate.optimizer.muon_tp_mode == "duplicated"
        assert not candidate.optimizer.muon_split_qkv
        assert candidate.optimizer.muon_scalar_optimizer == "adam"
        assert not candidate.optimizer.use_distributed_optimizer
        assert not candidate.ddp.use_distributed_optimizer
        assert not candidate.ddp.overlap_param_gather
        assert candidate.ddp.grad_reduce_in_fp32
        assert candidate.scheduler.lr_warmup_iters == 5
        assert candidate.scheduler.start_weight_decay == candidate.scheduler.end_weight_decay == 0.0
        assert candidate.train.micro_batch_size == 1
        assert candidate.mixed_precision.grad_reduce_in_fp32
        assert not candidate.ddp.average_in_collective
        assert candidate.optimizer.main_grads_dtype == torch.float32
        assert candidate.optimizer.main_params_dtype == torch.float32
        assert candidate.optimizer.exp_avg_dtype == torch.float32
        assert candidate.optimizer.exp_avg_sq_dtype == torch.float32
        assert candidate.optimizer.lr == candidate.optimizer.min_lr == 1e-4
        assert candidate.optimizer.muon_num_ns_steps == 5
        assert candidate.optimizer.muon_scale_mode == "unit_rms_norm"
        assert candidate.optimizer.muon_extra_scale_factor == 0.2
        assert candidate.scheduler.lr_decay_style == "constant"
