# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from megatron.bridge.recipes.qwen.gb200 import qwen35


pytestmark = pytest.mark.unit
_BF16 = qwen35.qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_bf16_config
_MXFP8 = qwen35.qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_fp8mx_config
_RECIPES = [_BF16, _MXFP8]


@pytest.fixture
def hf_config(monkeypatch):
    config = AutoConfig.for_model(
        "qwen3_5_moe",
        text_config={
            "num_hidden_layers": 4,
            "hidden_size": 512,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "num_experts": 32,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "linear_num_key_heads": 8,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "tie_word_embeddings": True,
        },
        tie_word_embeddings=False,
    )
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return config

    monkeypatch.setattr(qwen35.AutoConfig, "from_pretrained", load)
    return config, calls


@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_text_long_context_validates_and_preserves_inputs(hf_config, monkeypatch, recipe_func):
    config, calls = hf_config
    original = deepcopy(config.to_dict())
    training_module = importlib.import_module("megatron.bridge.training.config")
    monkeypatch.setattr(training_module, "get_world_size_safe", lambda: 16)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: SimpleNamespace(major=10, name="NVIDIA GB200")
    )

    cfg = recipe_func()
    # Mirror runtime_config_update: mixed precision sets the parameter dtype before validation.
    cfg.mixed_precision.finalize()
    cfg.mixed_precision.setup(cfg.model, cfg.optimizer, cfg.ddp)
    cfg.validate()

    assert config.to_dict() == original
    assert calls == [("Qwen/Qwen3.5-35B-A3B", {"revision": qwen35._QWEN35_35B_A3B_INSTRUCT_REVISION})]
    assert cfg.tokenizer.hf_tokenizer_kwargs["revision"] == calls[0][1]["revision"]
    assert cfg.model.share_embeddings_and_output_weights is False
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.mtp_loss_scaling_factor == 0.1
    assert cfg.model.context_parallel_size == 8
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.recompute_modules == ["gdn_norm_out", "moe"]
    assert {
        "gdn_kernel_backend",
        "gdn_pre_gated_delta_rule_fusion",
        "gdn_gated_output_norm_fusion",
    } <= {field.name for field in fields(cfg.model)}
    assert cfg.model.gdn_kernel_backend == "transformer_engine"
    assert cfg.model.gdn_pre_gated_delta_rule_fusion is True
    assert cfg.model.gdn_gated_output_norm_fusion is False
    assert not any(name.startswith("vision_recompute_") for name in dir(cfg.model))
    assert cfg.model.apply_rope_fusion is False
    assert cfg.model.cuda_graph_impl == "none"
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.dataset.hf_dataset.dataset_name == "coderforge"
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": qwen35._CODERFORGE_REVISION}
    assert cfg.dataset.dataset_root is None
    assert cfg.dataset.hf_validation_proportion == 0.05
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 16
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 131072
    assert cfg.dataset.dataset_kwargs["pad_to_max_length"] is True
    assert cfg.dataset.seed == 1234
    assert cfg.dataset.persistent_workers is True
    assert cfg.train.global_batch_size == 32
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.train_iters == 500
    assert cfg.scheduler.lr_warmup_iters == 200
    assert cfg.scheduler.lr_decay_iters == 300000
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False
    assert cfg.optimizer.overlap_param_gather is False
    assert cfg.optimizer.overlap_param_gather_with_optimizer_step is False
    assert cfg.comm_overlap is None
    assert cfg.model.overlap_moe_expert_parallel_comm is False
    assert cfg.model.delay_wgrad_compute is False


def test_verified_mxfp8_kernel_and_parameter_settings(hf_config):
    cfg = _MXFP8()
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.checkpoint.load_main_params_from_ckpt is True
    assert cfg.checkpoint.load_optim is False
    assert cfg.checkpoint.load_rng is False
    assert cfg.model.moe_use_grouped_tensor is True
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.moe_single_grouped_weight is True
    assert cfg.model.moe_single_grouped_bias is False
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 16
    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL"] == 1
    assert cfg.env_vars["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == 1


def test_bf16_recipe_has_no_mxfp8_kernel_or_checkpoint_requirements(hf_config):
    cfg = _BF16()
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.mixed_precision.fp8_param_gather is False
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is False
    assert cfg.checkpoint.load_main_params_from_ckpt is False
    assert cfg.model.moe_use_grouped_tensor is False
    assert cfg.model.use_transformer_engine_op_fuser is False
    assert cfg.model.moe_mlp_glu_interleave_size is None
    assert cfg.model.moe_single_grouped_weight is False
    for name in (
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL",
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
    ):
        assert name not in cfg.env_vars


@pytest.mark.parametrize("seq_length", [0, -16, 15, 131073])
@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_invalid_sequence_length_fails_before_hub_access(hf_config, seq_length, recipe_func):
    _, calls = hf_config
    with pytest.raises(ValueError, match="positive and divisible by 16"):
        recipe_func(seq_length)
    assert calls == []


@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_local_text_config_uses_matching_tokenizer(hf_config, monkeypatch, recipe_func):
    config, _ = hf_config
    monkeypatch.setattr(qwen35.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: config.text_config)

    cfg = recipe_func(1024, hf_path="local-text-checkpoint", hf_revision=None)

    assert cfg.tokenizer.tokenizer_model == "local-text-checkpoint"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": None}
    assert cfg.model.seq_length == 1024
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 1024


@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_wrong_model_type_is_rejected(hf_config, recipe_func):
    config, _ = hf_config
    config.text_config.model_type = "qwen3_5_text"
    with pytest.raises(ValueError, match="Expected a Qwen3.5 MoE"):
        recipe_func()


@pytest.mark.parametrize("seq_length", [16, 32768, 65536, 131072])
@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_aligned_context_lengths(hf_config, seq_length, recipe_func):
    cfg = recipe_func(seq_length)
    assert cfg.model.seq_length == cfg.dataset.seq_length == seq_length
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == seq_length


@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_context_exceeding_model_limit_is_rejected(hf_config, recipe_func):
    config, _ = hf_config
    with pytest.raises(ValueError, match="max_position_embeddings"):
        recipe_func(config.text_config.max_position_embeddings + 16)


@pytest.mark.parametrize("recipe_func", _RECIPES)
def test_public_runner_finds_unique_text_recipe(hf_config, monkeypatch, recipe_func):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/training"))
    runner = importlib.import_module("recipe_runner")
    assert runner.find_library_recipe(recipe_func.__name__) is recipe_func
    assert runner.find_benchmark_recipe(recipe_func.__name__) is None
    cfg = runner.load_recipe(recipe_func.__name__)
    assert cfg.model.mtp_num_layers == 1
    assert cfg.dataset.hf_dataset.dataset_name == "coderforge"
