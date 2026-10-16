# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Focused tests for Muse Spark Open verification recipes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.muse_spark_open.gb200 import muse_spark_open as muse_spark_recipes
from megatron.bridge.recipes.muse_spark_open.gb200 import (
    muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config,
    muse_spark_open_peft_64gpu_gb200_bf16_config,
    muse_spark_open_pretrain_192gpu_gb200_bf16_config,
    muse_spark_open_pretrain_256gpu_gb200_bf16_config,
    muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config,
    muse_spark_open_sft_192gpu_gb200_bf16_config,
    muse_spark_open_sft_192gpu_gb200_bf16_long_context_config,
    muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config,
    muse_spark_open_sft_256gpu_gb200_bf16_config,
)
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def test_muse_spark_multimodal_peft_recipe_contract() -> None:
    cfg = muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    # Transformer Engine rejects P2P context parallelism for sliding-window layers.
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.sequence_parallel is True
    assert cfg.model.recompute_granularity is None
    # 16 GPUs cannot hold EP64; the EP16 group is the whole job inside one NVL72 domain.
    _assert_hybridep(cfg, ep=16, pad_uneven=False)
    assert cfg.model.seq_length == 4096
    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.hf_processor_path == muse_spark_recipes._MODEL_ID
    assert cfg.dataset.hf_processor_kwargs == {"max_image_tokens": 256}
    assert cfg.dataset.source.load_kwargs == {
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret
    }
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.dataset.enable_in_batch_packing is False
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.validation.eval_iters == 0
    assert cfg.validation.eval_interval == 0
    assert cfg.logger.log_throughput is True
    assert cfg.checkpoint.load is None


def test_muse_spark_lora_targets_native_attention_projections() -> None:
    cfg = muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config()

    assert isinstance(cfg.peft, LoRA)
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert cfg.peft.dim == 8
    assert cfg.peft.alpha == 16
    assert cfg.peft.dropout == 0.0
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == pytest.approx(1e-4)
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100


def test_muse_spark_multimodal_sft_recipe_contract_and_balanced_stages() -> None:
    cfg = muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config()

    assert isinstance(cfg, ConfigContainer)
    assert cfg.peft is None
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 3
    assert cfg.model.pipeline_dtype is torch.bfloat16
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.sequence_parallel is True
    _assert_three_rack_stages(cfg)
    assert cfg.model.recompute_granularity is None
    # Fixed-length image-text samples give every EP rank the same token count, so no padding.
    _assert_hybridep(cfg, ep=64, pad_uneven=False)
    assert _data_parallel_size(cfg, 192) == 16
    assert cfg.train.global_batch_size % _data_parallel_size(cfg, 192) == 0
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is False
    assert cfg.model.freeze_audio_model is True
    assert cfg.model.freeze_vision_projection is False
    assert cfg.model.seq_length == 4096
    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.hf_processor_kwargs == {"max_image_tokens": 256}
    assert cfg.dataset.pad_to_max_length is True
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 64
    assert cfg.train.micro_batch_size == 1
    assert cfg.rng.seed == 5678
    assert cfg.optimizer.lr == pytest.approx(5e-6)
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.save_optim is False
    assert cfg.checkpoint.load is None


# The 8K contract shared with other models' cards: Tulu 3 train[:10000], 8K packs, GBS 8, 100 steps.
_TULU3_8K = {"split": "train[:10000]", "seq_length": 8192, "max_train_samples": 10000, "steps": 100, "gbs": 8}
# One epoch of the first 100,000 Tulu 3 conversations: 10,951 4K packs, 85 full batches of 128.
_TULU3_100K = {"split": "train[:100000]", "seq_length": 4096, "max_train_samples": 85 * 128, "steps": 85, "gbs": 128}


def _assert_tulu3_packed_contract(
    cfg: ConfigContainer,
    *,
    pad_seq_to_mult: int,
    split: str,
    seq_length: int,
    max_train_samples: int,
    steps: int,
    gbs: int,
) -> None:
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == muse_spark_recipes._MODEL_ID
    assert cfg.model.seq_length == seq_length
    assert cfg.dataset.seq_length == seq_length
    assert cfg.dataset.hf_dataset.split == split
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": muse_spark_recipes._TULU3_REVISION}
    assert cfg.dataset.max_train_samples == max_train_samples
    assert cfg.dataset.seed == 1234
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == seq_length
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == pad_seq_to_mult
    assert cfg.dataset.offline_packing_specs.max_single_sequence_length == seq_length - pad_seq_to_mult
    assert cfg.dataset.do_validation is False
    assert cfg.rng.seed == 5678
    assert cfg.train.train_iters == steps
    assert cfg.train.global_batch_size == gbs
    assert cfg.train.micro_batch_size == 1
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == steps
    assert cfg.checkpoint.save_interval == steps
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.optimizer.use_precision_aware_optimizer is False
    # Text-only data never exercises the perception towers.
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_audio_model is True
    assert cfg.model.freeze_vision_projection is True


def _assert_hybridep(cfg: ConfigContainer, *, ep: int, pad_uneven: bool) -> None:
    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.model.moe_shared_expert_overlap is False
    assert getattr(cfg.model, "moe_hybridep_pad_uneven_dispatch_inputs", False) is pad_uneven
    # Every EP group is one pipeline stage inside one NVL72 domain, so its HybridEP rank count is the EP size.
    assert cfg.model.expert_model_parallel_size == ep
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == ep
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def _data_parallel_size(cfg: ConfigContainer, world_size: int) -> int:
    model = cfg.model
    return world_size // (
        model.tensor_model_parallel_size * model.pipeline_model_parallel_size * model.context_parallel_size
    )


def _stage_blocks(cfg: ConfigContainer) -> list[int]:
    stages = cfg.model.hybrid_layer_pattern.split("|")
    assert all(stage == "*E" * (len(stage) // 2) for stage in stages)
    return [len(stage) // 2 for stage in stages]


def _assert_three_rack_stages(cfg: ConfigContainer) -> None:
    # 192 GPUs at EP64: one pipeline stage per 64-GPU NVL72 domain. The last stage gives up blocks for the
    # 202K-vocabulary output layer.
    assert cfg.model.pipeline_model_parallel_size == 3
    assert _stage_blocks(cfg) == [21, 22, 19]
    assert cfg.model.virtual_pipeline_model_parallel_size is None


def test_muse_spark_sft_recipe_owns_tulu3_contract() -> None:
    cfg = muse_spark_open_sft_192gpu_gb200_bf16_config()

    assert cfg.peft is None
    # Sequence parallelism over TP4 without CP aligns each sequence to a multiple of four.
    _assert_tulu3_packed_contract(cfg, pad_seq_to_mult=4, **_TULU3_100K)
    assert cfg.optimizer.lr == pytest.approx(5e-6)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.model.context_parallel_size == 1
    _assert_three_rack_stages(cfg)
    assert cfg.model.recompute_granularity is None
    _assert_hybridep(cfg, ep=64, pad_uneven=True)
    assert _data_parallel_size(cfg, 192) == 16
    assert cfg.train.global_batch_size % _data_parallel_size(cfg, 192) == 0
    assert cfg.checkpoint.save_optim is False


def test_muse_spark_peft_recipe_owns_tulu3_contract() -> None:
    cfg = muse_spark_open_peft_64gpu_gb200_bf16_config()

    assert isinstance(cfg.peft, LoRA)
    # Attention projections, the routed experts, and the latent MoE projections.
    assert cfg.peft.target_modules == [
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "fc1_latent_proj",
        "fc2_latent_proj",
    ]
    assert (cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout) == (32, 32, 0.0)
    # One adapter per expert, without rank normalization or the shared-outer layout.
    assert cfg.peft.share_expert_adapters is False
    assert cfg.peft.normalize_moe_lora is False
    assert cfg.peft.experts_shared_outer_loras is False
    _assert_tulu3_packed_contract(cfg, pad_seq_to_mult=4, **_TULU3_100K)
    assert cfg.optimizer.lr == pytest.approx(1e-4)
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.pipeline_dtype is None
    assert cfg.model.recompute_granularity is None
    _assert_hybridep(cfg, ep=64, pad_uneven=True)
    assert _data_parallel_size(cfg, 64) == 16
    assert cfg.train.global_batch_size % _data_parallel_size(cfg, 64) == 0


def test_muse_spark_peft_recipe_keeps_dora_on_attention_projections() -> None:
    cfg = muse_spark_open_peft_64gpu_gb200_bf16_config(peft_scheme="dora")

    assert not isinstance(cfg.peft, LoRA)
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj"]
    assert (cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout) == (8, 16, 0.0)
    assert not hasattr(cfg.peft, "share_expert_adapters")


def test_muse_spark_pretrain_recipe_contract() -> None:
    cfg = muse_spark_open_pretrain_192gpu_gb200_bf16_config()
    sft = muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config()

    assert isinstance(cfg, ConfigContainer)
    assert isinstance(cfg.dataset, GPTDatasetConfig)
    assert cfg.peft is None
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.tokenizer.tokenizer_model == muse_spark_recipes._MODEL_ID
    for name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "pipeline_dtype",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "sequence_parallel",
        "hybrid_layer_pattern",
        "recompute_granularity",
    ):
        assert getattr(cfg.model, name) == getattr(sft.model, name)
    # Text-only data never exercises the perception towers.
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_audio_model is True
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.seq_length == 4096
    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.random_seed == 1234
    assert cfg.rng.seed == 1234
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 1024
    assert cfg.train.micro_batch_size == 1
    # A from-scratch 3e-4 peak degrades the warm-started model; see the recipe docstring.
    assert cfg.optimizer.lr == pytest.approx(3e-5)
    assert cfg.optimizer.min_lr == pytest.approx(3e-6)
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.validation.eval_iters == 0
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.load is None
    # The card command supplies the imported checkpoint; the reusable recipe must not.
    assert cfg.checkpoint.pretrained_checkpoint is None
    _assert_three_rack_stages(cfg)
    assert cfg.model.recompute_granularity is None
    assert _data_parallel_size(cfg, 192) == 16
    # Fixed-length sequences give every EP rank the same token count, so no padding.
    _assert_hybridep(cfg, ep=64, pad_uneven=False)


def test_muse_spark_long_context_sft_packs_32k_with_cp2() -> None:
    reference = muse_spark_open_sft_192gpu_gb200_bf16_config()
    cfg = muse_spark_open_sft_192gpu_gb200_bf16_long_context_config()

    assert cfg.model.seq_length == 32768
    assert cfg.dataset.seq_length == 32768
    assert cfg.model.context_parallel_size == 2
    # Transformer Engine rejects P2P context parallelism for sliding-window layers.
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.sequence_parallel is True
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 32768
    # lcm(2 * CP, TP * CP) = lcm(4, 8)
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 8
    assert cfg.dataset.offline_packing_specs.max_single_sequence_length == 32768 - 8
    assert cfg.dataset.hf_dataset.split == "train[:10000]"
    for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "hybrid_layer_pattern"):
        assert getattr(cfg.model, name) == getattr(reference.model, name)
    # 4K tokens per GPU: only whole-MoE selective recompute keeps the first stage within memory.
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe"]
    _assert_hybridep(cfg, ep=64, pad_uneven=True)
    data_parallel = 192 // (
        cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size
    )
    assert data_parallel == 8
    assert cfg.train.global_batch_size == 8
    assert cfg.train.global_batch_size % data_parallel == 0
    assert cfg.train.train_iters == 20
    assert cfg.optimizer.lr == pytest.approx(1e-6)
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.scheduler.lr_warmup_iters == 2
    assert cfg.scheduler.lr_decay_iters == 20
    assert cfg.checkpoint.save_interval == 20


def test_muse_spark_coderforge_sft_packs_64k_with_cp4() -> None:
    cfg = muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config()

    assert cfg.model.seq_length == 65536
    assert cfg.dataset.seq_length == 65536
    assert cfg.model.context_parallel_size == 4
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    # lcm(2 * CP, TP * CP) = lcm(8, 16)
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 16
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 65536
    assert cfg.dataset.offline_packing_specs.max_single_sequence_length == 65536 - 16
    assert cfg.dataset.hf_dataset.split == "SWE_Rebench[:600]"
    assert cfg.dataset.hf_dataset.load_kwargs == {
        "revision": muse_spark_recipes._CODERFORGE_REVISION,
        "data_files": {"SWE_Rebench": "trajectories/SWE_Rebench-*.parquet"},
        "verification_mode": "no_checks",
    }
    _assert_three_rack_stages(cfg)
    # 4K tokens per GPU: only whole-MoE selective recompute keeps the first stage within memory.
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe"]
    _assert_hybridep(cfg, ep=64, pad_uneven=True)
    data_parallel = 192 // (
        cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size * cfg.model.context_parallel_size
    )
    assert data_parallel == 4
    assert cfg.train.global_batch_size % data_parallel == 0
    assert cfg.train.train_iters == 20


def test_muse_spark_pretrain_256gpu_recipe_changes_only_the_pipeline_split() -> None:
    reference = muse_spark_open_pretrain_192gpu_gb200_bf16_config()
    cfg = muse_spark_open_pretrain_256gpu_gb200_bf16_config()

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    # One pipeline stage per 64-GPU NVL72 domain; the last stage gives up blocks for the output layer.
    assert cfg.model.pipeline_model_parallel_size == 4
    assert _stage_blocks(cfg) == [16, 16, 16, 14]
    # 256 GPUs / (TP4 * PP4) leaves 16 data-parallel replicas; EP64 shards each stage's experts once.
    assert _data_parallel_size(cfg, 256) == 16
    _assert_hybridep(cfg, ep=64, pad_uneven=False)
    for section, names in (
        ("train", ("train_iters", "global_batch_size", "micro_batch_size")),
        ("optimizer", ("lr", "min_lr", "use_precision_aware_optimizer")),
        ("scheduler", ("lr_warmup_iters", "lr_decay_iters")),
        ("rng", ("seed",)),
        ("ddp", ("grad_reduce_in_fp32",)),
    ):
        for name in names:
            assert getattr(getattr(cfg, section), name) == getattr(getattr(reference, section), name)
    for name in ("freeze_vision_model", "freeze_audio_model", "recompute_granularity", "seq_length"):
        assert getattr(cfg.model, name) == getattr(reference.model, name)
    assert cfg.checkpoint.pretrained_checkpoint is None


def test_muse_spark_sft_256gpu_recipe_keeps_the_8k_tulu3_contract() -> None:
    reference = muse_spark_open_sft_192gpu_gb200_bf16_config()
    cfg = muse_spark_open_sft_256gpu_gb200_bf16_config()

    assert cfg.peft is None
    # lcm(2 * CP, TP * CP) = lcm(4, 8)
    _assert_tulu3_packed_contract(cfg, pad_seq_to_mult=8, **_TULU3_8K)
    assert cfg.dataset.hf_output_root != reference.dataset.hf_output_root

    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.sequence_parallel is True
    assert cfg.model.context_parallel_size == 2
    assert cfg.model.cp_comm_type == "a2a"
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.pipeline_dtype is torch.bfloat16
    assert _stage_blocks(cfg) == [16, 16, 16, 14]
    _assert_hybridep(cfg, ep=64, pad_uneven=True)
    # 256 GPUs / (TP4 * CP2 * PP4) leaves eight data-parallel replicas: one packed sequence each at GBS 8.
    assert _data_parallel_size(cfg, 256) == 8
    # The optimizer, seeds, and frozen towers match the 192-GPU recipe; only data, batch, and horizon differ.
    for section, names in (
        ("train", ("micro_batch_size",)),
        ("optimizer", ("lr", "min_lr", "use_precision_aware_optimizer")),
        ("scheduler", ("lr_warmup_iters",)),
        ("rng", ("seed",)),
        ("ddp", ("grad_reduce_in_fp32",)),
        ("checkpoint", ("save_optim", "save_rng", "pretrained_checkpoint")),
    ):
        for name in names:
            assert getattr(getattr(cfg, section), name) == getattr(getattr(reference, section), name)
    for name in ("freeze_vision_model", "freeze_audio_model", "recompute_granularity"):
        assert getattr(cfg.model, name) == getattr(reference.model, name)


def test_muse_spark_recipe_uses_builder_model_config_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bridge:
        def get_model_config(self):
            return SimpleNamespace()

        def to_megatron_provider(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("legacy GPT provider path used")

    class _AutoBridge:
        @staticmethod
        def from_hf_pretrained(*args, **kwargs):
            del args, kwargs
            return _Bridge()

    monkeypatch.setattr(muse_spark_recipes, "AutoBridge", _AutoBridge)

    cfg = muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config()

    assert cfg.model is not None


@pytest.mark.parametrize("recipe_name", muse_spark_recipes.__all__)
def test_muse_spark_recipes_keep_each_hybridep_group_in_one_nvlink_domain(recipe_name: str) -> None:
    cfg = getattr(muse_spark_recipes, recipe_name)()

    ep = cfg.model.expert_model_parallel_size
    # An NVL72 rack holds 16 nodes of 4 GPUs; HybridEP needs the EP group inside one NVLink domain.
    assert ep in (16, 64)
    _assert_hybridep(cfg, ep=ep, pad_uneven=getattr(cfg.model, "moe_hybridep_pad_uneven_dispatch_inputs", False))
    # No recipe falls back to full activation recompute.
    assert cfg.model.recompute_granularity != "full"


@pytest.mark.parametrize("recipe_name", muse_spark_recipes.__all__)
def test_muse_spark_recipes_enable_megatron_core_fusions(recipe_name: str) -> None:
    cfg = getattr(muse_spark_recipes, recipe_name)()

    # The model config leaves these off; every recipe turns them on.
    for name in (
        "bias_activation_fusion",
        "gradient_accumulation_fusion",
        "apply_rope_fusion",
        "moe_router_fusion",
        "moe_permute_fusion_into_hybridep",
        "deallocate_pipeline_outputs",
    ):
        assert getattr(cfg.model, name) is True, name


@pytest.mark.parametrize("recipe_name", muse_spark_recipes.__all__)
def test_muse_spark_packed_recipes_use_fixed_width_packs(recipe_name: str) -> None:
    cfg = getattr(muse_spark_recipes, recipe_name)()
    if not getattr(cfg.dataset, "enable_offline_packing", False):
        pytest.skip("not an offline-packed recipe")

    # Equal token counts on every rank keep the compiled fusions from recompiling on one rank at a time.
    assert cfg.dataset.dataset_kwargs == {"pad_to_max_length": True}
