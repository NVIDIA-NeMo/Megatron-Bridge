# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused tests for the Inkling2 GB200 recipes.

The recipes run against the real ``Inkling2ModelConfig`` that the bridge builds
from an Inkling2 Small configuration, so assigning an undeclared model field fails
here instead of creating a phantom attribute.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Callable

import pytest
import torch

from megatron.bridge.data.builders.direct_hf_sft import DirectHFSFTDatasetConfig
from megatron.bridge.data.sft_processing import ChatSFTPreprocessingConfig
from megatron.bridge.models.inkling import Inkling2Bridge, Inkling2Config, Inkling2ModelConfig
from megatron.bridge.models.inkling.data.processor import Inkling2Processor
from megatron.bridge.models.inkling.data.tokenizer import Inkling2MegatronTokenizer
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.inkling.gb200 import inkling2 as inkling2_recipes
from megatron.bridge.recipes.inkling.gb200 import (
    inkling2_small_peft_16gpu_gb200_bf16_config,
    inkling2_small_peft_32gpu_gb200_bf16_config,
    inkling2_small_peft_32gpu_gb200_bf16_multimodal_config,
    inkling2_small_pretrain_128gpu_gb200_bf16_config,
    inkling2_small_sft_96gpu_gb200_bf16_config,
    inkling2_small_sft_128gpu_gb200_bf16_config,
    inkling2_small_sft_128gpu_gb200_bf16_long_context_config,
    inkling2_small_sft_128gpu_gb200_bf16_multimodal_config,
)
from megatron.bridge.training.config import ConfigContainer
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

_NUM_LAYERS = 48
_FULL_ATTENTION_LAYERS = tuple(range(5, _NUM_LAYERS, 6))
_SMALL_TEXT_CONFIG = {
    "hidden_size": 4608,
    "num_hidden_layers": _NUM_LAYERS,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "d_rel": 16,
    "rel_extent": 1024,
    "log_scaling_n_floor": 8192,
    "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-6,
    "pre_norm_pre_affine": False,
    "final_norm_pre_affine": False,
    "use_value_norm_no_gamma": True,
    "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [i for i in range(_NUM_LAYERS) if i not in _FULL_ATTENTION_LAYERS],
    "sconv_kernel_size": 4,
    "final_logit_softcapping": 30.0,
    "sliding_window_size": 1024,
    "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": True,
    "n_routed_experts": 384,
    "num_experts_per_tok": 8,
    "n_shared_experts": 0,
    "intermediate_size": 2304,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "route_scale": 8.0,
    "use_gate_bias": True,
    "torch_dtype": "bfloat16",
}


class _ModelConfigBridge:
    """Stand-in for ``AutoBridge`` that returns the real Inkling2 model config."""

    @classmethod
    def from_hf_pretrained(cls, model_id: str, **kwargs: object) -> "_ModelConfigBridge":
        del kwargs
        assert model_id == inkling2_recipes._MODEL_ID
        return cls()

    def get_model_config(self) -> Inkling2ModelConfig:
        hf_config = Inkling2Config(
            text_config=dict(_SMALL_TEXT_CONFIG),
            vision_config={"hidden_size": 1024, "num_hidden_layers": 23, "native_patch_size": 20},
            architectures=["Inkling2ForConditionalGeneration"],
        )
        bridge = Inkling2Bridge()
        bridge.hf_config = hf_config
        return bridge.hf_config_to_model_config(hf_config)


@pytest.fixture(autouse=True)
def _offline_recipe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)
    monkeypatch.setattr(inkling2_recipes, "AutoBridge", _ModelConfigBridge)


_RECIPES: dict[str, tuple[Callable[[], ConfigContainer], int]] = {
    "pretrain": (inkling2_small_pretrain_128gpu_gb200_bf16_config, 128),
    "sft": (inkling2_small_sft_128gpu_gb200_bf16_config, 128),
    "sft_96gpu": (inkling2_small_sft_96gpu_gb200_bf16_config, 96),
    "sft_long_context": (inkling2_small_sft_128gpu_gb200_bf16_long_context_config, 128),
    "peft": (inkling2_small_peft_32gpu_gb200_bf16_config, 32),
    "peft_16gpu": (inkling2_small_peft_16gpu_gb200_bf16_config, 16),
}

# Settings a recipe variant may change without changing what it trains: the parallel layout, activation
# recompute, the MoE token transport, and fusions. Data, batch, optimizer, schedule, precision, and routing match.
_EXECUTION_MODEL_FIELDS = frozenset(
    {
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "sequence_parallel",
        "hybrid_layer_pattern",
        "recompute_granularity",
        "recompute_method",
        "recompute_num_layers",
        "recompute_modules",
        "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
        "moe_flex_dispatcher_num_sms",
        "moe_permute_fusion_into_hybridep",
        "moe_hybridep_pad_uneven_dispatch_inputs",
        "moe_router_fusion",
    }
)


def _model_settings(cfg: ConfigContainer) -> dict[str, object]:
    model = cfg.model
    settings = {field.name: getattr(model, field.name) for field in dataclasses.fields(model)}
    del settings["transformer"]
    settings.update(
        {field.name: getattr(model.transformer, field.name) for field in dataclasses.fields(model.transformer)}
    )
    return settings


def _assert_execution_only_changes(variant: ConfigContainer, reference: ConfigContainer) -> None:
    """Assert that ``variant`` trains exactly what ``reference`` trains, possibly on another layout."""
    variant_model, reference_model = _model_settings(variant), _model_settings(reference)
    changed = {name for name, value in variant_model.items() if value != reference_model[name]}
    assert not changed - _EXECUTION_MODEL_FIELDS, sorted(changed - _EXECUTION_MODEL_FIELDS)
    for section in ("dataset", "tokenizer", "optimizer", "scheduler", "mixed_precision", "rng", "peft", "checkpoint"):
        assert getattr(variant, section) == getattr(reference, section), section
    variant_train, reference_train = dataclasses.asdict(variant.train), dataclasses.asdict(reference.train)
    # The micro-batch size is an execution setting while the global batch size stays fixed.
    del variant_train["micro_batch_size"], reference_train["micro_batch_size"]
    assert variant_train == reference_train


def _assert_hybridep_dispatch(cfg: ConfigContainer) -> None:
    """Assert that ``cfg`` dispatches its experts with HybridEP inside one NVL72 NVLink domain."""
    model = cfg.model
    assert (model.moe_token_dispatcher_type, model.moe_flex_dispatcher_backend) == ("flex", "hybridep")
    assert model.moe_flex_dispatcher_num_sms == 32
    assert model.moe_permute_fusion_into_hybridep is True
    # HybridEP needs equal dispatch shapes: packed SFT data pads every rank to the group maximum, and unpacked
    # samples are already padded to the full sequence length.
    assert model.moe_hybridep_pad_uneven_dispatch_inputs is bool(getattr(cfg.dataset, "enable_offline_packing", False))
    # HybridEP sizes its NVLink-domain communication from the launch environment.
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == model.expert_model_parallel_size
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def _assert_compiled_ops_and_sparse_gc(cfg: ConfigContainer) -> None:
    """Assert that ``cfg`` compiles the Inkling2 elementwise operators and collects garbage every 100 steps."""
    assert cfg.model.inkling2_compile_ops is True
    assert cfg.train.manual_gc is True
    assert cfg.train.manual_gc_interval == 100


def _data_parallel_size(cfg: ConfigContainer, world_size: int) -> int:
    model = cfg.model
    dense_model_parallel = (
        model.tensor_model_parallel_size * model.pipeline_model_parallel_size * model.context_parallel_size
    )
    assert world_size % dense_model_parallel == 0
    return world_size // dense_model_parallel


def _expert_data_parallel_size(cfg: ConfigContainer, world_size: int) -> int:
    model = cfg.model
    expert_model_parallel = (
        model.pipeline_model_parallel_size * model.expert_model_parallel_size * model.expert_tensor_parallel_size
    )
    assert world_size % expert_model_parallel == 0
    return world_size // expert_model_parallel


@pytest.mark.parametrize("recipe_name", sorted(_RECIPES))
def test_recipe_layout_is_valid_for_inkling2_small(recipe_name: str) -> None:
    recipe, world_size = _RECIPES[recipe_name]
    cfg = recipe()

    assert isinstance(cfg.model, Inkling2ModelConfig)
    # The model config rejects hierarchical context parallelism, virtual-pipeline parallelism, CUDA graphs,
    # EP overlap, and FP8.
    cfg.model.finalize()
    model = cfg.model
    # Only the 32K recipe splits its packs over context-parallel ranks.
    assert model.context_parallel_size == (4 if recipe_name == "sft_long_context" else 1)
    assert model.virtual_pipeline_model_parallel_size is None
    # Every pipeline stage holds whole decoder blocks: one attention and one MoE Hybrid layer each.
    stages = model.hybrid_layer_pattern.split("|")
    assert len(stages) == model.pipeline_model_parallel_size
    assert all(stage and stage == "*E" * (len(stage) // 2) for stage in stages)
    assert sum(len(stage) for stage in stages) == model.num_layers
    assert model.num_moe_experts % model.expert_model_parallel_size == 0
    assert model.num_query_groups % model.tensor_model_parallel_size == 0
    assert model.sequence_parallel is (model.tensor_model_parallel_size > 1)
    assert model.pipeline_dtype is torch.bfloat16
    assert model.expert_tensor_parallel_size == 1
    # Each expert-parallel group stays inside one 72-GPU NVLink domain.
    assert model.expert_model_parallel_size <= 72
    _assert_hybridep_dispatch(cfg)
    _assert_compiled_ops_and_sparse_gc(cfg)
    assert model.moe_router_fusion is False
    # Every recipe recomputes whole layers, except 128-GPU SFT and 32-GPU LoRA, whose single pass fits without it.
    assert model.recompute_granularity == (None if recipe_name in ("sft", "peft") else "full")
    assert model.freeze_language_model is False
    assert model.freeze_vision_model is True
    assert model.freeze_vision_projection is True
    assert cfg.tokenizer.tokenizer_model == inkling2_recipes._MODEL_ID
    # Megatron-Core loads the Hugging Face tokenizer as the class that this metadata names.
    metadata = cfg.tokenizer.metadata_path
    assert metadata["library"] == "huggingface"
    assert (
        getattr(importlib.import_module(metadata["class_path"]), metadata["class_name"]) is Inkling2MegatronTokenizer
    )
    assert cfg.train.global_batch_size % (_data_parallel_size(cfg, world_size) * cfg.train.micro_batch_size) == 0
    assert _expert_data_parallel_size(cfg, world_size) >= 1
    assert cfg.checkpoint.load is None
    assert cfg.validation.eval_interval == 0


def test_pretrain_recipe_contract() -> None:
    cfg = inkling2_small_pretrain_128gpu_gb200_bf16_config()

    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (1, 4)
    # Later blocks cost more and the last stage also computes the loss, so the earlier stages hold more blocks.
    assert [len(stage) // 2 for stage in model.hybrid_layer_pattern.split("|")] == [13, 13, 12, 10]
    assert model.expert_model_parallel_size == 32
    assert (model.moe_token_dispatcher_type, model.moe_flex_dispatcher_backend) == ("flex", "hybridep")
    assert model.seq_length == cfg.dataset.seq_length == 4096
    assert _data_parallel_size(cfg, 128) == 32
    assert (cfg.ddp.overlap_grad_reduce, cfg.ddp.overlap_param_gather) == (False, False)
    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 1024
    assert cfg.optimizer.lr == pytest.approx(3e-5)
    assert cfg.optimizer.min_lr == pytest.approx(3e-6)
    assert cfg.scheduler.lr_warmup_iters == 40
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.checkpoint.save_interval == 50


def _assert_tulu3_packed_contract(cfg: ConfigContainer, *, seq_length: int, output_root: str) -> None:
    # The tokenizer renders with the checkpoint's own chat_template.jinja.
    assert cfg.tokenizer.chat_template is None
    assert cfg.model.seq_length == cfg.dataset.seq_length == seq_length
    assert cfg.dataset.hf_dataset.split == "train[:10000]"
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": inkling2_recipes._TULU3_REVISION}
    assert cfg.dataset.hf_output_root == output_root
    assert cfg.dataset.max_train_samples == 10000
    assert cfg.dataset.seed == 1234
    assert cfg.dataset.do_validation is False
    assert cfg.dataset.enable_offline_packing is True
    # Fixed-width packs keep one compiled FlexAttention shape.
    assert cfg.dataset.dataset_kwargs == {"pad_to_max_length": True}
    specs = cfg.dataset.offline_packing_specs
    assert specs.packed_sequence_size == seq_length
    # Context parallelism cuts every sequence into 2 x CP chunks, and sequence parallelism splits each CP rank's
    # share across the TP ranks.
    model = cfg.model
    cp_multiple = 2 * model.context_parallel_size if model.context_parallel_size > 1 else 1
    sp_multiple = model.context_parallel_size * model.tensor_model_parallel_size if model.sequence_parallel else 1
    assert specs.pad_seq_to_mult == math.lcm(cp_multiple, sp_multiple)
    assert specs.max_single_sequence_length == seq_length - specs.pad_seq_to_mult
    assert cfg.rng.seed == 5678
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.optimizer.min_lr == 0.0
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.optimizer.use_precision_aware_optimizer is False


def test_sft_recipe_contract() -> None:
    cfg = inkling2_small_sft_128gpu_gb200_bf16_config()

    _assert_tulu3_packed_contract(cfg, seq_length=8192, output_root="work/data/inkling2/tulu3-10k-8k")
    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 4)
    assert model.expert_model_parallel_size == 32
    assert _data_parallel_size(cfg, 128) == 8
    # The activations of a single pass through four pipeline stages fit without recompute.
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == (None, None, None)
    assert cfg.peft is None
    assert cfg.train.train_iters == 100
    assert cfg.optimizer.lr == pytest.approx(5e-6)
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    # The final step writes the full model weights for HF export.
    assert cfg.checkpoint.save_interval == 100
    assert cfg.checkpoint.save_optim is False
    assert cfg.checkpoint.save_rng is False


def test_sft_96gpu_recipe_trains_like_the_128gpu_recipe() -> None:
    cfg = inkling2_small_sft_96gpu_gb200_bf16_config()

    _assert_execution_only_changes(cfg, inkling2_small_sft_128gpu_gb200_bf16_config())
    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 3)
    assert model.expert_model_parallel_size == 32
    # Eight replicas take one pack each, so a step is a single pass through the decoder.
    assert _data_parallel_size(cfg, 96) == cfg.train.global_batch_size == 8
    assert [len(stage) // 2 for stage in model.hybrid_layer_pattern.split("|")] == [16, 16, 16]
    # Recomputing the first 20 of each stage's 32 Hybrid layers bounds the last stage's activations.
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == ("full", "block", 20)


def test_long_context_sft_recipe_contract() -> None:
    cfg = inkling2_small_sft_128gpu_gb200_bf16_long_context_config()

    _assert_tulu3_packed_contract(cfg, seq_length=32768, output_root="work/data/inkling2/tulu3-10k-32k-pad16")
    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 4)
    assert model.expert_model_parallel_size == 32
    assert model.context_parallel_size == 4
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 16
    # Two data-parallel replicas each run four packs per step.
    assert _data_parallel_size(cfg, 128) == 2
    # The first stage holds four packs in flight; recomputing the first 20 of each stage's 24 layers bounds it.
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == ("full", "block", 20)
    # Context-parallel SFT needs the token-weighted loss reduction.
    assert model.calculate_per_token_loss is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.peft is None
    assert cfg.train.train_iters == 20
    assert cfg.optimizer.lr == pytest.approx(1e-6)
    assert cfg.scheduler.lr_warmup_iters == 2
    assert cfg.scheduler.lr_decay_iters == 20
    assert cfg.checkpoint.save_interval is None


def test_peft_recipe_contract() -> None:
    cfg = inkling2_small_peft_32gpu_gb200_bf16_config()

    _assert_tulu3_packed_contract(cfg, seq_length=8192, output_root="work/data/inkling2/tulu3-10k-8k")
    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 1)
    assert model.expert_model_parallel_size == 32
    # Eight replicas take one pack each, so a step is a single pass through the decoder.
    assert _data_parallel_size(cfg, 32) == cfg.train.global_batch_size == 8
    # The frozen base's activations for that single pass fit without recompute.
    assert (model.recompute_granularity, model.recompute_method, model.recompute_num_layers) == (None, None, None)
    assert (model.moe_token_dispatcher_type, model.moe_flex_dispatcher_backend) == ("flex", "hybridep")
    assert isinstance(cfg.peft, LoRA)
    # Inkling2 attention keeps separate Q, K, and V projections instead of a fused linear_qkv.
    assert cfg.peft.target_modules == ["linear_q", "linear_k", "linear_v", "linear_proj"]
    assert (cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout) == (8, 16, 0.0)
    assert cfg.train.train_iters == 100
    assert cfg.optimizer.lr == pytest.approx(1e-4)
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.checkpoint.save_interval == 100


def test_peft_16gpu_recipe_trains_like_the_32gpu_recipe() -> None:
    cfg = inkling2_small_peft_16gpu_gb200_bf16_config()

    _assert_execution_only_changes(cfg, inkling2_small_peft_32gpu_gb200_bf16_config())
    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (2, 1)
    assert model.expert_model_parallel_size == 16
    # Eight replicas take one pack each, so a step is a single pass through the decoder.
    assert _data_parallel_size(cfg, 16) == cfg.train.global_batch_size == 8
    assert model.recompute_granularity == "full"


_MULTIMODAL_RECIPES: dict[str, tuple[Callable[[], ConfigContainer], int]] = {
    "peft_multimodal": (inkling2_small_peft_32gpu_gb200_bf16_multimodal_config, 32),
    "sft_multimodal": (inkling2_small_sft_128gpu_gb200_bf16_multimodal_config, 128),
}


@pytest.mark.parametrize("recipe_name", sorted(_MULTIMODAL_RECIPES))
def test_multimodal_recipe_contract(recipe_name: str) -> None:
    recipe, world_size = _MULTIMODAL_RECIPES[recipe_name]
    cfg = recipe()

    cfg.model.finalize()
    model = cfg.model
    stages = model.hybrid_layer_pattern.split("|")
    assert len(stages) == model.pipeline_model_parallel_size
    assert all(stage and stage == "*E" * (len(stage) // 2) for stage in stages)
    assert model.context_parallel_size == 1
    assert model.sequence_parallel is True
    _assert_hybridep_dispatch(cfg)
    _assert_compiled_ops_and_sparse_gc(cfg)
    # Full SFT recomputes whole layers; LoRA's frozen base fits without recompute.
    assert model.recompute_granularity == ("full" if recipe_name == "sft_multimodal" else None)
    assert model.freeze_language_model is False
    # CORD-v2 receipts through the checkpoint's Inkling2 processor, one padded 4K width.
    dataset = cfg.dataset
    assert isinstance(dataset, DirectHFSFTDatasetConfig)
    assert dataset.hf_processor_path == inkling2_recipes._MODEL_ID
    # Bridge's processor, not the one AutoProcessor resolves from Transformers 5.23 on.
    module_name, _, class_name = dataset.hf_processor_class.rpartition(".")
    assert getattr(importlib.import_module(module_name), class_name) is Inkling2Processor
    assert dataset.hf_processor_kwargs == {"max_image_tokens": 3072}
    assert dataset.source.dataset_name == "cord_v2"
    assert dataset.source.split == "train"
    assert dataset.source.load_kwargs == {"revision": inkling2_recipes._CORD_V2_REVISION}
    assert isinstance(dataset.preprocessing, ChatSFTPreprocessingConfig)
    assert dataset.preprocessing.loss_mode == "assistant"
    assert model.seq_length == dataset.seq_length == 4096
    assert dataset.pad_to_max_length is True
    assert dataset.enable_in_batch_packing is False
    assert dataset.do_validation is False and dataset.do_test is False
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size % _data_parallel_size(cfg, world_size) == 0
    assert cfg.train.train_iters == 100
    assert cfg.scheduler.lr_warmup_iters == 10
    assert cfg.scheduler.lr_decay_iters == 100
    assert cfg.rng.seed == 5678
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.save_interval == 100
    assert cfg.validation.eval_interval == 0


def test_multimodal_sft_recipe_trains_the_image_encoder() -> None:
    cfg = inkling2_small_sft_128gpu_gb200_bf16_multimodal_config()

    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 4)
    assert model.expert_model_parallel_size == 32
    assert model.freeze_vision_model is False
    assert model.freeze_vision_projection is False
    assert cfg.peft is None
    assert _data_parallel_size(cfg, 128) == 8
    assert cfg.train.global_batch_size == 64
    assert cfg.optimizer.lr == pytest.approx(5e-6)
    assert cfg.checkpoint.save_optim is False


def test_multimodal_peft_recipe_keeps_the_base_frozen() -> None:
    cfg = inkling2_small_peft_32gpu_gb200_bf16_multimodal_config()

    model = cfg.model
    assert (model.tensor_model_parallel_size, model.pipeline_model_parallel_size) == (4, 2)
    assert model.expert_model_parallel_size == 16
    assert model.freeze_vision_model is True
    assert model.freeze_vision_projection is True
    assert isinstance(cfg.peft, LoRA)
    assert cfg.peft.target_modules == ["linear_q", "linear_k", "linear_v", "linear_proj"]
    assert (cfg.peft.dim, cfg.peft.alpha, cfg.peft.dropout) == (8, 16, 0.0)
    assert _data_parallel_size(cfg, 32) == 4
    assert cfg.train.global_batch_size == 8
    assert cfg.optimizer.lr == pytest.approx(1e-4)
