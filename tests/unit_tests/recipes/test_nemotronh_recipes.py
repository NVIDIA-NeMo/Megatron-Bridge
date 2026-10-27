# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#
# Test purpose:
# - Parametrize over all exported NemotronH recipe functions in `megatron.bridge.recipes.nemotronh`.
# - Build each config without I/O and assert its batch, sequence, parallelism, tokenizer, and task contracts.
# - Verify flat CLI overrides reach the nested transformer config, and stale fields fail without creating
#   phantom config attributes.
#

import importlib
import re
from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.hybrid import BridgeHybridModelConfig
from megatron.bridge.training import checkpointing
from megatron.bridge.training.utils.omegaconf_utils import (
    OverridesError,
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
    process_config_with_overrides,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import (
    NemotronHBuilderOnlyBridge,
    NemotronHUltraBuilderOnlyBridge,
    discover_recipe_factories,
    patch_recipe_module_global,
    recipe_factory_id,
)


_nemotronh_module = importlib.import_module("megatron.bridge.recipes.nemotronh")
_NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret
_NEMOTRONH_RECIPE_FUNCS = [
    getattr(_nemotronh_module, name)
    for name in getattr(_nemotronh_module, "__all__", [])
    if callable(getattr(_nemotronh_module, name, None))
]
_NEMOTRONH_PERF_RECIPE_FUNCS = discover_recipe_factories(
    importlib.import_module("megatron.bridge.perf_recipes.nemotronh")
)


def test_nemotron_3_5_lightning_library_recipe_names_are_hardware_agnostic():
    """Library names describe behavior and do not collide with performance recipes."""
    perf_module = importlib.import_module("megatron.bridge.perf_recipes.nemotronh")
    library_names = {name for name in _nemotronh_module.__all__ if name.startswith("nemotron_3_5_lightning_")}
    perf_names = {name for name in dir(perf_module) if name.startswith("nemotron_3_5_lightning_")}

    assert all(re.search(r"(?:\d+gpu|h100|b200|b300|gb200|gb300)", name) is None for name in library_names)
    assert library_names.isdisjoint(perf_names)


@pytest.fixture(autouse=True)
def _patch_hf_backed_recipe_model_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build strict Nemotron-H model configs offline instead of reading Hugging Face configs."""
    for module_name in (
        "megatron.bridge.recipes.nemotronh.gb200.nemotron_3_nano",
        "megatron.bridge.recipes.nemotronh.gb200.nemotron_3_super",
        "megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano",
        "megatron.bridge.recipes.nemotronh.h100.nemotron_3_nano_4b",
        "megatron.bridge.recipes.nemotronh.h100.nemotron_3_super",
        "megatron.bridge.recipes.nemotronh.nemotron_3_super",
    ):
        module = importlib.import_module(module_name)
        patch_recipe_module_global(monkeypatch, module, "AutoBridge", NemotronHBuilderOnlyBridge)
    ultra_module = importlib.import_module("megatron.bridge.recipes.nemotronh.nemotron_3_ultra")
    patch_recipe_module_global(monkeypatch, ultra_module, "AutoBridge", NemotronHUltraBuilderOnlyBridge)


def _model_fields(model) -> dict:
    """Return the outer and nested transformer fields of a builder-backed model config."""
    return {**vars(model.transformer), **{key: value for key, value in vars(model).items() if key != "transformer"}}


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert isinstance(cfg.model, BridgeHybridModelConfig)
    assert cfg.train is not None
    assert cfg.validation is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.ddp is not None
    assert cfg.mixed_precision is not None

    assert cfg.train.micro_batch_size >= 1
    assert cfg.train.global_batch_size >= cfg.train.micro_batch_size
    assert cfg.train.global_batch_size % cfg.train.micro_batch_size == 0

    assert 1 <= cfg.dataset.seq_length <= cfg.model.seq_length

    assert cfg.model.tensor_model_parallel_size >= 1
    assert cfg.model.pipeline_model_parallel_size >= 1
    assert cfg.model.context_parallel_size >= 1
    assert cfg.model.expert_model_parallel_size >= 1


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_RECIPE_FUNCS)
def test_each_nemotronh_recipe_builds_config(recipe_func: Callable):
    """Test that each NemotronH recipe builds a valid config."""
    # All configs use parameterless API (peft configs have optional peft_scheme)
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer choice matches recipe type
    is_sft = "sft" in recipe_func.__name__.lower()
    is_peft = "peft" in recipe_func.__name__.lower()
    is_finetune = is_sft or is_peft

    if is_finetune:
        # Finetuning recipes always use HF tokenizer
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None
    else:
        # Pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
        if cfg.tokenizer.tokenizer_type == "NullTokenizer":
            assert cfg.tokenizer.vocab_size is not None
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model is not None

    if is_peft:
        assert cfg.peft is not None
    else:
        assert cfg.peft is None


@pytest.mark.parametrize("recipe_func", _NEMOTRONH_PERF_RECIPE_FUNCS, ids=recipe_factory_id)
def test_each_nemotronh_perf_recipe_uses_strict_builder_config(recipe_func: Callable):
    """Perf recipes only set fields that the builder-backed model config declares."""
    cfg = recipe_func()

    assert isinstance(cfg.model, BridgeHybridModelConfig)
    assert cfg.model.as_dict()["_builder_"] == BridgeHybridModelConfig.builder


def test_nemotronh_recipe_rejects_unknown_cli_override():
    """A stale recipe override must fail instead of creating a phantom field."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_config()

    with pytest.raises(OverridesError, match="Failed to parse Hydra overrides"):
        process_config_with_overrides(cfg, cli_overrides=["model.not_a_real_field=true"])

    assert not hasattr(cfg.model, "not_a_real_field")


# The overrides run_recipe.py emits for -tp/-pp/-cp/-vp/-ep/-etp.
_RUNNER_PARALLELISM_OVERRIDES = {
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 2,
    "context_parallel_size": 2,
    "virtual_pipeline_model_parallel_size": 2,
    "expert_model_parallel_size": 4,
    "expert_tensor_parallel_size": 2,
}


def _apply_overrides_like_example_scripts(cfg, cli_overrides: list[str]):
    """Apply CLI overrides as the Nemotron 3 example scripts that the Slurm launchers call do."""
    omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)
    omega_conf = parse_hydra_overrides(omega_conf, cli_overrides)
    apply_overrides(cfg, OmegaConf.to_container(omega_conf, resolve=True), excluded_fields)
    return cfg


@pytest.mark.parametrize(
    "apply_cli_overrides",
    [
        pytest.param(lambda cfg, overrides: process_config_with_overrides(cfg, cli_overrides=overrides), id="runner"),
        pytest.param(_apply_overrides_like_example_scripts, id="example-scripts"),
    ],
)
def test_nemotronh_builder_accepts_existing_flat_cli_overrides(apply_cli_overrides):
    """Flat transformer overrides reach the nested owner; the model's own fields stay outer."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_config()
    overrides = [f"model.{name}={value}" for name, value in _RUNNER_PARALLELISM_OVERRIDES.items()]

    apply_cli_overrides(
        cfg,
        [*overrides, "model.sequence_parallel=true", "model.mamba_chunk_size=256", "model.seq_length=1024"],
    )

    for name, value in _RUNNER_PARALLELISM_OVERRIDES.items():
        assert getattr(cfg.model.transformer, name) == value, name
    assert cfg.model.transformer.sequence_parallel is True
    assert (cfg.model.mamba_chunk_size, cfg.model.seq_length) == (256, 1024)
    assert set(vars(cfg.model)).isdisjoint(_RUNNER_PARALLELISM_OVERRIDES)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["model.transformer.tensor_model_parallel_size=4", "model.tensor_model_parallel_size=2"], 2),
        (["model.tensor_model_parallel_size=2", "model.transformer.tensor_model_parallel_size=4"], 4),
    ],
)
def test_nemotronh_flat_and_nested_cli_overrides_apply_in_order(overrides, expected):
    """Both spellings name one field, so the last override wins."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_config()

    process_config_with_overrides(cfg, cli_overrides=overrides)

    assert cfg.model.transformer.tensor_model_parallel_size == expected


def test_nemotronh_builder_accepts_hf_source_override():
    """The text-only Super VL card points the recipe at a local HF export with model.hf_model_id."""
    cfg = _nemotronh_module.nemotron_3_super_sft_config()

    process_config_with_overrides(
        cfg, cli_overrides=["model.hf_model_id=work/hf", "tokenizer.tokenizer_model=work/hf"]
    )

    assert cfg.model.extra_checkpoint_metadata["hf_model_id"] == "work/hf"
    assert checkpointing._resolve_hf_source(cfg) == "work/hf"


def test_super_builder_sidecar_resolves_model_source_before_tokenizer(monkeypatch, tmp_path):
    """HF sidecar exports template from the recorded Super source, not the recipe's Nano tokenizer repository."""
    cfg = _nemotronh_module.nemotron_3_super_pretrain_config()
    peft_cfg = _nemotronh_module.nemotron_3_super_peft_config()
    super_id = cfg.model.extra_checkpoint_metadata["hf_model_id"]
    assert "Super" in super_id
    assert "Nano" in cfg.tokenizer.tokenizer_model
    assert checkpointing._resolve_hf_source(cfg) == super_id

    monkeypatch.setattr(checkpointing, "_AUTO_BRIDGE_CACHE", {})
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    bridge = Mock()
    factory = Mock(return_value=bridge)
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", factory)

    checkpointing._save_hf_weights(SimpleNamespace(cfg=cfg), [], str(tmp_path))
    factory.assert_called_once_with(super_id, trust_remote_code=False)
    bridge.hf_pretrained.save_artifacts.assert_called_once_with(str(tmp_path), original_source_path=super_id)

    checkpointing._save_hf_adapter_weights(SimpleNamespace(cfg=peft_cfg), [], str(tmp_path))
    assert bridge.save_hf_adapter.call_args.kwargs["base_model_name_or_path"] == super_id


def test_nemotron_3_nano_gb200_defers_vocab_size_to_training_tokenizer():
    """The GB200 pretraining model vocabulary must follow its runtime tokenizer."""
    cfg = _nemotronh_module.nemotron_3_nano_pretrain_8gpu_gb200_bf16_config()

    assert cfg.model.vocab_size is None


@pytest.mark.parametrize(
    ("module_name", "factory_name", "has_comm_overlap", "expected_preprocessing_sms"),
    [
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
            "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config",
            True,
            108,
            id="nano-gb200",
        ),
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
            "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config",
            True,
            108,
            id="nano-gb300",
        ),
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.vrnvl72.nemotronh",
            "nemotron_3_nano_pretrain_8gpu_vrnvl72_fp8mx_config",
            True,
            108,
            id="nano-vrnvl72",
        ),
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh",
            "nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config",
            False,
            32,
            id="super-gb200",
        ),
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.gb300.nemotronh",
            "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config",
            False,
            32,
            id="super-gb300",
        ),
        pytest.param(
            "megatron.bridge.perf_recipes.nemotronh.vrnvl72.nemotronh",
            "nemotron_3_super_pretrain_64gpu_vrnvl72_fp8mx_config",
            False,
            32,
            id="super-vrnvl72",
        ),
    ],
)
def test_nemotron_3_mxfp8_perf_recipes_enable_cutedsl_fusion(
    module_name: str,
    factory_name: str,
    has_comm_overlap: bool,
    expected_preprocessing_sms: int,
) -> None:
    """Selected MXFP8 recipes match the measured CutDSL and MoE overlap settings."""
    module = importlib.import_module(module_name)
    cfg = getattr(module, factory_name)()

    assert cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert cfg.env_vars["CUDNNFE_CLUSTER_OVERLAP_MARGIN"] == 8
    assert cfg.model.use_transformer_engine_op_fuser is True
    assert cfg.model.moe_mlp_glu_interleave_size == 32
    assert cfg.model.high_priority_a2a_comm_stream is False
    assert cfg.model.moe_hybridep_num_sms_preprocessing == expected_preprocessing_sms
    assert cfg.mixed_precision.fp8_dot_product_attention is True
    if has_comm_overlap:
        assert cfg.comm_overlap is not None
        assert cfg.comm_overlap.overlap_moe_expert_parallel_comm is False
        assert cfg.comm_overlap.delay_wgrad_compute is False
    else:
        assert cfg.comm_overlap is None


def test_nemotron_3_super_64gpu_gb200_matches_benchmark_hardware_configuration():
    """The training recipe should share the tuned GB200 layout without benchmark-only behavior."""
    from megatron.bridge.perf_recipes.nemotronh.gb200.nemotronh import (
        nemotron_3_super_pretrain_64gpu_gb200_bf16_config,
    )

    training_cfg = _nemotronh_module.nemotron_3_super_pretrain_64gpu_gb200_bf16_config()
    benchmark_cfg = nemotron_3_super_pretrain_64gpu_gb200_bf16_config()

    for field_name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "expert_tensor_parallel_size",
        "expert_model_parallel_size",
        "moe_flex_dispatcher_backend",
        "moe_hybridep_num_sms",
        "moe_token_dispatcher_type",
        "moe_shared_expert_overlap",
        "cuda_graph_impl",
        "apply_rope_fusion",
    ):
        assert getattr(training_cfg.model, field_name) == getattr(benchmark_cfg.model, field_name)

    assert cuda_graph_module_names(training_cfg.model) == cuda_graph_module_names(benchmark_cfg.model)

    assert training_cfg.train.global_batch_size == benchmark_cfg.train.global_batch_size == 512
    assert training_cfg.train.micro_batch_size == benchmark_cfg.train.micro_batch_size == 1
    assert training_cfg.env_vars == benchmark_cfg.env_vars

    assert training_cfg.model.moe_router_force_load_balancing is False
    assert benchmark_cfg.model.moe_router_force_load_balancing is True
    assert training_cfg.train.train_iters == 39735
    assert benchmark_cfg.train.train_iters == 50
    assert training_cfg.checkpoint.async_save is True
    assert benchmark_cfg.checkpoint.async_save is False
    assert training_cfg.ddp.check_for_nan_in_grad is True
    assert benchmark_cfg.ddp.check_for_nan_in_grad is False
    assert training_cfg.ddp.overlap_grad_reduce is True
    assert training_cfg.ddp.overlap_param_gather is True
    assert training_cfg.model.recompute_granularity is None
    assert training_cfg.optimizer.optimizer_cpu_offload is False
    assert training_cfg.optimizer.optimizer_offload_fraction == 0.0

    # The GB200 recipe derives from the memory-bounded H100 support config, so it
    # must restore overlapped collectives and full-precision optimizer state.
    assert training_cfg.optimizer.use_precision_aware_optimizer is False
    assert training_cfg.optimizer.main_params_dtype == torch.float32
    assert training_cfg.optimizer.exp_avg_dtype == torch.float32
    assert training_cfg.optimizer.exp_avg_sq_dtype == torch.float32


def test_nemotron_3_super_64gpu_h100_matches_benchmark_execution_configuration():
    """The H100 convergence and benchmark recipes should share their execution mapping."""
    from megatron.bridge.perf_recipes.nemotronh.h100.nemotronh import (
        nemotron_3_super_pretrain_64gpu_h100_bf16_config,
    )

    training_cfg = _nemotronh_module.nemotron_3_super_pretrain_config()
    benchmark_cfg = nemotron_3_super_pretrain_64gpu_h100_bf16_config()

    for field_name in (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "context_parallel_size",
        "virtual_pipeline_model_parallel_size",
        "expert_tensor_parallel_size",
        "expert_model_parallel_size",
        "sequence_parallel",
        "overlap_p2p_comm",
        "batch_p2p_comm",
        "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
        "moe_flex_dispatcher_num_sms",
        "moe_hybridep_num_sms",
        "moe_hybridep_pad_uneven_dispatch_inputs",
        "moe_shared_expert_overlap",
        "cuda_graph_impl",
        "apply_rope_fusion",
        "recompute_granularity",
        "recompute_modules",
    ):
        assert getattr(training_cfg.model, field_name) == getattr(benchmark_cfg.model, field_name)

    assert training_cfg.model.seq_length == benchmark_cfg.model.seq_length == 4096
    assert training_cfg.dataset.seq_length == benchmark_cfg.dataset.seq_length == 4096
    assert training_cfg.train.global_batch_size == benchmark_cfg.train.global_batch_size == 1280
    assert training_cfg.train.micro_batch_size == benchmark_cfg.train.micro_batch_size == 1
    assert training_cfg.ddp.overlap_grad_reduce == benchmark_cfg.ddp.overlap_grad_reduce is False
    assert training_cfg.ddp.overlap_param_gather == benchmark_cfg.ddp.overlap_param_gather is False
    assert training_cfg.dist.distributed_timeout_minutes == benchmark_cfg.dist.distributed_timeout_minutes == 15
    assert training_cfg.model.moe_hybridep_pad_uneven_dispatch_inputs is False
    assert training_cfg.model.moe_expert_capacity_factor == benchmark_cfg.model.moe_expert_capacity_factor == 1.10
    assert (
        training_cfg.model.moe_pad_expert_input_to_capacity
        is benchmark_cfg.model.moe_pad_expert_input_to_capacity
        is True
    )
    assert training_cfg.model.recompute_modules == ["layernorm", "moe_act", "moe", "core_attn"]
    assert training_cfg.model.apply_rope_fusion is benchmark_cfg.model.apply_rope_fusion is True
    assert training_cfg.tokenizer.use_tokenizer_vocab_size is benchmark_cfg.tokenizer.use_tokenizer_vocab_size is False
    assert training_cfg.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert training_cfg.optimizer.use_precision_aware_optimizer is True
    assert benchmark_cfg.optimizer.use_precision_aware_optimizer is True
    assert training_cfg.optimizer.main_grads_dtype == benchmark_cfg.optimizer.main_grads_dtype == torch.bfloat16
    assert training_cfg.optimizer.main_params_dtype == benchmark_cfg.optimizer.main_params_dtype == torch.float16
    assert training_cfg.optimizer.exp_avg_dtype == benchmark_cfg.optimizer.exp_avg_dtype == torch.bfloat16
    assert training_cfg.optimizer.exp_avg_sq_dtype == benchmark_cfg.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert training_cfg.optimizer.optimizer_cpu_offload is benchmark_cfg.optimizer.optimizer_cpu_offload is False
    assert (
        training_cfg.optimizer.optimizer_offload_fraction == benchmark_cfg.optimizer.optimizer_offload_fraction == 0.0
    )
    assert (
        training_cfg.optimizer.overlap_cpu_optimizer_d2h_h2d
        is benchmark_cfg.optimizer.overlap_cpu_optimizer_d2h_h2d
        is False
    )
    assert training_cfg.env_vars == benchmark_cfg.env_vars
    assert training_cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32

    assert training_cfg.model.moe_router_force_load_balancing is False
    assert benchmark_cfg.model.moe_router_force_load_balancing is True
    assert training_cfg.train.train_iters == 100
    assert benchmark_cfg.train.train_iters == 50
    assert training_cfg.ddp.check_for_nan_in_grad is True
    assert benchmark_cfg.ddp.check_for_nan_in_grad is False
    assert training_cfg.checkpoint.save is not None
    assert benchmark_cfg.checkpoint.save is None


@pytest.mark.parametrize(
    "recipe_name",
    [
        "nemotron_3_super_pretrain_64gpu_b200_bf16_config",
        "nemotron_3_super_pretrain_64gpu_b200_fp8mx_config",
        "nemotron_3_super_pretrain_64gpu_b200_nvfp4_config",
    ],
)
def test_nemotron_3_super_64gpu_b200_uses_single_nvlink_domain_ep(recipe_name):
    """B200 BF16, MXFP8, and NVFP4 recipes keep HybridEP within one NVLink domain."""
    module = importlib.import_module("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh")
    cfg = getattr(module, recipe_name)()

    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8


def test_nemotron_3_super_64gpu_b200_fp8mx_uses_selective_recompute():
    """B200 MXFP8 recomputes activations to fit EP8."""
    module = importlib.import_module("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh")
    cfg = module.nemotron_3_super_pretrain_64gpu_b200_fp8mx_config()

    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe_act", "moe", "layernorm", "core_attn"]
    assert cfg.model.cuda_graph_impl == "none"


def test_nemotron_3_super_64gpu_b200_nvfp4_keeps_cuda_graphs_with_light_recompute():
    """B200 NVFP4 retains CUDA graphs with graph-compatible recompute."""
    module = importlib.import_module("megatron.bridge.perf_recipes.nemotronh.b200.nemotronh")
    cfg = module.nemotron_3_super_pretrain_64gpu_b200_nvfp4_config()

    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe_act", "layernorm"]
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cfg.model.cuda_graph_scope == ["mamba", "attn", "moe_router", "moe_preprocess"]


def test_nemotron_3_5_lightning_h100_convergence_recipe_uses_perf_execution_policy():
    """The H100 convergence recipe keeps safety checks while using the measured fast path."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_pretrain_config

    cfg = nemotron_3_5_lightning_pretrain_config()

    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 1

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.cross_entropy_fusion_impl == "native"
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["mamba"]
    assert cfg.model.recompute_granularity == "selective"
    assert cfg.model.recompute_modules == ["moe", "layernorm", "core_attn"]

    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp16 is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.optimizer.use_precision_aware_optimizer is False
    assert cfg.optimizer.main_grads_dtype == torch.float32
    assert cfg.optimizer.main_params_dtype == torch.float32
    assert cfg.optimizer.exp_avg_dtype == torch.float32
    assert cfg.optimizer.exp_avg_sq_dtype == torch.float32
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.check_for_large_grads is True
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.checkpoint.save_interval == 200
    assert cfg.checkpoint.async_save is False

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.model.extra_checkpoint_metadata["hf_model_id"] == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert cfg.model.extra_checkpoint_metadata["hf_model_revision"] == _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    assert cfg.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert cfg.env_vars["USE_MNNVL"] == 0


def test_nemotron_3_5_lightning_8k_convergence_recipe_uses_perf_execution_policy():
    """The 8K convergence recipe keeps safety checks while using the measured fast path."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_5_lightning_pretrain_8k_config
    from megatron.bridge.utils.cuda_graph import cuda_graph_module_names

    cfg = nemotron_3_5_lightning_pretrain_8k_config()

    assert cfg.model.seq_length == 8192
    assert cfg.dataset.seq_length == 8192
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 2

    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.cross_entropy_fusion_impl == "te"
    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None

    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp16 is False
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.check_for_nan_in_grad is True
    assert cfg.ddp.check_for_large_grads is True
    assert cfg.ddp.average_in_collective is False
    assert cfg.rerun_state_machine.check_for_nan_in_loss is True
    assert cfg.checkpoint.save_interval == 50
    assert cfg.checkpoint.async_save is False

    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
    assert cfg.model.extra_checkpoint_metadata["hf_model_id"] == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert cfg.model.extra_checkpoint_metadata["hf_model_revision"] == _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    assert cfg.tokenizer.tokenizer_model == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def test_nemotron_3_5_lightning_h100_and_gb200_differ_only_in_execution_policy():
    """H100 and GB200 must share one training contract."""
    from megatron.bridge.recipes.nemotronh import (
        nemotron_3_5_lightning_pretrain_8k_config,
        nemotron_3_5_lightning_pretrain_config,
    )

    h100 = nemotron_3_5_lightning_pretrain_config()
    gb200 = nemotron_3_5_lightning_pretrain_8k_config()

    # Normalize the explicit GB200 execution/performance overrides. Any other
    # difference, including a convergence-setting drift, must fail this test.
    gb200.train.micro_batch_size = h100.train.micro_batch_size
    gb200.model.context_parallel_size = h100.model.context_parallel_size
    gb200.model.cp_comm_type = h100.model.cp_comm_type
    gb200.model.cross_entropy_fusion_impl = h100.model.cross_entropy_fusion_impl
    gb200.model.cuda_graph_impl = h100.model.cuda_graph_impl
    gb200.model.cuda_graph_modules = h100.model.cuda_graph_modules
    gb200.model.recompute_granularity = h100.model.recompute_granularity
    gb200.model.recompute_modules = h100.model.recompute_modules
    gb200.model.recompute_method = h100.model.recompute_method
    gb200.model.recompute_num_layers = h100.model.recompute_num_layers
    gb200.ddp.average_in_collective = h100.ddp.average_in_collective
    gb200.checkpoint.save_interval = h100.checkpoint.save_interval
    gb200.env_vars = h100.env_vars

    assert gb200 == h100


def test_nemotron_3_5_lightning_8k_fsdp_recipe_preserves_convergence_contract():
    """The BF16 FSDP recipe changes only FSDP-required execution settings."""
    from megatron.bridge.recipes.nemotronh import (
        nemotron_3_5_lightning_pretrain_8k_config,
        nemotron_3_5_lightning_pretrain_8k_fsdp_config,
    )
    from megatron.bridge.utils.cuda_graph import cuda_graph_module_names

    reference = nemotron_3_5_lightning_pretrain_8k_config()
    cfg = nemotron_3_5_lightning_pretrain_8k_fsdp_config()

    assert cfg.model.seq_length == reference.model.seq_length == 8192
    assert cfg.dataset.seq_length == reference.dataset.seq_length == 8192
    assert cfg.train.global_batch_size == reference.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == reference.train.micro_batch_size == 2
    assert cfg.model.moe_router_force_load_balancing is reference.model.moe_router_force_load_balancing is False
    assert cfg.optimizer == reference.optimizer
    assert cfg.scheduler == reference.scheduler
    assert cfg.mixed_precision == reference.mixed_precision
    assert cfg.rng == reference.rng

    assert cfg.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(cfg.model) == []
    assert getattr(cfg.model, "init_model_with_meta_device", False) is False
    assert getattr(reference.model, "init_model_with_meta_device", False) is False
    assert cfg.ddp.use_megatron_fsdp is True
    assert cfg.ddp.num_distributed_optimizer_instances == 1
    assert cfg.ddp.data_parallel_sharding_strategy == "optim_grads_params"
    assert cfg.ddp.outer_dp_sharding_strategy == "no_shard"
    assert cfg.ddp.average_in_collective is False
    assert cfg.ddp.megatron_fsdp_main_params_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
    assert cfg.ddp.megatron_fsdp_grad_comm_dtype == torch.bfloat16
    assert cfg.checkpoint.load is None
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"


def test_nemotron_3_5_lightning_openmath_sft_tp1_recipe_uses_tuned_defaults():
    """The TP1 packed SFT recipe owns the measured topology and run contract."""
    cfg = _nemotronh_module.nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config()

    assert cfg.model.seq_length == 4096
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.moe_flex_dispatcher_backend == "hybridep"
    assert cfg.model.moe_hybridep_num_sms == 32
    assert cfg.model.recompute_granularity is None
    assert cfg.model.recompute_modules is None
    assert cfg.model.extra_checkpoint_metadata["hf_model_id"] == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    assert cfg.model.extra_checkpoint_metadata["hf_model_revision"] == _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    assert cfg.tokenizer.hf_tokenizer_kwargs == {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}

    assert cfg.dataset.seq_length == 4096
    assert cfg.dataset.hf_dataset.dataset_name == "openmathinstruct2"
    expected_revision = "469216e3f46f4dacf476b382e192485ea51a143e"  # pragma: allowlist secret
    assert cfg.dataset.hf_dataset.load_kwargs == {"revision": expected_revision}
    assert cfg.dataset.enable_offline_packing is True
    assert cfg.dataset.offline_packing_specs.packed_sequence_size == 4096
    assert cfg.dataset.offline_packing_specs.pad_seq_to_mult == 2
    assert cfg.dataset.offline_packing_specs.tokenizer_model_name == (
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    )

    assert cfg.train.train_iters == 100
    assert cfg.train.global_batch_size == 128
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.empty_unused_memory_level == 0
    assert cfg.mixed_precision.grad_reduce_in_fp32 is False
    assert cfg.ddp.grad_reduce_in_fp32 is False
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.optimizer.overlap_param_gather is True
    assert cfg.checkpoint.async_save is False

    assert cfg.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert cfg.env_vars["USE_MNNVL"] == 1


def test_nemotron_3_5_lightning_h100_and_gb200_sft_differ_only_in_execution_policy():
    """H100 and GB200 packed SFT must share one training contract."""
    from megatron.bridge.recipes.nemotronh import (
        nemotron_3_5_lightning_sft_openmathinstruct2_packed_config,
        nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config,
    )

    h100 = nemotron_3_5_lightning_sft_openmathinstruct2_packed_config()
    gb200 = nemotron_3_5_lightning_sft_openmathinstruct2_packed_tp1_config()

    # Remove the explicit GB200 execution/performance model overrides. Any
    # other difference, including a convergence-setting drift, must fail.
    model_execution_fields = {
        "tensor_model_parallel_size",
        "sequence_parallel",
        "moe_flex_dispatcher_backend",
        "moe_flex_dispatcher_num_sms",
        "moe_hybridep_num_sms",
        "recompute_granularity",
        "recompute_modules",
        "recompute_method",
        "recompute_num_layers",
    }
    h100_model = {key: value for key, value in _model_fields(h100.model).items() if key not in model_execution_fields}
    gb200_model = {
        key: value for key, value in _model_fields(gb200.model).items() if key not in model_execution_fields
    }
    assert gb200_model == h100_model

    gb200.model = h100.model
    gb200.train.empty_unused_memory_level = h100.train.empty_unused_memory_level
    gb200.ddp.overlap_param_gather = h100.ddp.overlap_param_gather
    gb200.optimizer.overlap_param_gather = h100.optimizer.overlap_param_gather
    gb200.env_vars = h100.env_vars

    assert gb200 == h100


# --- Nemotron 3 Super tests ---


def test_nemotron_3_super_pretrain_defaults():
    """Test that Nemotron 3 Super pretrain has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_pretrain_config

    cfg = nemotron_3_super_pretrain_config()

    _assert_basic_config(cfg)

    # Pretrain uses the measured 64-H100 TP-free layout.
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 2
    assert cfg.model.sequence_parallel is False
    assert cfg.model.expert_model_parallel_size == 32


def test_nemotron_3_super_peft_lora_defaults():
    """Test that Nemotron 3 Super PEFT with LoRA has correct default parallelism."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_peft_config

    cfg = nemotron_3_super_peft_config()

    _assert_basic_config(cfg)

    # For LoRA, should use TP=1, PP=1
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True

    # Check PEFT config
    assert cfg.peft is not None
    assert cfg.peft.target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]


def test_nemotron_3_super_sft_defaults():
    """Test that Nemotron 3 Super SFT has correct defaults."""
    from megatron.bridge.recipes.nemotronh import nemotron_3_super_sft_config

    cfg = nemotron_3_super_sft_config()

    _assert_basic_config(cfg)

    # Full SFT uses the checkpoint-safe TP8/EP16 layout.
    assert cfg.model.tensor_model_parallel_size == 8
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.expert_model_parallel_size == 16
    assert cfg.peft is None
