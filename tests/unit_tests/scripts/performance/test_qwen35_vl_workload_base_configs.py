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

"""Tests for Qwen3.5-VL performance workload presets."""

import dataclasses
import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.transformer_block import get_num_layers_to_build

from megatron.bridge.perf_recipes.qwen_vl.common import _select_gdn_kernel_backend
from megatron.bridge.perf_recipes.qwen_vl.gb200.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen_vl.gb300.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.qwen_vl.h100.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config,
    qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
    qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
)
from megatron.bridge.perf_recipes.qwen_vl.vr200.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8mx_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8cs_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8mx_config,
)
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names
from tests.unit_tests.recipes.recipe_test_utils import (
    _OfflineModelProvider,
    discover_recipe_factories,
    patch_recipe_construction_dependencies,
    recipe_factory_id,
)


pytestmark = pytest.mark.unit

_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))


@pytest.mark.parametrize(
    ("recipe_fn", "expected_pp_size", "expected_vp_size"),
    [
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config,
            8,
            2,
        ),
        (
            qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config,
            8,
            2,
        ),
    ],
)
def test_qwen35_vl_122b_h100_pipeline_layout(
    recipe_fn: Callable,
    expected_pp_size: int,
    expected_vp_size: int,
) -> None:
    num_layers = 48
    config = recipe_fn()
    pp_size = config.model.pipeline_model_parallel_size
    vp_size = config.model.virtual_pipeline_model_parallel_size

    assert num_layers % pp_size == 0
    assert vp_size is not None
    assert (num_layers // pp_size) % vp_size == 0
    assert pp_size == expected_pp_size
    assert vp_size == expected_vp_size


def test_qwen35_vl_35b_h100_fp8cs_pipeline_layout_builds_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 35B FP8-CS benchmark topology should allocate all 40 language layers."""
    patch_recipe_construction_dependencies(monkeypatch)
    config = qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config()
    model = config.model
    allocation_config = SimpleNamespace(
        pipeline_model_parallel_layout=getattr(model, "pipeline_model_parallel_layout", None),
        num_layers_in_first_pipeline_stage=getattr(model, "num_layers_in_first_pipeline_stage", None),
        num_layers_in_last_pipeline_stage=getattr(model, "num_layers_in_last_pipeline_stage", None),
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        num_layers=40,
        pipeline_model_parallel_size=model.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=model.virtual_pipeline_model_parallel_size,
    )

    allocated_layers = sum(
        get_num_layers_to_build(allocation_config, vp_stage=vp_stage, pp_rank=pp_rank)
        for pp_rank in range(model.pipeline_model_parallel_size)
        for vp_stage in range(model.virtual_pipeline_model_parallel_size)
    )

    assert allocated_layers == 40


def test_qwen35_vl_35b_h100_measured_performance_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The H100 factory should retain the measured 185-TFLOP execution policy."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config()

    assert config.model.tensor_model_parallel_size == 1
    assert config.model.pipeline_model_parallel_size == 2
    assert config.model.context_parallel_size == 1
    assert config.model.virtual_pipeline_model_parallel_size is None
    assert config.model.num_layers_in_first_pipeline_stage == 17
    assert config.model.num_layers_in_last_pipeline_stage == 23
    assert config.model.expert_model_parallel_size == 8
    assert config.model.expert_tensor_parallel_size == 1
    assert config.model.sequence_parallel is False
    assert config.train.micro_batch_size == 1
    assert config.train.global_batch_size == 512

    assert config.model.moe_token_dispatcher_type == "flex"
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.model.moe_flex_dispatcher_num_sms == 16
    assert config.model.moe_hybridep_num_sms is None
    assert config.model.moe_hybridep_num_sms_preprocessing == 16
    assert config.model.moe_permute_fusion is True
    assert config.model.moe_permute_fusion_into_hybridep is True
    assert config.model.moe_router_force_load_balancing is True
    assert config.model.moe_shared_expert_overlap is False
    assert config.model.overlap_dispatch_backward_with_experts_wgrad is True

    assert config.model.recompute_granularity is None
    assert config.model.recompute_method is None
    assert config.model.recompute_num_layers is None
    assert config.model.recompute_modules == []
    assert config.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(config.model) == ["attn", "moe_router", "moe_preprocess"]
    assert config.model.vision_cuda_graph_impl == "transformer_engine"
    assert config.model.vision_cuda_graph_scope == ["attn", "mlp"]
    assert config.model.max_vision_cuda_graph_seq_length == 784
    assert config.model.use_te_rng_tracker is True
    assert config.rng.te_rng_tracker is True

    assert config.optimizer.use_precision_aware_optimizer is True
    assert config.optimizer.main_params_dtype == torch.float32
    assert config.optimizer.main_grads_dtype == torch.float32
    assert config.optimizer.exp_avg_dtype == torch.bfloat16
    assert config.optimizer.exp_avg_sq_dtype == torch.bfloat16
    assert config.optimizer.overlap_param_gather is False
    assert config.optimizer.overlap_param_gather_with_optimizer_step is False
    assert config.ddp.overlap_grad_reduce is False
    assert config.ddp.overlap_param_gather is False
    assert config.comm_overlap.tp_comm_overlap is False
    assert config.comm_overlap.overlap_grad_reduce is False
    assert config.comm_overlap.overlap_param_gather is False
    assert config.comm_overlap.overlap_param_gather_with_optimizer_step is False
    assert config.comm_overlap.overlap_moe_expert_parallel_comm is False
    assert config.comm_overlap.delay_wgrad_compute is False
    assert config.model.batch_p2p_sync is False

    assert config.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 64
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API"] == 64
    assert config.env_vars["NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API"] == 64
    assert config.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert config.env_vars["USE_MNNVL"] == 0
    assert config.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 0
    assert config.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 0
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1


@pytest.mark.parametrize(
    ("recipe_fn", "expected_micro_batch_size", "expected_graph_modules"),
    [
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config,
            2,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config,
            3,
            [],
        ),
        (
            qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config,
            3,
            ["moe_router", "moe_preprocess"],
        ),
    ],
)
def test_qwen35_vl_35b_gb200_measured_performance_defaults(
    recipe_fn: Callable,
    expected_micro_batch_size: int,
    expected_graph_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real GB200 factories should retain their precision-specific defaults."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.train.micro_batch_size == expected_micro_batch_size
    assert config.train.global_batch_size == 480
    assert config.model.moe_router_force_load_balancing is True
    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    expected_graph_impl = "transformer_engine" if expected_graph_modules else "none"
    assert config.model.cuda_graph_impl == expected_graph_impl
    assert config.model.cuda_graph_scope is None
    assert bool(config.model.cuda_graph_modules) is bool(expected_graph_modules)
    assert cuda_graph_module_names(config.model) == expected_graph_modules
    assert config.env_vars["NVTE_NORM_BWD_USE_CUDNN"] == 1
    assert config.env_vars["NVTE_NORM_FWD_USE_CUDNN"] == 1


def test_qwen35_vl_perf_recipes_enable_gdn_conv_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3.5-VL perf recipes should fuse the GatedDeltaNet pre-gated-delta-rule path.

    The flag lives in ``_qwen35_vl_common`` so it applies to every Qwen3.5-VL perf
    recipe regardless of platform -- it selects Triton kernels, not a
    hardware-specific path.

    Skipped when the installed Megatron-Core predates
    ``gdn_pre_gated_delta_rule_fusion``; the recipe guards on the same condition,
    so on an older core the flag is intentionally not set. The assertion below
    starts running as soon as the core is bumped.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config()

    if not hasattr(type(config.model), "gdn_pre_gated_delta_rule_fusion") and not hasattr(
        config.model, "gdn_pre_gated_delta_rule_fusion"
    ):
        pytest.skip("Megatron-Core does not expose gdn_pre_gated_delta_rule_fusion")

    assert config.model.gdn_pre_gated_delta_rule_fusion is True


def test_qwen35_vl_35b_gb300_fp8mx_enables_cutedsl_grouped_mlp(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GB300 MXFP8 recipe should route the MoE experts through the CuTe DSL grouped MLP.

    Transformer Engine only matches the fused grouped MLP when the op fuser, the
    32-wide GLU interleaving and the NVTE_CUTEDSL_FUSED_GROUPED_MLP gate are all
    set. If any one is missing TE does not raise -- it silently falls back to the
    cuBLASLt grouped GEMM -- so assert all three together.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config()

    assert config.model.use_transformer_engine_op_fuser is True
    assert config.model.moe_mlp_glu_interleave_size == 32
    assert config.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1

    # The fused kernel is MXFP8-only; the recipe must still select that recipe.
    assert config.mixed_precision.fp8 == "e4m3"
    assert config.mixed_precision.fp8_recipe == "mxfp8"

    # SwiGLU is a precondition of ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8, and the
    # fused kernel additionally requires FC1/FC2 dims divisible by 64 with
    # fc1_out == 2 * fc2_in.
    assert config.model.gated_linear_unit is True
    assert config.model.hidden_size % 64 == 0
    assert config.model.moe_ffn_hidden_size % 64 == 0


def test_qwen35_vl_35b_gb300_bf16_does_not_enable_cutedsl_grouped_mlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CuTe DSL grouped MLP is MXFP8-only, so the BF16 recipe must not enable it."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config()

    assert config.model.use_transformer_engine_op_fuser is False
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in config.env_vars


@pytest.mark.parametrize(
    "recipe_fn",
    [
        qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config,
        qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config,
        qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config,
    ],
)
def test_qwen35_vl_gb300_enables_attn_partial_cuda_graph(recipe_fn: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    """The GB300 35B/397B benchmarks must capture ``attn`` alongside the MoE graphs.

    This has to be asserted on the FINAL config, not on the assignment in the
    recipe body: ``_qwen35_vl_post`` runs afterwards and sets
    ``cuda_graph_impl="none"`` plus ``clear_cuda_graph_modules``, so a module list
    declared before it is dead code. ``_enable_partial_cuda_graphs`` re-enables the
    graphs after that hook.

    ``attn`` is the module that carries the win: at 397B under forced load
    balancing it moved the step from 390.5 to 580.2 TFLOP/s/GPU with GPU kernel
    time unchanged, the gain being per-launch CPU overhead removed by replay.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.model.cuda_graph_impl == "transformer_engine"
    # set_cuda_graph_modules writes cuda_graph_modules and nulls cuda_graph_scope;
    # MCore asserts if both are set at once.
    assert config.model.cuda_graph_scope is None
    assert set(cuda_graph_module_names(config.model)) == {"attn", "moe_router", "moe_preprocess"}
    # Graphs require the TE RNG tracker. `_benchmark_common` derives these flags from
    # the cuda_graph_impl value at its own call time, so re-enabling graphs later must
    # set them explicitly or model build asserts. Assert on the FINAL config.
    assert config.model.use_te_rng_tracker is True
    assert config.rng.te_rng_tracker is True


def test_qwen35_vl_122b_gb300_keeps_cuda_graphs_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 122B GB300 benchmark is deliberately NOT part of the attn-graph change."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()

    assert config.model.cuda_graph_impl == "none"
    assert cuda_graph_module_names(config.model) == []


@pytest.mark.parametrize(
    "recipe_fn",
    [
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_vr200_bf16_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8cs_config,
        qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8mx_config,
    ],
)
def test_qwen35_vl_397b_hybridep_domain_matches_expert_parallel_size(
    recipe_fn: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The VR200 aliases inherit EP from GB300 but restate the HybridEP env inline.

    HybridEP asserts that the EP group size is divisible by
    ``NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN``, and the launcher only re-derives
    that env when a CLI override changes EP, so the two must stay in sync here.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.model.moe_flex_dispatcher_backend == "hybridep"
    assert config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == config.model.expert_model_parallel_size


# GB300 recipes that run GatedDeltaNet on cuDNN once Megatron-Core exposes gdn_kernel_backend.
_GB300_CUDNN_GDN_RECIPES = (
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config,
)
# VR200 aliases built from those recipes; they keep FLA until cuDNN GDN is measured on VR200.
_VR200_CUDNN_GDN_ALIASES = (
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_bf16_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8cs_config,
    qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8mx_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_bf16_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8cs_config,
    qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8mx_config,
)
_QWEN35_VL_PERF_RECIPES = tuple(
    factory
    for factory in discover_recipe_factories(importlib.import_module("megatron.bridge.perf_recipes.qwen_vl"))
    if factory.__name__.startswith("qwen35_vl_")
)


@pytest.fixture
def megatron_core_with_gdn_kernel_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the offline provider the field NVIDIA/Megatron-LM#6645 adds, with Megatron-Core's default."""
    monkeypatch.setattr(_OfflineModelProvider, "gdn_kernel_backend", "fla", raising=False)


@pytest.mark.usefixtures("megatron_core_with_gdn_kernel_backend")
@pytest.mark.parametrize("recipe_fn", _GB300_CUDNN_GDN_RECIPES, ids=recipe_factory_id)
def test_qwen35_vl_gb300_runs_gdn_on_cudnn(recipe_fn: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    """The GB300 35B/397B benchmarks run GatedDeltaNet on Transformer Engine's cuDNN kernel.

    Asserted on the FINAL config: the FP8-CS and MXFP8 recipes switch precision after building the BF16 base,
    and the backend must survive that.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert config.model.gdn_kernel_backend == "transformer_engine"


@pytest.mark.usefixtures("megatron_core_with_gdn_kernel_backend")
@pytest.mark.parametrize("recipe_fn", _QWEN35_VL_PERF_RECIPES, ids=recipe_factory_id)
def test_qwen35_vl_cudnn_gdn_is_limited_to_gb300_35b_397b(
    recipe_fn: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the measured GB300 35B/397B recipes leave FLA.

    122B, B200, B300, GB200 and VR200 were not measured (the VR200 aliases inherit the GB300 model config and reset
    to FLA), and H100 is SM90, outside cuDNN's fast GDN engine.
    """
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    expected = "transformer_engine" if recipe_fn in _GB300_CUDNN_GDN_RECIPES else "fla"
    assert config.model.gdn_kernel_backend == expected


@pytest.mark.parametrize("recipe_fn", _GB300_CUDNN_GDN_RECIPES + _VR200_CUDNN_GDN_ALIASES, ids=recipe_factory_id)
def test_qwen35_vl_gdn_kernel_backend_not_invented_on_older_core(
    recipe_fn: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the Megatron-Core field the recipes must not create it: it would be silently unused."""
    patch_recipe_construction_dependencies(monkeypatch)

    config = recipe_fn()

    assert not hasattr(config.model, "gdn_kernel_backend")


def test_select_gdn_kernel_backend_assigns_only_declared_fields() -> None:
    """The guard on a dataclass, as on a real provider: assign a declared field, never invent one."""

    @dataclasses.dataclass
    class WithField:
        gdn_kernel_backend: str = "fla"

    @dataclasses.dataclass
    class WithFactoryField:  # no class attribute, only an instance one
        gdn_kernel_backend: str = dataclasses.field(default_factory=lambda: "fla")

    @dataclasses.dataclass
    class WithoutField:
        pass

    for model in (WithField(), WithFactoryField()):
        cfg = SimpleNamespace(model=model)
        _select_gdn_kernel_backend(cfg, "transformer_engine")
        assert cfg.model.gdn_kernel_backend == "transformer_engine"

    without_field = SimpleNamespace(model=WithoutField())
    _select_gdn_kernel_backend(without_field, "transformer_engine")
    assert not hasattr(without_field.model, "gdn_kernel_backend")


def test_megatron_core_gdn_kernel_backend_contract() -> None:
    """Megatron-Core's ``TransformerConfig.gdn_kernel_backend`` accepts ``"transformer_engine"``.

    ``_select_gdn_kernel_backend`` relies on that field (NVIDIA/Megatron-LM#6645, re-land NVIDIA/Megatron-LM#7583),
    and its guard turns a missing field into a no-op, so an upstream rename would leave the recipes on FLA without
    any error. Fail when Megatron-Core ships Transformer Engine GDN support without the field, and pin the default
    that ``megatron_core_with_gdn_kernel_backend`` gives the offline provider.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    if "gdn_kernel_backend" not in {field.name for field in dataclasses.fields(TransformerConfig)}:
        try:
            from megatron.core.extensions import transformer_engine as mcore_te
        except ImportError:
            mcore_te = None
        assert not (hasattr(mcore_te, "TEGatedDeltaNetAttention") or hasattr(mcore_te, "HAVE_TE_GDN")), (
            "Megatron-Core has Transformer Engine GDN but no TransformerConfig.gdn_kernel_backend; "
            "update _select_gdn_kernel_backend"
        )
        pytest.skip("Megatron-Core predates TransformerConfig.gdn_kernel_backend (NVIDIA/Megatron-LM#6645)")

    defaults = {field.name: field.default for field in dataclasses.fields(TransformerConfig)}
    assert defaults["gdn_kernel_backend"] == "fla", (
        "megatron_core_with_gdn_kernel_backend simulates the 'fla' default; update it, and pin 'fla' in the recipes "
        "that must keep FLA"
    )
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=2, gdn_kernel_backend="transformer_engine"
    )
    assert config.gdn_kernel_backend == "transformer_engine"


def test_qwen35_vl_perf_recipe_discovery_covers_the_cudnn_gdn_recipes() -> None:
    """The recipe matrix above must keep finding every recipe the cuDNN GDN tests reason about."""
    discovered = set(_QWEN35_VL_PERF_RECIPES)
    assert set(_GB300_CUDNN_GDN_RECIPES + _VR200_CUDNN_GDN_ALIASES) <= discovered
    platforms = {factory.__module__.split(".")[-2] for factory in discovered}
    assert {"b200", "b300", "gb200", "gb300", "h100", "vr200"} <= platforms
    assert any("122b" in factory.__name__ for factory in discovered)
