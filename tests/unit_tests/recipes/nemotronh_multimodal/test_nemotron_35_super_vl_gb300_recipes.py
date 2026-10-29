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

"""Offline tests for the Nemotron 3.5 Super VL 64-GPU GB300 MXFP8 pretraining recipe.

The recipe is the shipped GB200 BF16 recipe plus the measured MXFP8, Transformer Engine op-fuser and NCCL EP settings,
so these tests pin that exact difference instead of re-asserting everything the GB200 recipe already owns.
"""

import dataclasses
import importlib
import inspect
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.bridge import recipes
from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import NemotronOmniModelProvider
from megatron.bridge.perf_recipes.environment import HYBRID_EP_ENV_NAMES
from megatron.bridge.recipes import nemotronh_multimodal
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer, apply_environment_variables
from megatron.bridge.training.flex_dispatcher_backend import validate_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import bf16_mixed, get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names, is_full_iteration_cuda_graph
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_construction_dependencies


pytestmark = pytest.mark.unit

# Import at collection time: the offline-construction fixture patches ``AutoBridge`` in every recipe module that is
# already loaded, so a lazy import inside a test would escape the patch.
_GB300_PACKAGE = importlib.import_module("megatron.bridge.recipes.nemotronh_multimodal.gb300")
_GB300_MODULE = importlib.import_module("megatron.bridge.recipes.nemotronh_multimodal.gb300.nemotron_35_super_vl")
_GB200_MODULE = importlib.import_module("megatron.bridge.recipes.nemotronh_multimodal.gb200.nemotron_35_super_vl")

_RECIPE_NAME = "nemotron_35_super_vl_pretrain_64gpu_gb300_fp8mx_config"
_RECIPE = getattr(_GB300_MODULE, _RECIPE_NAME)
_BASE_RECIPE = _GB200_MODULE.nemotron_35_super_vl_pretrain_64gpu_gb200_bf16_config

# The measured run applied these CLI overrides on top of the GB200 BF16 recipe (plus three exported variables).
_MEASURED_MODEL_OVERRIDES = {
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    "moe_router_padding_for_quantization": True,
    "use_transformer_engine_op_fuser": True,
    "moe_flex_dispatcher_backend": "ncclep",
    "moe_dispatch_fwd_dtype": "mxfp8",
    "moe_combine_bwd_dtype": "mxfp8",
}
_MEASURED_PRECISION_OVERRIDES = {
    "fp8": "e4m3",
    "fp8_recipe": "mxfp8",
    "fp8_param_gather": True,
    "reuse_grad_buf_for_mxfp8_param_ag": True,
}
_MEASURED_ENV_VARS = {"NCCL_EP_HT_EM_PULL_PUSH": 1, "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1, "NVTE_FLASH_ATTN_V4": 0}

# Everything that may differ from the GB200 BF16 recipe. An extra difference is an unreviewed change; a missing one
# means the recipe no longer carries a measured setting. ``moe_use_grouped_tensor`` is the device-side token count
# that every NCCL EP recipe sets explicitly, and ``fp8_param`` follows ``fp8_param_gather``.
_CHANGED_MODEL_FIELDS = {*_MEASURED_MODEL_OVERRIDES, "moe_use_grouped_tensor"}
_CHANGED_PRECISION_FIELDS = {*_MEASURED_PRECISION_OVERRIDES, "fp8_param"}
# The four HybridEP names are dropped because nothing reads them on the NCCL EP backend (perf_recipes/environment.py);
# a recipe that keeps them must remove HYBRID_EP_ENV_NAMES here and the isdisjoint assertion further down.
_CHANGED_ENV_VARS = {*_MEASURED_ENV_VARS, *HYBRID_EP_ENV_NAMES}


@pytest.fixture(autouse=True)
def _keep_recipe_construction_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_recipe_construction_dependencies(monkeypatch)


def _changed_attributes(base: object, cfg: object) -> set[str]:
    """Return the attribute names whose value differs, counting an attribute only one side has."""
    return {
        name for name in vars(base).keys() | vars(cfg).keys() if getattr(base, name, None) != getattr(cfg, name, None)
    }


def test_recipe_is_public_and_unique_across_both_recipe_packages() -> None:
    assert _RECIPE.__module__ == _GB300_MODULE.__name__
    assert inspect.signature(_RECIPE).parameters == {}
    for namespace in (_GB300_PACKAGE, nemotronh_multimodal, recipes):
        assert getattr(namespace, _RECIPE_NAME) is _RECIPE
    assert _RECIPE_NAME in _GB300_PACKAGE.__all__
    assert _RECIPE_NAME in nemotronh_multimodal.__all__


def test_recipe_is_the_gb200_bf16_recipe_plus_the_measured_overrides() -> None:
    cfg = _RECIPE()
    base = _BASE_RECIPE()

    assert _changed_attributes(base.model, cfg.model) == _CHANGED_MODEL_FIELDS
    assert _changed_attributes(base.mixed_precision, cfg.mixed_precision) == _CHANGED_PRECISION_FIELDS
    changed_env_vars = {
        name
        for name in cfg.env_vars.keys() | base.env_vars.keys()
        if cfg.env_vars.get(name) != base.env_vars.get(name)
    }
    assert changed_env_vars == _CHANGED_ENV_VARS
    # Everything outside model, precision and environment is inherited unchanged.
    owned = {"model", "mixed_precision", "env_vars"}
    for section in (field.name for field in dataclasses.fields(ConfigContainer) if field.name not in owned):
        assert getattr(cfg, section) == getattr(base, section), section


def test_measured_parallel_layout_and_batch_shape() -> None:
    cfg = _RECIPE()

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 1
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.expert_model_parallel_size == 64
    assert cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.sequence_parallel is False
    assert cfg.model.seq_length == cfg.dataset.seq_length == 8192
    assert cfg.train.global_batch_size == 512
    assert cfg.train.micro_batch_size == 1

    # The name promises 64 GPUs: both the dense and the expert mesh must tile them, with whole micro batches per rank.
    world_size = 64
    dense_ranks = cfg.model.tensor_model_parallel_size * cfg.model.pipeline_model_parallel_size
    dense_ranks *= cfg.model.context_parallel_size
    expert_ranks = cfg.model.expert_model_parallel_size * cfg.model.expert_tensor_parallel_size
    expert_ranks *= cfg.model.pipeline_model_parallel_size
    assert world_size % dense_ranks == 0
    assert world_size % expert_ranks == 0
    assert cfg.train.global_batch_size % (cfg.train.micro_batch_size * (world_size // dense_ranks)) == 0


def test_nccl_ep_stack_is_eager_and_matches_the_gb300_perf_recipes() -> None:
    cfg = _RECIPE()

    assert cfg.model.moe_token_dispatcher_type == "flex"
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"
    assert cfg.model.moe_use_grouped_tensor is True
    # Eager NCCL EP: no expert-rank capacity factor, and the TE graphs keep the shipped GB200 scope.
    assert getattr(cfg.model, "moe_expert_rank_capacity_factor", None) is None
    assert cfg.model.cuda_graph_impl == "transformer_engine"
    assert cuda_graph_module_names(cfg.model) == ["attn", "mamba", "moe_router", "moe_preprocess"]
    assert not is_full_iteration_cuda_graph(cfg.model)
    assert cfg.model.moe_router_force_load_balancing is False
    assert cfg.model.moe_flex_dispatcher_num_sms == 32
    assert cfg.ddp.overlap_grad_reduce is False
    assert cfg.ddp.overlap_param_gather is False


def test_recipe_selects_the_backend_without_probing_the_build_host_and_validates_on_gb300() -> None:
    with patch("torch.cuda.get_device_properties", side_effect=AssertionError("recipe probed the build host")):
        cfg = _RECIPE()

    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(major=10, name="NVIDIA GB300")):
        validate_flex_dispatcher_backend(cfg.model)
    assert cfg.model.moe_flex_dispatcher_backend == "ncclep"


def test_mxfp8_op_fuser_and_wire_formats() -> None:
    cfg = _RECIPE()

    assert cfg.model.use_transformer_engine_op_fuser is True
    # The measured run left the GLU interleave unset (null in its resolved config); do not copy another model's 32.
    assert getattr(cfg.model, "moe_mlp_glu_interleave_size", None) is None
    assert cfg.model.moe_router_padding_for_quantization is True
    assert cfg.model.moe_dispatch_fwd_dtype == "mxfp8"
    assert cfg.model.moe_combine_bwd_dtype == "mxfp8"
    assert cfg.mixed_precision.fp8 == "e4m3"
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather is True
    assert cfg.mixed_precision.fp8_param is True
    assert cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag is True
    # Gradient reduction stays in FP32, as in the BF16 recipe the run started from.
    assert cfg.mixed_precision.grad_reduce_in_fp32 is True
    assert cfg.ddp.grad_reduce_in_fp32 is True


def test_mixed_precision_replays_the_measured_overrides_on_the_bf16_preset() -> None:
    expected = bf16_mixed()
    for key, value in _MEASURED_PRECISION_OVERRIDES.items():
        if not hasattr(expected, key):
            raise ValueError(f"MixedPrecisionConfig has no field '{key}'")
        setattr(expected, key, value)

    assert _RECIPE().mixed_precision == expected


def test_precision_setup_reaches_the_ddp_and_optimizer_configs() -> None:
    cfg = _RECIPE()

    get_mixed_precision_config(cfg.mixed_precision).setup(cfg.model, cfg.optimizer, cfg.ddp)

    assert cfg.ddp.fp8_param_gather is True
    assert cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.optimizer.fp8_recipe == "mxfp8"
    assert cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag is True
    assert cfg.optimizer.use_precision_aware_optimizer is False


def test_environment_is_declared_explicitly_and_gated_for_nccl_ep() -> None:
    cfg = _RECIPE()
    base = _BASE_RECIPE()

    assert COMMON_RECIPE_ENV_VARS.items() <= cfg.env_vars.items()
    assert {name: cfg.env_vars[name] for name in _MEASURED_ENV_VARS} == _MEASURED_ENV_VARS
    # Pull/push mode keeps the NCCL EP staging buffers small: by the library's buffer sizing, the default mode would
    # need roughly 80-124 GB more per rank at expert parallelism 64 (confirmed at small scale, not run at that size).
    assert cfg.env_vars["NCCL_EP_HT_EM_PULL_PUSH"] == 1
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    # Nothing reads the HybridEP tuning on the NCCL EP backend, so the launched environment must not advertise it.
    assert cfg.env_vars.keys().isdisjoint(HYBRID_EP_ENV_NAMES)
    # Building the recipe must not leak its settings into the shared defaults or into the recipe it derives from.
    assert _MEASURED_ENV_VARS.keys().isdisjoint(COMMON_RECIPE_ENV_VARS)
    assert _MEASURED_ENV_VARS.keys().isdisjoint(base.env_vars)


def test_recipe_requires_a_megatron_core_with_the_nccl_ep_wire_fields() -> None:
    declares_both = SimpleNamespace(moe_dispatch_fwd_dtype="bf16", moe_combine_bwd_dtype="bf16")
    _GB300_MODULE._require_nccl_ep_wire_fields(declares_both)

    for missing in ("moe_dispatch_fwd_dtype", "moe_combine_bwd_dtype"):
        provider = SimpleNamespace(**{name: "bf16" for name in vars(declares_both) if name != missing})
        with pytest.raises(ValueError, match=missing):
            _GB300_MODULE._require_nccl_ep_wire_fields(provider)


def test_each_call_returns_an_independent_environment() -> None:
    first, second = _RECIPE(), _RECIPE()

    first.env_vars["SENTINEL"] = 1

    assert first.env_vars is not second.env_vars
    assert "SENTINEL" not in second.env_vars


def test_environment_is_exported_by_the_runtime_hook_without_overriding_the_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _RECIPE()
    for name in cfg.env_vars:
        # Record the original state first (monkeypatch only restores variables it saw), then start from an empty slate.
        monkeypatch.setenv(name, "unset-by-test")
        monkeypatch.delenv(name)
    # A value the launcher already exports wins over the recipe default.
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")

    apply_environment_variables(cfg)

    assert os.environ["NCCL_EP_HT_EM_PULL_PUSH"] == "1"
    assert os.environ["NVTE_FLASH_ATTN_V4"] == "0"
    assert os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == "0"


def test_every_changed_model_field_is_declared_by_the_real_provider() -> None:
    """The offline provider accepts any attribute, so a typo or an older Megatron Core would pass silently.

    Assigning a field the dataclass does not declare creates a phantom attribute: the recipe then looks tuned while the
    runtime never reads it. Check the names against the provider class that ``AutoBridge`` builds for this model.
    """
    declared = {field.name for field in dataclasses.fields(NemotronOmniModelProvider)}

    assert _CHANGED_MODEL_FIELDS - declared == set()
