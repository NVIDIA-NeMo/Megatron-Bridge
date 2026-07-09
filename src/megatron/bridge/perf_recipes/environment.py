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

"""Environment composition for flat performance recipes.

The rules in this module describe training-process behavior. Cluster fabric,
credentials, cache locations, and scheduler settings remain executor-owned.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Set

from megatron.bridge.training.config import ConfigContainer


_HYBRIDEP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def _set_derived(
    env_vars: MutableMapping[str, str | int | float | bool],
    name: str,
    value: str | int | float | bool,
    protected_env_names: Set[str],
) -> None:
    """Set a derived value unless the user explicitly protected that name."""
    if name not in protected_env_names:
        env_vars[name] = value


def _remove_derived(
    env_vars: MutableMapping[str, str | int | float | bool],
    names: set[str],
    protected_env_names: Set[str],
) -> None:
    """Remove inapplicable derived values without deleting explicit overrides."""
    for name in names - protected_env_names:
        env_vars.pop(name, None)


def _scope_names(scope: object) -> set[str]:
    """Normalize CUDA graph scopes represented as strings or enum-like objects."""
    if not isinstance(scope, (list, tuple, set)):
        return set()
    return {item if isinstance(item, str) else getattr(item, "name", "") for item in scope}


def _hybridep_topology(gpu: str, expert_model_parallel_size: int) -> Mapping[str, int]:
    """Return topology-dependent HybridEP process settings."""
    if expert_model_parallel_size <= 0:
        raise ValueError("HybridEP expert parallel size must be positive.")

    normalized_gpu = gpu.lower()
    if normalized_gpu in {"h100", "b200", "b300"}:
        return {
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": min(expert_model_parallel_size, 8),
            "NVLINK_DOMAIN_SIZE": 8,
            "USE_MNNVL": 0,
        }
    if normalized_gpu not in {"gb200", "gb300", "vr200", "r100"}:
        raise ValueError(f"Unsupported GPU type for HybridEP topology: {gpu!r}.")
    if expert_model_parallel_size > 72:
        raise ValueError("HybridEP expert parallel size must not exceed the 72-rank NVLink domain.")
    return {
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": expert_model_parallel_size,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }


def apply_perf_recipe_environment(
    config: ConfigContainer,
    *,
    model_family_name: str,
    model_recipe_name: str,
    gpu: str,
    compute_dtype: str,
    train_task: str,
    protected_env_names: set[str] | None = None,
) -> None:
    """Compose nemo-ci performance settings into a flat recipe.

    The common rules are derived from finalized recipe features rather than
    model-family lookup whenever possible. Only measured model exceptions stay
    family-specific. Explicit Hydra ``env_vars`` overrides can be protected
    while CLI model overrides still cause dependent settings to be recomputed.

    Args:
        config: Final flat performance recipe to update.
        model_family_name: Model family used for measured legacy exceptions.
        model_recipe_name: Recipe selector used for measured model exceptions.
        gpu: Target GPU architecture.
        compute_dtype: Benchmark precision selector.
        train_task: Training task such as pretrain or sft.
        protected_env_names: Explicit environment overrides that derived rules
            must not replace or remove.
    """
    protected = protected_env_names or set()
    env_vars = config.env_vars
    model = config.model
    ddp = config.ddp
    comm_overlap = getattr(config, "comm_overlap", None)

    backend = getattr(model, "moe_flex_dispatcher_backend", None)
    tp_size = getattr(model, "tensor_model_parallel_size", 1) or 1
    cp_size = getattr(model, "context_parallel_size", 1) or 1
    pp_size = getattr(model, "pipeline_model_parallel_size", 1) or 1
    ep_size = getattr(model, "expert_model_parallel_size", 1) or 1
    moe_a2a_overlap = bool(getattr(comm_overlap, "overlap_moe_expert_parallel_comm", False))

    # Common process scheduling rules. Their values vary by finalized config.
    cuda_device_max_connections = 8
    if gpu.lower() in {"b200", "b300", "gb200", "gb300"}:
        cuda_device_max_connections = 32
    elif (tp_size > 1 or cp_size > 1) and not moe_a2a_overlap:
        cuda_device_max_connections = 1
    elif backend in {"deepep", "hybridep"}:
        cuda_device_max_connections = 32
    _set_derived(
        env_vars,
        "CUDA_DEVICE_MAX_CONNECTIONS",
        cuda_device_max_connections,
        protected,
    )

    layernorm_sm_margin = 20 if backend in {"deepep", "hybridep"} else 16
    _set_derived(env_vars, "NVTE_FWD_LAYERNORM_SM_MARGIN", layernorm_sm_margin, protected)
    _set_derived(env_vars, "NVTE_BWD_LAYERNORM_SM_MARGIN", layernorm_sm_margin, protected)

    # The Slurm executor historically supplied these allocator defaults. Keep
    # them recipe-owned so Kubeflow and direct rank-local launches agree.
    remove_allocator_defaults = getattr(ddp, "nccl_ub", None) is True or (
        model_family_name == "llama" and getattr(ddp, "use_megatron_fsdp", None) is True
    )
    nccl_ub = getattr(ddp, "nccl_ub", None) is True
    _set_derived(env_vars, "NCCL_NVLS_ENABLE", int(nccl_ub), protected)
    if nccl_ub:
        _set_derived(env_vars, "NCCL_CTA_POLICY", 1, protected)
    if remove_allocator_defaults:
        _remove_derived(env_vars, {"PYTORCH_CUDA_ALLOC_CONF", "NCCL_GRAPH_REGISTER"}, protected)
    else:
        if "PYTORCH_CUDA_ALLOC_CONF" not in protected:
            env_vars.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if "NCCL_GRAPH_REGISTER" not in protected:
            env_vars.setdefault("NCCL_GRAPH_REGISTER", 0)

    cuda_graph_impl = getattr(model, "cuda_graph_impl", None)
    cuda_graph_scope = _scope_names(getattr(model, "cuda_graph_scope", None))
    full_iteration_graph = cuda_graph_impl == "full_iteration" or (
        cuda_graph_impl == "local" and "full_iteration" in cuda_graph_scope
    )
    if full_iteration_graph:
        if "PYTORCH_CUDA_ALLOC_CONF" not in protected and not remove_allocator_defaults:
            current_allocator = str(env_vars.get("PYTORCH_CUDA_ALLOC_CONF", ""))
            if "graph_capture_record_stream_reuse" not in current_allocator:
                separator = "," if current_allocator else ""
                env_vars["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"{current_allocator}{separator}graph_capture_record_stream_reuse:True"
                )
        _set_derived(env_vars, "TORCH_NCCL_AVOID_RECORD_STREAMS", 0, protected)
    else:
        _set_derived(env_vars, "TORCH_NCCL_AVOID_RECORD_STREAMS", 1, protected)

    cutedsl_fused_grouped_mlp = bool(getattr(model, "use_transformer_engine_op_fuser", False))
    if cutedsl_fused_grouped_mlp:
        _set_derived(env_vars, "NVTE_CUTEDSL_FUSED_GROUPED_MLP", 1, protected)
    if cutedsl_fused_grouped_mlp and moe_a2a_overlap:
        _set_derived(env_vars, "CUDNNFE_CLUSTER_OVERLAP_MARGIN", 8, protected)

    if backend == "hybridep":
        for name, value in _hybridep_topology(gpu, ep_size).items():
            _set_derived(env_vars, name, value, protected)
        _set_derived(env_vars, "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", 128, protected)
    else:
        _remove_derived(env_vars, _HYBRIDEP_ENV_NAMES, protected)

    # cuDNN LayerNorm is opt-in. The legacy executor defaulted it on and then
    # removed it for every workload except these measured cases.
    keep_cudnn_layernorm = False
    if gpu.lower() == "h100":
        keep_cudnn_layernorm = (
            model_family_name == "llama"
            and model_recipe_name == "llama3_8b"
            and train_task == "pretrain"
            and compute_dtype == "fp8_cs"
        )
    elif gpu.lower() in {"gb200", "gb300"}:
        keep_cudnn_layernorm = (
            (
                model_family_name == "llama"
                and model_recipe_name == "llama3_70b"
                and train_task == "pretrain"
                and compute_dtype in {"bf16", "fp8_cs"}
            )
            or (
                model_family_name == "llama"
                and model_recipe_name == "llama31_405b"
                and train_task == "pretrain"
                and compute_dtype == "fp8_cs"
            )
            or (model_family_name in {"deepseek", "kimi"} and compute_dtype == "fp8_mx")
            or (model_family_name == "gpt_oss" and model_recipe_name == "gpt_oss_20b" and train_task == "pretrain")
        )
    if model_family_name == "llama" and train_task == "sft":
        keep_cudnn_layernorm = True
    if model_recipe_name == "nemotron_3_nano":
        keep_cudnn_layernorm = True

    cudnn_layernorm_names = {"NVTE_NORM_FWD_USE_CUDNN", "NVTE_NORM_BWD_USE_CUDNN"}
    if keep_cudnn_layernorm:
        for name in cudnn_layernorm_names:
            _set_derived(env_vars, name, 1, protected)
    else:
        _remove_derived(env_vars, cudnn_layernorm_names, protected)

    # Family/model exceptions retained from measured nemo-ci baselines.
    if model_family_name == "deepseek":
        _set_derived(env_vars, "NVTE_ALLOW_NONDETERMINISTIC_ALGO", 0, protected)
    if pp_size > 1 and (
        (model_recipe_name in {"llama3_70b", "llama31_405b"} and train_task == "pretrain")
        or (model_family_name == "llama" and train_task == "sft")
    ):
        _set_derived(env_vars, "NCCL_P2P_NET_CHUNKSIZE", 2097152, protected)
    if (
        gpu.lower() == "h100"
        and model_family_name == "llama"
        and model_recipe_name == "llama3_8b"
        and train_task == "pretrain"
        and compute_dtype == "fp8_cs"
    ):
        _set_derived(env_vars, "NCCL_CTA_POLICY", 1, protected)

    if gpu.lower() == "b300":
        _set_derived(env_vars, "NCCL_IGNORE_CPU_AFFINITY", 1, protected)
    if compute_dtype == "nvfp4":
        _set_derived(env_vars, "NVTE_USE_FAST_MATH", 1, protected)
    if getattr(model, "fine_grained_activation_offloading", False):
        _set_derived(env_vars, "NVTE_CPU_OFFLOAD_V1", 1, protected)
