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

"""Tests for feature-derived flat performance recipe environment settings."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from megatron.bridge.perf_recipes.environment import apply_perf_recipe_environment, perf_recipe_environment


_CANONICAL_RECIPE_NAME = re.compile(
    r".+_(?:pretrain|sft|peft)_\d+gpu_[a-z0-9]+_(?:bf16|fp8cs|fp8mx|fp8sc|nvfp4)(?:_.+)?_config"
)


def _config(
    *,
    backend=None,
    tp=1,
    pp=1,
    cp=1,
    ep=1,
    cuda_graph_impl=None,
    cuda_graph_scope=None,
    cutedsl=False,
    moe_a2a_overlap=False,
    nccl_ub=False,
    use_megatron_fsdp=False,
    fine_grained_activation_offloading=False,
    env_vars=None,
):
    return SimpleNamespace(
        env_vars=dict(env_vars or {}),
        model=SimpleNamespace(
            moe_flex_dispatcher_backend=backend,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_scope=cuda_graph_scope or [],
            use_transformer_engine_op_fuser=cutedsl,
            fine_grained_activation_offloading=fine_grained_activation_offloading,
        ),
        ddp=SimpleNamespace(nccl_ub=nccl_ub, use_megatron_fsdp=use_megatron_fsdp),
        comm_overlap=SimpleNamespace(overlap_moe_expert_parallel_comm=moe_a2a_overlap),
    )


def _apply(
    config,
    *,
    family="qwen",
    recipe="qwen3_30b_a3b",
    gpu="h100",
    dtype="bf16",
    task="pretrain",
    protected=None,
):
    apply_perf_recipe_environment(
        config,
        model_family_name=family,
        model_recipe_name=recipe,
        gpu=gpu,
        compute_dtype=dtype,
        train_task=task,
        protected_env_names=protected,
    )
    return config.env_vars


@pytest.mark.parametrize(
    ("gpu", "ep", "expected"),
    [
        (
            "h100",
            32,
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
                "NVLINK_DOMAIN_SIZE": 8,
                "USE_MNNVL": 0,
            },
        ),
        (
            "gb200",
            32,
            {
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
            },
        ),
    ],
)
def test_hybridep_environment_is_topology_derived(gpu, ep, expected):
    env = _apply(_config(backend="hybridep", ep=ep), gpu=gpu)

    assert env.items() >= expected.items()
    assert env["NUM_OF_TOKENS_PER_CHUNK_COMBINE_API"] == 128
    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == 32
    assert env["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert env["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20


def test_non_flex_hopper_environment_uses_parallelism_ordering_defaults():
    env = _apply(_config(tp=2))

    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == 1
    assert env["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 16
    assert env["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 16
    assert env["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1
    assert env["NCCL_NVLS_ENABLE"] == 0
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["NCCL_GRAPH_REGISTER"] == 0


def test_hopper_program_ordering_wins_over_flex_backend_default():
    env = _apply(_config(backend="hybridep", tp=2, ep=8))

    assert env["CUDA_DEVICE_MAX_CONNECTIONS"] == 1


def test_full_iteration_cutedsl_environment_is_feature_derived():
    env = _apply(
        _config(
            backend="hybridep",
            ep=32,
            cuda_graph_impl="full_iteration",
            cutedsl=True,
            moe_a2a_overlap=True,
        ),
        gpu="gb200",
        dtype="fp8_mx",
    )

    assert env["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 0
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == ("expandable_segments:True,graph_capture_record_stream_reuse:True")
    assert env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] == 1
    assert env["CUDNNFE_CLUSTER_OVERLAP_MARGIN"] == 8


def test_explicit_recipe_environment_override_beats_derived_values():
    config = _config(
        backend="hybridep",
        ep=32,
        env_vars={"NVLINK_DOMAIN_SIZE": 8, "NVTE_FWD_LAYERNORM_SM_MARGIN": 48},
    )

    env = _apply(
        config,
        gpu="gb200",
        protected={"NVLINK_DOMAIN_SIZE", "NVTE_FWD_LAYERNORM_SM_MARGIN"},
    )

    assert env["NVLINK_DOMAIN_SIZE"] == 8
    assert env["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 48
    assert env["USE_MNNVL"] == 1


def test_allocator_defaults_are_removed_for_llama_megatron_fsdp():
    env = _apply(
        _config(
            use_megatron_fsdp=True,
            env_vars={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_GRAPH_REGISTER": 0},
        ),
        family="llama",
        recipe="llama3_70b",
        gpu="gb200",
    )

    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert "NCCL_GRAPH_REGISTER" not in env


def test_nccl_ub_environment_replaces_allocator_and_nvls_defaults():
    env = _apply(_config(nccl_ub=True))

    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert "NCCL_GRAPH_REGISTER" not in env
    assert env["NCCL_NVLS_ENABLE"] == 1
    assert env["NCCL_CTA_POLICY"] == 1


def test_recipe_decorator_finalizes_a_direct_builder():
    config = _config(backend="hybridep", ep=32)

    @perf_recipe_environment(model_family_name="qwen")
    def qwen3_30b_a3b_pretrain_32gpu_gb200_fp8mx_config():
        return config

    resolved = qwen3_30b_a3b_pretrain_32gpu_gb200_fp8mx_config()

    assert resolved is config
    assert resolved.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert resolved.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20


def test_reapplying_environment_removes_stale_derived_values():
    config = _config(
        backend="hybridep",
        ep=32,
        cuda_graph_impl="full_iteration",
        cutedsl=True,
        moe_a2a_overlap=True,
        nccl_ub=True,
    )
    _apply(config, gpu="gb200", dtype="nvfp4")

    config.model.moe_flex_dispatcher_backend = None
    config.model.cuda_graph_impl = None
    config.model.use_transformer_engine_op_fuser = False
    config.ddp.nccl_ub = False
    env = _apply(config, gpu="h100", dtype="bf16")

    assert "NVLINK_DOMAIN_SIZE" not in env
    assert "NVTE_CUTEDSL_FUSED_GROUPED_MLP" not in env
    assert "CUDNNFE_CLUSTER_OVERLAP_MARGIN" not in env
    assert "NCCL_CTA_POLICY" not in env
    assert "NVTE_USE_FAST_MATH" not in env
    assert env["NCCL_NVLS_ENABLE"] == 0
    assert env["TORCH_NCCL_AVOID_RECORD_STREAMS"] == 1


def test_every_flat_recipe_builder_is_explicitly_decorated():
    recipe_root = Path(__file__).resolve().parents[3] / "src" / "megatron" / "bridge" / "perf_recipes"
    undecorated = []
    families = set()

    for path in recipe_root.glob("*/*/*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or _CANONICAL_RECIPE_NAME.fullmatch(node.name) is None:
                continue
            families.add(path.relative_to(recipe_root).parts[0])
            is_decorated = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "perf_recipe_environment"
                for decorator in node.decorator_list
            )
            if not is_decorated:
                undecorated.append(f"{path.relative_to(recipe_root)}:{node.name}")

    assert not undecorated
    assert families >= {"deepseek", "gpt_oss", "kimi", "llama", "nemotronh", "qwen", "qwen_vl", "wan"}


@pytest.mark.parametrize(
    ("family", "recipe", "gpu", "dtype", "task", "config", "expected"),
    [
        (
            "deepseek",
            "deepseek_v3",
            "gb200",
            "fp8_mx",
            "pretrain",
            _config(backend="hybridep", ep=64),
            {
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
            },
        ),
        (
            "kimi",
            "kimi_k2",
            "gb300",
            "fp8_mx",
            "pretrain",
            _config(backend="hybridep", ep=64),
            {"NVTE_NORM_FWD_USE_CUDNN": 1, "NVTE_NORM_BWD_USE_CUDNN": 1},
        ),
        (
            "gpt_oss",
            "gpt_oss_120b",
            "gb200",
            "fp8_mx",
            "pretrain",
            _config(backend="hybridep", ep=64),
            {
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NVLINK_DOMAIN_SIZE": 72,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            },
        ),
        (
            "llama",
            "llama3_8b",
            "h100",
            "fp8_cs",
            "pretrain",
            _config(),
            {
                "NCCL_CTA_POLICY": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
            },
        ),
        (
            "llama",
            "llama3_70b",
            "gb200",
            "bf16",
            "pretrain",
            _config(pp=4),
            {
                "NCCL_P2P_NET_CHUNKSIZE": 2097152,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
            },
        ),
        (
            "nemotronh",
            "nemotron_3_nano",
            "b200",
            "bf16",
            "pretrain",
            _config(backend="hybridep", ep=8),
            {"NVTE_NORM_FWD_USE_CUDNN": 1, "NVTE_NORM_BWD_USE_CUDNN": 1},
        ),
        (
            "qwen",
            "qwen3_30b_a3b",
            "gb200",
            "fp8_mx",
            "pretrain",
            _config(backend="hybridep", ep=32, cuda_graph_impl="full_iteration", cutedsl=True),
            {
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            },
        ),
        (
            "qwen_vl",
            "qwen3_vl_30b_a3b",
            "gb200",
            "bf16",
            "pretrain",
            _config(backend="hybridep", ep=8),
            {"NVLINK_DOMAIN_SIZE": 72, "USE_MNNVL": 1},
        ),
        (
            "wan",
            "wan_14b",
            "h100",
            "bf16",
            "pretrain",
            _config(tp=2),
            {"CUDA_DEVICE_MAX_CONNECTIONS": 1, "NVTE_FWD_LAYERNORM_SM_MARGIN": 16},
        ),
    ],
)
def test_family_baseline_exceptions(family, recipe, gpu, dtype, task, config, expected):
    env = _apply(config, family=family, recipe=recipe, gpu=gpu, dtype=dtype, task=task)

    assert env.items() >= expected.items()
