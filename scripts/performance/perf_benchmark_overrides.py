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

"""Compose library recipes with explicit perf benchmark overrides."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from hydra.core.override_parser.overrides_parser import OverridesParser
from utils.overrides import (
    _apply_flat_cli_environment_compatibility,
    apply_one_gpu_per_rank_device_mapping,
    set_cli_overrides,
    set_user_overrides,
)
from utils.utils import (
    _recipe_function_name,
    apply_target_topology_environment,
    explicit_environment_override_names,
    finalize_config_overrides,
)

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides import PERF_BENCHMARK_CASES, build_perf_benchmark_config
from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph


_COMMON_SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "common"
if str(_COMMON_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_SCRIPTS_ROOT))

from benchmark_parallelism import data_parallel_size as _validated_data_parallel_size  # noqa: E402
from benchmark_parallelism import topology_from_config  # noqa: E402


def selected_perf_benchmark_case(args: argparse.Namespace) -> str | None:
    """Resolve migrated cases using the existing model/task/hardware selectors."""
    name = _recipe_function_name(
        model_recipe_name=args.model_recipe_name,
        task=args.task,
        num_gpus=args.num_gpus,
        gpu=args.gpu,
        precision=args.compute_dtype,
        config_variant=getattr(args, "config_variant", None),
    )
    return name if name in PERF_BENCHMARK_CASES else None


def validate_perf_benchmark_dependencies(cfg: ConfigContainer, *, name: str, num_gpus: int) -> None:
    """Reject incompatible final selections without rescaling batches or precision.

    This checks configuration dependencies, not GPU memory fit. The normal
    training initialization still validates all model-specific requirements.
    """
    dp = _validated_data_parallel_size(num_gpus=num_gpus, topology=topology_from_config(cfg.model))
    mbs, gbs = cfg.train.micro_batch_size, cfg.train.global_batch_size
    if mbs <= 0 or gbs <= 0 or gbs % (mbs * dp):
        raise ValueError(f"GBS={gbs} must be a positive multiple of MBS={mbs} * DP={dp}; set it explicitly.")
    experts = getattr(cfg.model, "num_moe_experts", None)
    ep = getattr(cfg.model, "expert_model_parallel_size", 1)
    if experts and experts % ep:
        raise ValueError(f"Expert count {experts} must be divisible by EP={ep}.")
    if getattr(cfg.model, "sequence_parallel", False) and getattr(cfg.model, "tensor_model_parallel_size", 1) == 1:
        raise ValueError("Sequence parallelism requires tensor parallelism greater than one.")
    if cfg.optimizer.use_precision_aware_optimizer and cfg.optimizer.optimizer != "adam":
        raise ValueError(
            "Precision-aware optimizer requires Adam; disable it explicitly when selecting another optimizer."
        )
    case = PERF_BENCHMARK_CASES[name]
    if case.execution.endswith(("pretrain_fp8", "a3b_fp8")) and not getattr(cfg.mixed_precision, "fp8", None):
        raise ValueError(
            "This execution layout requires FP8; select the BF16 case before changing precision/topology."
        )
    if is_full_iteration_cuda_graph(cfg.model) and cfg.rerun_state_machine.check_for_nan_in_loss:
        raise ValueError("Full-iteration graphs require an explicit compatible loss-check policy.")


def _explicit_config_fields(overrides: list[str]) -> set[str]:
    """Include dotted and dictionary-form selections when determining precedence."""
    selected: set[str] = set()

    def collect(path: str, value: object) -> None:
        selected.add(path)
        if isinstance(value, dict):
            for key, item in value.items():
                collect(f"{path}.{key}", item)

    for override in OverridesParser.create().parse_overrides(overrides):
        collect(override.key_or_group, override.value())
    return selected


def prepare_perf_benchmark_config(args: argparse.Namespace, cli_overrides: list[str], *, name: str) -> ConfigContainer:
    """Apply case, argparse and dotted overrides in that order, then validate."""
    cfg = build_perf_benchmark_config(name)
    selected = _explicit_config_fields(cli_overrides)
    # Builder-backed models expose both nested and forwarded transformer paths.
    selected |= {field.replace("model.transformer.", "model.", 1) for field in selected}
    base_topology = {
        field: getattr(cfg.model, field)
        for field in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "expert_model_parallel_size",
        )
    }
    base_env = dict(cfg.env_vars)
    base_ep = getattr(cfg.model, "expert_model_parallel_size", 1)
    base_backend = getattr(cfg.model, "moe_flex_dispatcher_backend", None)
    base_overlap = bool(cfg.comm_overlap and cfg.comm_overlap.overlap_moe_expert_parallel_comm)
    cfg = set_user_overrides(cfg, args, preserve_training_policy=True)
    cfg = set_cli_overrides(cfg, cli_overrides)
    # Retain the old BF16 Adam default only when the effective optimizer is Adam
    # and the user did not select precision-aware optimization explicitly.
    if (
        args.compute_dtype == "bf16"
        and cfg.optimizer.optimizer == "adam"
        and "optimizer.use_precision_aware_optimizer" not in selected
    ):
        cfg.optimizer.use_precision_aware_optimizer = True
    protected = explicit_environment_override_names(cli_overrides, base_env, cfg.env_vars)
    dependent_selections = {
        path: getattr(getattr(cfg, section), field)
        for path in (
            "ddp.average_in_collective",
            "ddp.fsdp_manual_registration",
            "model.moe_token_dispatcher_type",
            "model.moe_shared_expert_overlap",
        )
        if path in selected
        for section, field in [path.split(".", 1)]
    }
    cfg = finalize_config_overrides(cfg)
    for path, selected_value in dependent_selections.items():
        section, field = path.split(".", 1)
        required = getattr(getattr(cfg, section), field)
        if required != selected_value:
            raise ValueError(
                f"Explicit {path}={selected_value!r} conflicts with the selected configuration; requires {required!r}."
            )
    effective_args = copy.copy(args)
    # Derive environment from final config values, including dotted-only changes.
    for field, base_value in base_topology.items():
        value = getattr(cfg.model, field)
        if getattr(args, field) is not None or f"model.{field}" in selected or value != base_value:
            setattr(effective_args, field, value)
    effective_args.nccl_ub = cfg.ddp.nccl_ub
    if (
        args.moe_a2a_overlap is not None
        or "comm_overlap.overlap_moe_expert_parallel_comm" in selected
        or bool(cfg.comm_overlap and cfg.comm_overlap.overlap_moe_expert_parallel_comm) != base_overlap
    ):
        effective_args.moe_a2a_overlap = bool(cfg.comm_overlap and cfg.comm_overlap.overlap_moe_expert_parallel_comm)
    cfg = _apply_flat_cli_environment_compatibility(
        cfg,
        effective_args,
        base_dispatcher_backend=base_backend,
        base_moe_a2a_overlap=base_overlap,
        protected_env_names=protected,
    )
    if getattr(cfg.model, "expert_model_parallel_size", 1) != base_ep:
        apply_target_topology_environment(cfg, gpu=args.gpu, protected_env_names=protected)
    validate_perf_benchmark_dependencies(cfg, name=name, num_gpus=args.num_gpus)
    return apply_one_gpu_per_rank_device_mapping(cfg)
