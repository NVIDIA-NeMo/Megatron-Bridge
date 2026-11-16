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

"""Explicit throughput benchmark cases composed from ordinary library recipes.

The launcher and temporary ``perf_recipes`` compatibility wrappers share this
module. Execution choices belong to library helpers; the tuples below declare
benchmark changes to the workload, numerics, RNG, schedule and check policy.
They are not recommended defaults for training to convergence.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import cast

from omegaconf import OmegaConf

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    MixedPrecisionConfig,
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
)
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger = logging.getLogger(__name__)

# These are independent selections: --max_steps changes train_iters, not the
# learning-rate horizon. Explicit dotted overrides can replace any selection.
THROUGHPUT_MEASUREMENT = (
    "train.train_iters=50",
    "train.eval_iters=0",
    "train.manual_gc=True",
    "train.manual_gc_interval=100",
    "checkpoint.save=null",
    "checkpoint.load=null",
    "logger.log_interval=1",
    "logger.tensorboard_dir=null",
    "ddp.check_for_nan_in_grad=False",
    "ddp.check_for_large_grads=False",
    "rerun_state_machine.check_for_nan_in_loss=False",
    "scheduler.lr_decay_iters=50",
    "scheduler.lr_warmup_iters=10",
    "tokenizer.use_tokenizer_vocab_size=False",
)

# Numerical choices remain benchmark selections, including existing fusions
# whose supported numerical contract is not established by this refactor.
BENCHMARK_NUMERICS = (
    "mixed_precision.grad_reduce_in_fp32=False",
    "ddp.grad_reduce_in_fp32=False",
    "model.use_transformer_engine_op_fuser=False",
    "model.cross_entropy_fusion_impl=te",
)
BENCHMARK_CUDNN_NORM = (
    "+env_vars.NVTE_NORM_BWD_USE_CUDNN=1",
    "+env_vars.NVTE_NORM_FWD_USE_CUDNN=1",
)
_LLAMA_SHAPE = (
    "tokenizer.vocab_size=128256",
    "model.should_pad_vocab=True",
    "model.apply_rope_fusion=True",
    "rng.te_rng_tracker=False",
    "model.use_te_rng_tracker=False",
)
_LLAMA_SFT = (
    "model.seq_length=4096",
    "train.global_batch_size=32",
    "train.micro_batch_size=1",
    "ddp.use_distributed_optimizer=True",
    "optimizer.use_distributed_optimizer=True",
    "model.disable_parameter_transpose_cache=True",
)
_QWEN_MOE_NUMERICS = (
    "model.apply_rope_fusion=True",
    "model.moe_router_force_load_balancing=True",
    "rng.te_rng_tracker=True",
    "model.use_te_rng_tracker=True",
)


@dataclass(frozen=True)
class PerfBenchmarkCase:
    """A library recipe, its execution selection, and explicit benchmark deltas.

    Factory and execution names are lazy imports so constructing one case does
    not import every other model family. The execution helper must not select
    optimizer precision, routing, batches, data, losses or training horizons.
    """

    library_recipe: str
    execution: str
    precision: str
    overrides: tuple[str, ...]


_LLAMA_LIBRARY = "megatron.bridge.recipes.llama.llama3."
_LLAMA_EXECUTION = "megatron.bridge.recipes.llama.h100.execution."
_QWEN_EXECUTION = "megatron.bridge.recipes.qwen.h100.execution."

PERF_BENCHMARK_CASES: dict[str, PerfBenchmarkCase] = {
    # Explicit allocation variants preserve the existing benchmark GBS; changing
    # topology later never implicitly rescales it on this migrated path.
    **{
        f"llama3_8b_pretrain_{gpus}gpu_h100_{precision}_config": PerfBenchmarkCase(
            library_recipe=_LLAMA_LIBRARY + "llama3_8b_pretrain_config",
            execution=_LLAMA_EXECUTION
            + ("llama3_8b_pretrain_bf16" if precision == "bf16" else "llama3_8b_pretrain_fp8"),
            precision="bf16" if precision == "bf16" else "fp8_cs",
            overrides=THROUGHPUT_MEASUREMENT
            + BENCHMARK_NUMERICS
            + _LLAMA_SHAPE
            + (BENCHMARK_CUDNN_NORM if precision != "bf16" else ())
            + (
                f"train.global_batch_size={128 if gpus == 8 else 1024}",
                "train.micro_batch_size=1",
            ),
        )
        for gpus in (8, 64)
        for precision in ("bf16", "fp8cs")
    },
    **{
        f"llama3_8b_sft_8gpu_h100_{precision}_config": PerfBenchmarkCase(
            library_recipe=_LLAMA_LIBRARY + "llama3_8b_sft_config",
            execution=_LLAMA_EXECUTION + "llama3_8b_sft",
            precision="bf16" if precision == "bf16" else "fp8_cs",
            overrides=THROUGHPUT_MEASUREMENT + BENCHMARK_NUMERICS + _LLAMA_SHAPE + _LLAMA_SFT + BENCHMARK_CUDNN_NORM,
        )
        for precision in ("bf16", "fp8cs")
    },
    "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config": PerfBenchmarkCase(
        library_recipe="megatron.bridge.recipes.qwen.h100.qwen3_moe.qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
        execution=_QWEN_EXECUTION + "qwen3_30b_a3b_bf16",
        precision="bf16",
        overrides=THROUGHPUT_MEASUREMENT + BENCHMARK_NUMERICS + _QWEN_MOE_NUMERICS,
    ),
    **{
        f"qwen3_30b_a3b_pretrain_16gpu_h100_{suffix}_config": PerfBenchmarkCase(
            library_recipe="megatron.bridge.recipes.qwen.h100.qwen3_moe.qwen3_30b_a3b_pretrain_8gpu_h100_bf16_config",
            execution=_QWEN_EXECUTION + "qwen3_30b_a3b_fp8",
            precision=precision,
            overrides=THROUGHPUT_MEASUREMENT
            + BENCHMARK_NUMERICS
            + _QWEN_MOE_NUMERICS
            + BENCHMARK_CUDNN_NORM
            + (
                "model.bias_activation_fusion=True",
                "model.moe_router_fusion=True",
                "model.seq_length=4096",
                "dataset.seq_length=4096",
                "train.global_batch_size=1024",
                "train.micro_batch_size=1",
            ),
        )
        for suffix, precision in (("fp8cs", "fp8_cs"), ("fp8ds", "fp8_ds"))
    },
    "qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config": PerfBenchmarkCase(
        library_recipe="megatron.bridge.recipes.qwen_vl.h100.qwen35_vl.qwen35_vl_35b_a3b_pretrain_config",
        # The existing VL library recipe already owns this complete execution
        # policy, including its graph/input constraints. Do not broaden them.
        execution="megatron.bridge.recipes.qwen_vl.h100.qwen35_vl._apply_qwen35_vl_35b_a3b_16gpu_h100_execution_config",
        precision="bf16",
        overrides=THROUGHPUT_MEASUREMENT
        + BENCHMARK_NUMERICS
        + (
            "tokenizer.tokenizer_type=NullTokenizer",
            "tokenizer.tokenizer_model=null",
            "model.bias_activation_fusion=True",
            "model.moe_router_fusion=True",
            "model.seq_length=4096",
            "dataset.seq_length=4096",
            "model.moe_router_force_load_balancing=True",
            "model.freeze_language_model=False",
            "model.freeze_vision_model=False",
        ),
    ),
}


def benchmark_precision(precision: str) -> MixedPrecisionConfig:
    """Construct an explicitly selected benchmark compute/reduction precision."""
    factories: dict[str, Callable[[], MixedPrecisionConfig]] = {
        "bf16": bf16_mixed,
        "fp8_cs": bf16_with_fp8_current_scaling_mixed,
        "fp8_ds": bf16_with_fp8_delayed_scaling_mixed,
    }
    if precision not in factories:
        raise ValueError(f"Unsupported migrated benchmark precision: {precision}")
    cfg = factories[precision]()
    if precision in {"fp8_cs", "fp8_ds"}:
        cfg.first_last_layers_bf16 = False
    cfg.grad_reduce_in_fp32 = False
    return cfg


def _changed_fields(config: object, before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    """Keep untouched objects out of the serialization roundtrip.

    OmegaConf represents tuples as lists and HF configs can add metadata on
    serialization. Neither should change merely because a benchmark sets an
    unrelated scalar. Mapping-valued fields are replaced as complete mappings.
    """
    changed: dict[str, object] = {}
    for key, value in after.items():
        if key in before and value == before[key]:
            continue
        current = getattr(config, key)
        previous = before.get(key)
        if is_dataclass(current) and isinstance(value, dict) and isinstance(previous, dict):
            changed[key] = _changed_fields(current, previous, value)
        else:
            changed[key] = value
    return changed


def apply_perf_benchmark_overrides(cfg: ConfigContainer, overrides: tuple[str, ...]) -> ConfigContainer:
    """Apply explicit benchmark selections through the existing typed overrides.

    Only selected fields are written back, preserving all other recipe objects
    and their types. Normal user CLI overrides are applied later by the launcher.
    """
    omega, _ = create_omegaconf_dict_config(cfg)
    before = OmegaConf.to_container(omega, resolve=True)
    omega = parse_hydra_overrides(omega, list(overrides))
    after = OmegaConf.to_container(omega, resolve=True)
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise TypeError("Benchmark overrides must preserve a configuration mapping.")
    # ConfigContainer serialization has string field names at the root.
    before_fields = cast(dict[str, object], before)
    after_fields = cast(dict[str, object], after)
    apply_overrides(cfg, _changed_fields(cfg, before_fields, after_fields), {})
    return cfg


def build_perf_benchmark_config(name: str) -> ConfigContainer:
    """Compose one named benchmark without importing its old factory.

    Callers apply argparse selections and then explicit dotted overrides before
    validating dependencies. This function preserves the old raw factory config;
    launcher-only choices (such as BF16 Adam PA) are made by the launcher.
    """
    case = PERF_BENCHMARK_CASES[name]
    module, attribute = case.library_recipe.rsplit(".", 1)
    # The registry names zero-argument library recipe factories.
    factory = cast(Callable[[], ConfigContainer], getattr(importlib.import_module(module), attribute))
    cfg = factory()
    cfg.mixed_precision = benchmark_precision(case.precision)
    if case.execution:
        module, attribute = case.execution.rsplit(".", 1)
        getattr(importlib.import_module(module), attribute)(cfg)
    overrides = list(case.overrides)
    if name.startswith("qwen35_vl_"):
        overrides.append(f"tokenizer.vocab_size={getattr(cfg.model, 'vocab_size')}")
    logger.info("Perf benchmark overrides for %s: %s", name, overrides)
    return apply_perf_benchmark_overrides(cfg, tuple(overrides))
