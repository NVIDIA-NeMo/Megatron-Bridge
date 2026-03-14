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

"""Equivalence tests: old perf pipeline vs new flat perf recipes.

The old path (scripts/performance/configs/) and the new path
(src/megatron/bridge/recipes/llama/llama3_perf.py) must produce identical
ConfigContainer objects for every migrated recipe.
"""

import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest


# Add the performance scripts directory to sys.path so we can import the old pipeline
_PERF_SCRIPTS_DIR = str(Path(__file__).resolve().parents[4] / "scripts" / "performance")
if _PERF_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _PERF_SCRIPTS_DIR)

from utils.utils import get_perf_optimized_recipe

from megatron.bridge.recipes.llama.llama3_perf import (
    llama3_8b_pretrain_8gpu_b200_bf16_config,
    llama3_8b_pretrain_8gpu_b200_fp8cs_config,
    llama3_8b_pretrain_8gpu_b200_fp8mx_config,
    llama3_8b_pretrain_8gpu_b200_nvfp4_config,
    llama3_8b_pretrain_8gpu_b300_bf16_config,
    llama3_8b_pretrain_8gpu_b300_fp8cs_config,
    llama3_8b_pretrain_8gpu_b300_fp8mx_config,
    llama3_8b_pretrain_8gpu_b300_nvfp4_config,
    llama3_8b_pretrain_8gpu_gb200_bf16_config,
    llama3_8b_pretrain_8gpu_gb200_fp8cs_config,
    llama3_8b_pretrain_8gpu_gb200_fp8mx_config,
    llama3_8b_pretrain_8gpu_gb200_nvfp4_config,
    llama3_8b_pretrain_8gpu_gb300_bf16_config,
    llama3_8b_pretrain_8gpu_gb300_fp8cs_config,
    llama3_8b_pretrain_8gpu_gb300_fp8mx_config,
    llama3_8b_pretrain_8gpu_gb300_nvfp4_config,
    llama3_8b_pretrain_8gpu_h100_bf16_config,
    llama3_8b_pretrain_8gpu_h100_fp8cs_config,
    llama3_8b_pretrain_8gpu_r100_bf16_config,
    llama3_8b_pretrain_8gpu_r100_fp8cs_config,
    llama3_8b_pretrain_8gpu_r100_fp8mx_config,
    llama3_8b_pretrain_8gpu_r100_nvfp4_config,
    llama3_70b_pretrain_64gpu_b200_bf16_config,
    llama3_70b_pretrain_64gpu_b200_fp8cs_config,
    llama3_70b_pretrain_64gpu_b200_fp8mx_config,
    llama3_70b_pretrain_64gpu_b200_nvfp4_config,
    llama3_70b_pretrain_64gpu_b300_bf16_config,
    llama3_70b_pretrain_64gpu_b300_fp8cs_config,
    llama3_70b_pretrain_64gpu_b300_fp8mx_config,
    llama3_70b_pretrain_64gpu_b300_nvfp4_config,
    llama3_70b_pretrain_64gpu_gb200_bf16_config,
    llama3_70b_pretrain_64gpu_gb200_fp8cs_config,
    llama3_70b_pretrain_64gpu_gb200_fp8mx_config,
    llama3_70b_pretrain_64gpu_gb200_nvfp4_config,
    llama3_70b_pretrain_64gpu_gb300_bf16_config,
    llama3_70b_pretrain_64gpu_gb300_fp8cs_config,
    llama3_70b_pretrain_64gpu_gb300_fp8mx_config,
    llama3_70b_pretrain_64gpu_gb300_nvfp4_config,
    llama3_70b_pretrain_64gpu_h100_bf16_config,
    llama3_70b_pretrain_64gpu_h100_fp8cs_config,
)
from megatron.bridge.training.config import ConfigContainer


# Map short precision names used in new recipes to the strings used by get_perf_optimized_recipe
_PRECISION_MAP = {
    "bf16": "bf16",
    "fp8cs": "fp8_cs",
    "fp8mx": "fp8_mx",
    "nvfp4": "nvfp4",
}


def _get_old_recipe(model: str, gpu: str, precision_short: str) -> ConfigContainer:
    """Build a ConfigContainer via the old perf pipeline."""
    precision = _PRECISION_MAP[precision_short]
    return get_perf_optimized_recipe(
        model_family_name="llama",
        model_recipe_name=model,
        train_task="pretrain",
        gpu=gpu,
        compute_dtype=precision,
        mock=True,
        config_variant="v1",
    )


def _compare_dataclass(old, new, path: str = "") -> list[str]:
    """Recursively compare two dataclass instances, return list of mismatches."""
    diffs = []
    if not is_dataclass(old) or not is_dataclass(new):
        if old != new:
            diffs.append(f"{path}: {old!r} != {new!r}")
        return diffs
    for f in fields(old):
        old_val = getattr(old, f.name, None)
        new_val = getattr(new, f.name, None)
        field_path = f"{path}.{f.name}" if path else f.name
        if is_dataclass(old_val) and is_dataclass(new_val):
            diffs.extend(_compare_dataclass(old_val, new_val, field_path))
        elif old_val != new_val:
            diffs.append(f"{field_path}: {old_val!r} != {new_val!r}")
    return diffs


# Fields that matter for perf recipe equivalence
_MODEL_FIELDS = [
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "context_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "sequence_parallel",
    "seq_length",
    "cuda_graph_impl",
    "cuda_graph_scope",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "cpu_offloading",
    "cpu_offloading_weights",
    "cpu_offloading_num_layers",
    "init_model_with_meta_device",
    "gradient_accumulation_fusion",
    "should_pad_vocab",
    "apply_rope_fusion",
    "cross_entropy_fusion_impl",
    "use_te_rng_tracker",
]

_DDP_FIELDS = [
    "use_megatron_fsdp",
    "data_parallel_sharding_strategy",
    "keep_fp8_transpose_cache",
    "average_in_collective",
    "nccl_ub",
    "fsdp_manual_registration",
    "fsdp_double_buffer",
    "suggested_communication_unit_size",
    "check_for_nan_in_grad",
    "check_for_large_grads",
    "grad_reduce_in_fp32",
    "overlap_grad_reduce",
    "overlap_param_gather",
]


def assert_configs_equal(old: ConfigContainer, new: ConfigContainer) -> None:
    """Compare key fields of two ConfigContainers, raising on any mismatch."""
    diffs = []

    # Model fields
    for f in _MODEL_FIELDS:
        old_val = getattr(old.model, f, "MISSING")
        new_val = getattr(new.model, f, "MISSING")
        if old_val != new_val:
            diffs.append(f"model.{f}: old={old_val!r}  new={new_val!r}")

    # Training fields
    for f in ["train_iters", "global_batch_size", "micro_batch_size", "eval_iters"]:
        old_val = getattr(old.train, f)
        new_val = getattr(new.train, f)
        if old_val != new_val:
            diffs.append(f"train.{f}: old={old_val!r}  new={new_val!r}")

    # DDP fields
    for f in _DDP_FIELDS:
        old_val = getattr(old.ddp, f, "MISSING")
        new_val = getattr(new.ddp, f, "MISSING")
        if old_val != new_val:
            diffs.append(f"ddp.{f}: old={old_val!r}  new={new_val!r}")

    # Scheduler
    for f in ["lr_decay_iters", "lr_warmup_iters"]:
        old_val = getattr(old.scheduler, f)
        new_val = getattr(new.scheduler, f)
        if old_val != new_val:
            diffs.append(f"scheduler.{f}: old={old_val!r}  new={new_val!r}")

    # Checkpoint
    if old.checkpoint.save != new.checkpoint.save:
        diffs.append(f"checkpoint.save: old={old.checkpoint.save!r}  new={new.checkpoint.save!r}")
    if old.checkpoint.load != new.checkpoint.load:
        diffs.append(f"checkpoint.load: old={old.checkpoint.load!r}  new={new.checkpoint.load!r}")

    # Logger
    if old.logger.log_interval != new.logger.log_interval:
        diffs.append(f"logger.log_interval: old={old.logger.log_interval!r}  new={new.logger.log_interval!r}")
    if old.logger.tensorboard_dir != new.logger.tensorboard_dir:
        diffs.append(f"logger.tensorboard_dir: old={old.logger.tensorboard_dir!r}  new={new.logger.tensorboard_dir!r}")

    # Tokenizer
    if old.tokenizer.vocab_size != new.tokenizer.vocab_size:
        diffs.append(f"tokenizer.vocab_size: old={old.tokenizer.vocab_size!r}  new={new.tokenizer.vocab_size!r}")

    # Mixed precision (compare as dataclass)
    mp_diffs = _compare_dataclass(old.mixed_precision, new.mixed_precision, "mixed_precision")
    diffs.extend(mp_diffs)

    # Comm overlap
    if old.comm_overlap is not None and new.comm_overlap is not None:
        co_fields = ["tp_comm_overlap", "defer_embedding_wgrad_compute"]
        for f in co_fields:
            old_val = getattr(old.comm_overlap, f, "MISSING")
            new_val = getattr(new.comm_overlap, f, "MISSING")
            if old_val != new_val:
                diffs.append(f"comm_overlap.{f}: old={old_val!r}  new={new_val!r}")
        # tp_comm_overlap_cfg identity check (should be same object)
        if old.comm_overlap.tp_comm_overlap_cfg is not new.comm_overlap.tp_comm_overlap_cfg:
            diffs.append("comm_overlap.tp_comm_overlap_cfg: objects differ")
    elif (old.comm_overlap is None) != (new.comm_overlap is None):
        diffs.append(f"comm_overlap: old={old.comm_overlap!r}  new={new.comm_overlap!r}")

    # RNG
    if hasattr(old.rng, "te_rng_tracker") and hasattr(new.rng, "te_rng_tracker"):
        if old.rng.te_rng_tracker != new.rng.te_rng_tracker:
            diffs.append(f"rng.te_rng_tracker: old={old.rng.te_rng_tracker!r}  new={new.rng.te_rng_tracker!r}")

    # Rerun state machine
    if old.rerun_state_machine.check_for_nan_in_loss != new.rerun_state_machine.check_for_nan_in_loss:
        diffs.append(
            f"rerun_state_machine.check_for_nan_in_loss: "
            f"old={old.rerun_state_machine.check_for_nan_in_loss!r}  "
            f"new={new.rerun_state_machine.check_for_nan_in_loss!r}"
        )

    if diffs:
        msg = "ConfigContainer mismatch:\n" + "\n".join(f"  {d}" for d in diffs)
        raise AssertionError(msg)


# ─── Parametrized test cases ────────────────────────────────────────────


_LLAMA3_70B_PRETRAIN_CASES = [
    ("gb300", "bf16", llama3_70b_pretrain_64gpu_gb300_bf16_config),
    ("gb300", "fp8cs", llama3_70b_pretrain_64gpu_gb300_fp8cs_config),
    ("gb300", "fp8mx", llama3_70b_pretrain_64gpu_gb300_fp8mx_config),
    ("gb300", "nvfp4", llama3_70b_pretrain_64gpu_gb300_nvfp4_config),
    ("gb200", "bf16", llama3_70b_pretrain_64gpu_gb200_bf16_config),
    ("gb200", "fp8cs", llama3_70b_pretrain_64gpu_gb200_fp8cs_config),
    ("gb200", "fp8mx", llama3_70b_pretrain_64gpu_gb200_fp8mx_config),
    ("gb200", "nvfp4", llama3_70b_pretrain_64gpu_gb200_nvfp4_config),
    ("b300", "bf16", llama3_70b_pretrain_64gpu_b300_bf16_config),
    ("b300", "fp8cs", llama3_70b_pretrain_64gpu_b300_fp8cs_config),
    ("b300", "fp8mx", llama3_70b_pretrain_64gpu_b300_fp8mx_config),
    ("b300", "nvfp4", llama3_70b_pretrain_64gpu_b300_nvfp4_config),
    ("b200", "bf16", llama3_70b_pretrain_64gpu_b200_bf16_config),
    ("b200", "fp8cs", llama3_70b_pretrain_64gpu_b200_fp8cs_config),
    ("b200", "fp8mx", llama3_70b_pretrain_64gpu_b200_fp8mx_config),
    ("b200", "nvfp4", llama3_70b_pretrain_64gpu_b200_nvfp4_config),
    ("h100", "bf16", llama3_70b_pretrain_64gpu_h100_bf16_config),
    ("h100", "fp8cs", llama3_70b_pretrain_64gpu_h100_fp8cs_config),
]

_LLAMA3_8B_PRETRAIN_CASES = [
    ("r100", "bf16", llama3_8b_pretrain_8gpu_r100_bf16_config),
    ("r100", "fp8cs", llama3_8b_pretrain_8gpu_r100_fp8cs_config),
    ("r100", "fp8mx", llama3_8b_pretrain_8gpu_r100_fp8mx_config),
    ("r100", "nvfp4", llama3_8b_pretrain_8gpu_r100_nvfp4_config),
    ("gb300", "bf16", llama3_8b_pretrain_8gpu_gb300_bf16_config),
    ("gb300", "fp8cs", llama3_8b_pretrain_8gpu_gb300_fp8cs_config),
    ("gb300", "fp8mx", llama3_8b_pretrain_8gpu_gb300_fp8mx_config),
    ("gb300", "nvfp4", llama3_8b_pretrain_8gpu_gb300_nvfp4_config),
    ("gb200", "bf16", llama3_8b_pretrain_8gpu_gb200_bf16_config),
    ("gb200", "fp8cs", llama3_8b_pretrain_8gpu_gb200_fp8cs_config),
    ("gb200", "fp8mx", llama3_8b_pretrain_8gpu_gb200_fp8mx_config),
    ("gb200", "nvfp4", llama3_8b_pretrain_8gpu_gb200_nvfp4_config),
    ("b300", "bf16", llama3_8b_pretrain_8gpu_b300_bf16_config),
    ("b300", "fp8cs", llama3_8b_pretrain_8gpu_b300_fp8cs_config),
    ("b300", "fp8mx", llama3_8b_pretrain_8gpu_b300_fp8mx_config),
    ("b300", "nvfp4", llama3_8b_pretrain_8gpu_b300_nvfp4_config),
    ("b200", "bf16", llama3_8b_pretrain_8gpu_b200_bf16_config),
    ("b200", "fp8cs", llama3_8b_pretrain_8gpu_b200_fp8cs_config),
    ("b200", "fp8mx", llama3_8b_pretrain_8gpu_b200_fp8mx_config),
    ("b200", "nvfp4", llama3_8b_pretrain_8gpu_b200_nvfp4_config),
    ("h100", "bf16", llama3_8b_pretrain_8gpu_h100_bf16_config),
    ("h100", "fp8cs", llama3_8b_pretrain_8gpu_h100_fp8cs_config),
]


@pytest.mark.unit
class TestLlama3PerfEquivalence:
    """Old perf path and new flat recipe must produce identical ConfigContainer."""

    @pytest.mark.parametrize(
        "gpu,precision,new_fn",
        _LLAMA3_70B_PRETRAIN_CASES,
        ids=[f"70b_{gpu}_{prec}" for gpu, prec, _ in _LLAMA3_70B_PRETRAIN_CASES],
    )
    def test_llama3_70b_pretrain(self, gpu, precision, new_fn):
        old_cfg = _get_old_recipe("llama3_70b", gpu, precision)
        new_cfg = new_fn()
        assert_configs_equal(old_cfg, new_cfg)

    @pytest.mark.parametrize(
        "gpu,precision,new_fn",
        _LLAMA3_8B_PRETRAIN_CASES,
        ids=[f"8b_{gpu}_{prec}" for gpu, prec, _ in _LLAMA3_8B_PRETRAIN_CASES],
    )
    def test_llama3_8b_pretrain(self, gpu, precision, new_fn):
        old_cfg = _get_old_recipe("llama3_8b", gpu, precision)
        new_cfg = new_fn()
        assert_configs_equal(old_cfg, new_cfg)
