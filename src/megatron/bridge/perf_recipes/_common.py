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

"""Shared helpers for flat performance benchmark recipes.

``_benchmark_common`` applies throughput-measurement defaults.
``_perf_precision`` returns a mixed-precision config for a given dtype.
"""

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_mxfp8_mixed,
    bf16_with_nvfp4_mixed,
)


def _benchmark_common(cfg: ConfigContainer, cross_entropy_impl: str = "te") -> None:
    """Apply benchmark-mode defaults that prioritize throughput measurement over convergence.

    Intended for performance benchmark recipes only. Sets short training runs,
    disables checkpointing/eval, tunes scheduler, and enables perf-oriented kernels.

    Must stay in sync with ``_set_common_perf_overrides`` in
    ``scripts/performance/utils/overrides.py``.

    Individual recipes may override any of these after calling this function
    (e.g. Kimi K2 sets ``grad_reduce_in_fp32 = True``).
    """
    cfg.train.train_iters = 50
    cfg.train.eval_iters = 0

    cfg.checkpoint.save = None

    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = None

    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.check_for_large_grads = False

    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.scheduler.lr_warmup_iters = 10

    if hasattr(cfg.model, "use_transformer_engine_op_fuser") and cfg.model.use_transformer_engine_op_fuser:
        cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = cross_entropy_impl

    if not isinstance(cfg.mixed_precision, str):
        cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cuda_impl = getattr(cfg.model, "cuda_graph_impl", None)
    if cuda_impl is not None:
        if cuda_impl != "none":
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
        else:
            cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    if getattr(cfg.model, "moe_flex_dispatcher_backend", None) == "hybridep":
        cfg.model.moe_hybridep_num_sms = 32


def _perf_precision(compute_dtype: str):
    """Return mixed-precision config tuned for perf benchmarks.

    Identical to ``scripts/performance/utils/precision.get_precision_config``
    but importable from the library side.  Always sets
    ``grad_reduce_in_fp32=False`` so that callers that replace
    ``cfg.mixed_precision`` after ``_benchmark_common()`` still get the
    benchmark-mode default.
    """
    if compute_dtype == "bf16":
        cfg = bf16_mixed()
    elif compute_dtype == "fp8_cs":
        cfg = bf16_with_fp8_current_scaling_mixed()
        cfg.first_last_layers_bf16 = False
    elif compute_dtype == "fp8_mx":
        cfg = bf16_with_mxfp8_mixed()
    elif compute_dtype == "nvfp4":
        cfg = bf16_with_nvfp4_mixed()
    else:
        raise ValueError(f"Unknown compute_dtype: {compute_dtype}")
    cfg.grad_reduce_in_fp32 = False
    return cfg
