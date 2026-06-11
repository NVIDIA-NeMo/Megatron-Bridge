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

"""GPU-specific Llama 2 70B LoRA finetune presets for the MLPerf v6.0 LoRA workload.

Wraps llama2_70b_lora_config() with GPU/precision overrides; MLPERF_* variants additionally invoke set_llama2_mlperf_parity_overrides().
"""

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.llama import llama2_70b_lora_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def _set_llama2_common_peft_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for Llama 2 LoRA configs (vocab=32000, smaller than Llama 3)."""
    cfg.tokenizer.vocab_size = 32000
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.disable_parameter_transpose_cache = True

    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True


def set_llama2_mlperf_parity_overrides(cfg: ConfigContainer, precision: str) -> None:
    """Apply MLPerf v6.0 reference parity overrides to a Llama 2 LoRA config (universal recipe knobs + NVFP4 FP8-attn overlay)."""
    # Universal recipe overrides ----------------------------------------------
    if hasattr(cfg.model, "use_transformer_engine_op_fuser"):
        cfg.model.use_transformer_engine_op_fuser = True
    if hasattr(cfg, "comm_overlap") and cfg.comm_overlap is not None:
        cfg.comm_overlap.bucket_size = 768 * 1024 * 1024
    if hasattr(cfg.model, "fused_single_qkv_rope"):
        cfg.model.fused_single_qkv_rope = True
    if hasattr(cfg.model, "tp_only_amax_red"):
        cfg.model.tp_only_amax_red = True
    if hasattr(cfg.model, "cpu_offloading") and cfg.model.cpu_offloading:
        cfg.model.cpu_offloading = False
    if hasattr(cfg.model, "cpu_offloading_num_layers"):
        cfg.model.cpu_offloading_num_layers = None
    if hasattr(cfg.model, "use_te_rng_tracker"):
        cfg.model.use_te_rng_tracker = True
    if hasattr(cfg, "rng") and hasattr(cfg.rng, "te_rng_tracker"):
        cfg.rng.te_rng_tracker = True

    # Precision-specific overlay ----------------------------------------------
    if precision == "nvfp4":
        if hasattr(cfg.mixed_precision, "fp8_dot_product_attention"):
            cfg.mixed_precision.fp8_dot_product_attention = False
        if hasattr(cfg.ddp, "fp4_param_gather"):
            cfg.ddp.fp4_param_gather = False
        if hasattr(cfg.mixed_precision, "fp4_param_gather"):
            cfg.mixed_precision.fp4_param_gather = False

    # Context parallelism requires per-token loss during finetuning. Megatron asserts this in
    # training/config.py ("When finetuning with CP>1, calculate_per_token_loss must be True"),
    # and average_in_collective must be False whenever per-token loss is enabled. The nvfp4 LoRA
    # variant runs CP=2; this is independent of the FSDP path (which is disabled here).
    if cfg.model.context_parallel_size > 1:
        cfg.model.calculate_per_token_loss = True
        cfg.ddp.average_in_collective = False


def _is_mlperf_variant(config_variant: str) -> bool:
    """True if config_variant selects an MLPerf v6.0 reference parity variant."""
    return config_variant.lower().startswith("mlperf")


def llama2_70b_lora_config_gb200(precision: str = "fp8_cs", config_variant: str = "v1") -> ConfigContainer:
    """GB200 Llama 2 70B LoRA preset (MLPerf v6.0 LoRA workload reference shape)."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama2_70b",
        task="lora",
        gpu="gb200",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama2_70b_lora_config(peft_scheme="lora")
    cfg.mixed_precision = precision_config
    _set_llama2_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # set_workload_base_configs finalizes CP (e.g. NVFP4 V1 sets CP=2), so re-derive the
    # packed-sequence padding here: _peft_common requires pad_seq_to_mult = CP*2 when CP>1.
    # The recipe-level adjustment in llama2_70b_lora_config() runs before CP is overridden,
    # so without this the NVFP4 CP>1 path would keep pad_seq_to_mult=1.
    if cfg.model.context_parallel_size > 1 and cfg.dataset.packed_sequence_specs is not None:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    # Enable pad_cu_seqlens for CUDA graphs compatibility with packed sequences.
    if cfg.dataset.packed_sequence_specs is not None:
        cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    if cfg.dataset.dataset_kwargs is None:
        cfg.dataset.dataset_kwargs = {}
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True

    if _is_mlperf_variant(config_variant):
        set_llama2_mlperf_parity_overrides(cfg, precision)

    return cfg


def llama2_70b_lora_config_gb300(precision: str = "fp8_cs", config_variant: str = "v1") -> ConfigContainer:
    """GB300 Llama 2 70B LoRA preset (MLPerf v6.0 LoRA workload reference shape)."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama2_70b",
        task="lora",
        gpu="gb300",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama2_70b_lora_config(peft_scheme="lora")
    cfg.mixed_precision = precision_config
    _set_llama2_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    # set_workload_base_configs finalizes CP (e.g. NVFP4 V1 sets CP=2), so re-derive the
    # packed-sequence padding here: _peft_common requires pad_seq_to_mult = CP*2 when CP>1.
    # The recipe-level adjustment in llama2_70b_lora_config() runs before CP is overridden,
    # so without this the NVFP4 CP>1 path would keep pad_seq_to_mult=1.
    if cfg.model.context_parallel_size > 1 and cfg.dataset.packed_sequence_specs is not None:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    if cfg.dataset.packed_sequence_specs is not None:
        cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    if cfg.dataset.dataset_kwargs is None:
        cfg.dataset.dataset_kwargs = {}
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True

    if _is_mlperf_variant(config_variant):
        set_llama2_mlperf_parity_overrides(cfg, precision)

    return cfg
