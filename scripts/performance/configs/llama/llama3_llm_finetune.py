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

import logging

from utils.overrides import set_workload_base_configs
from utils.precision import get_precision_config
from utils.utils import get_workload_base_config

from megatron.bridge.recipes.llama import (
    llama3_8b_sft_config,
    llama3_70b_peft_config,
    llama3_70b_sft_config,
)
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


# Llama3 8B Finetune configs ---------------------------------------------------------


def set_llama3_common_peft_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3 8B PEFT configs."""
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.disable_parameter_transpose_cache = True

    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True


def llama3_8b_sft_config_gb200(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """GB200, SFT config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        task="sft",
        gpu="gb200",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = precision_config
    seq_length = 16384
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def llama3_8b_sft_config_h100(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """H100, SFT config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_8b",
        task="sft",
        gpu="h100",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    return cfg


def llama3_70b_sft_config_gb300(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """GB300, SFT config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="sft",
        gpu="gb300",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    # Enable pad_cu_seqlens for CUDA graphs compatibility with packed sequences.
    # This ensures consistent cu_seqlens tensor shapes across batches, which is required
    # for CUDA graphs and avoids NaN issues in attention kernels.
    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True

    return cfg


def llama3_70b_sft_config_gb200(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """GB200, SFT config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="sft",
        gpu="gb200",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    return cfg


def llama3_70b_sft_config_h100(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """H100, SFT config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="sft",
        gpu="h100",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1),
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    return cfg


def llama3_70b_lora_config_gb300(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """GB300, LORA config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="lora",
        gpu="gb300",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    # Override target_modules to only apply LoRA to QKV
    cfg.peft.target_modules = ["linear_qkv"]

    # Enable pad_cu_seqlens for CUDA graphs compatibility with packed sequences.
    # This ensures consistent cu_seqlens tensor shapes across batches, which is required
    # for CUDA graphs and avoids NaN issues in attention kernels.
    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True

    return cfg


def llama3_70b_lora_config_gb200(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """GB200, LORA config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="lora",
        gpu="gb200",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    # BF16 uses seq_length=2048, FP8 variants use seq_length=4096
    seq_length = 2048 if precision.lower() == "bf16" else 4096

    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = precision_config
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    # Enable pad_cu_seqlens for CUDA graphs compatibility with packed sequences.
    # This ensures consistent cu_seqlens tensor shapes across batches, which is required
    # for CUDA graphs and avoids NaN issues in attention kernels.
    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    # Override target_modules to only apply LoRA to QKV
    cfg.peft.target_modules = ["linear_qkv"]

    return cfg


def llama3_70b_lora_config_h100(precision: str = "bf16", config_variant: str = "v1") -> ConfigContainer:
    """H100, LORA config."""
    base_cfg = get_workload_base_config(
        model_family_name="llama",
        model_recipe_name="llama3_70b",
        task="lora",
        gpu="h100",
        compute_dtype=precision.upper(),
        config_variant=config_variant,
    )
    precision_config = get_precision_config(precision)

    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = precision_config
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset.seq_length = seq_length
    cfg.dataset.packed_sequence_specs.packed_sequence_size = seq_length
    set_llama3_common_peft_configs(cfg)
    set_workload_base_configs(cfg, base_cfg)
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    # Override target_modules to only apply LoRA to QKV
    cfg.peft.target_modules = ["linear_qkv"]

    return cfg
