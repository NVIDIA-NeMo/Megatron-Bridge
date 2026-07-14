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

"""H100 library recipes for Nemotron 3 Nano."""

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.nemotronh._nemotron_3_nano import (
    _nemotron_3_nano_peft_reference_config,
    _nemotron_3_nano_pretrain_reference_config,
    _nemotron_3_nano_sft_reference_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotron_3_nano_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano pre-training config for 8 H100 GPUs.

    The TP=1, EP=8 layout and HybridEP dispatcher follow the H100 performance
    recipe. Selective recompute keeps that topology practical, while scoped
    CUDA graphs cover only fixed-shape pre-training modules.

    Returns:
        ConfigContainer: H100 BF16 pre-training configuration.
    """
    cfg = _nemotron_3_nano_pretrain_reference_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba"])
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    # TP communication overlap requires TP > 1 and sequence parallelism.
    cfg.comm_overlap.tp_comm_overlap = False

    return cfg


def _apply_h100_finetune_execution_config(cfg: ConfigContainer) -> None:
    """Apply the evidenced H100 packed-finetuning execution contract."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    # No H100 packed-finetuning perf reference currently validates HybridEP.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_router_force_load_balancing = False


def nemotron_3_nano_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano SFT config for 8 H100 GPUs.

    Packed SFT retains the established DeepEP dispatcher and eager execution;
    no H100 packed-finetuning perf reference currently proves HybridEP or CUDA
    graphs for this workload.

    Returns:
        ConfigContainer: H100 BF16 SFT configuration.
    """
    cfg = _nemotron_3_nano_sft_reference_config()
    _apply_h100_finetune_execution_config(cfg)
    return cfg


def nemotron_3_nano_peft_8gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the Nemotron 3 Nano PEFT config for 8 H100 GPUs.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer: H100 BF16 PEFT configuration.
    """
    cfg = _nemotron_3_nano_peft_reference_config(peft_scheme=peft_scheme)
    _apply_h100_finetune_execution_config(cfg)
    return cfg


__all__ = [
    "nemotron_3_nano_peft_8gpu_h100_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_h100_bf16_config",
    "nemotron_3_nano_sft_8gpu_h100_bf16_config",
]
