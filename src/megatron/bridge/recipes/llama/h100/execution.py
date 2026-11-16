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

"""Reusable H100 execution choices for the Llama 3 8B library recipes.

These helpers preserve data, batch sizes, optimizer precision, schedule, loss
checks and checkpoints. Call after selecting compute precision. They do not
establish memory fit or convergence for a new workload.
"""

from typing import cast

from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


def llama3_8b_pretrain_bf16(cfg: ConfigContainer) -> None:
    """Select TP1/PP1/CP2 and communication settings for BF16 pretraining."""
    # These named library recipes return GPT configs, not the other ConfigContainer model families.
    model = cast(GPTModelConfig, cfg.model)
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 2
    model.virtual_pipeline_model_parallel_size = None
    model.sequence_parallel = False
    model.cuda_graph_scope = []
    model.moe_token_dispatcher_type = "alltoall"
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }


def llama3_8b_pretrain_fp8(cfg: ConfigContainer) -> None:
    """Select the existing CP1/full-recompute layout, requiring explicit FP8.

    This preserves the measured benchmark layout during migration. It is not
    a new recommendation to use full recompute or to lower compute precision.
    """
    # These named library recipes return GPT configs, not the other ConfigContainer model families.
    model = cast(GPTModelConfig, cfg.model)
    if not getattr(cfg.mixed_precision, "fp8", None):
        raise ValueError("The Llama CP1 FP8 execution layout requires an explicit FP8 precision config.")
    llama3_8b_pretrain_bf16(cfg)
    model.context_parallel_size = 1
    model.recompute_granularity = "full"
    model.recompute_method = "block"
    model.recompute_num_layers = 5
    cfg.env_vars.update(
        {
            "CUDA_DEVICE_MAX_CONNECTIONS": 32,
            "NCCL_CTA_POLICY": 1,
        }
    )


def llama3_8b_sft(cfg: ConfigContainer) -> None:
    """Select the existing TP1/PP1/CP1 SFT layout without changing its workload."""
    # These named library recipes return GPT configs, not the other ConfigContainer model families.
    model = cast(GPTModelConfig, cfg.model)
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.virtual_pipeline_model_parallel_size = None
    model.cuda_graph_impl = "none"
    model.cuda_graph_scope = []
    model.moe_token_dispatcher_type = "alltoall"
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
