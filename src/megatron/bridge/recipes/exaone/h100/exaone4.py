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
"""EXAONE 4.0 H100 training recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


_HF_PATH = "LGAI-EXAONE/EXAONE-4.0-1.2B"


def _set_optimizer_precision(cfg: ConfigContainer) -> None:
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32


def _apply_exaone4_common(cfg: ConfigContainer) -> None:
    cfg.model = AutoBridge.from_hf_pretrained(_HF_PATH).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _HF_PATH

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    _set_optimizer_precision(cfg)


def exaone4_1p2b_pretrain_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for EXAONE 4.0 1.2B."""
    cfg = _pretrain_common()
    _apply_exaone4_common(cfg)
    cfg.dataset.num_workers = 8
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def exaone4_1p2b_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for EXAONE 4.0 1.2B."""
    cfg = _sft_common()
    _apply_exaone4_common(cfg)
    cfg.model.seq_length = 2048
    cfg.train.global_batch_size = 128
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def exaone4_1p2b_peft_1gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for EXAONE 4.0 1.2B."""
    cfg = _peft_common()
    _apply_exaone4_common(cfg)
    cfg.model.seq_length = 2048
    cfg.peft = default_peft_config(peft_scheme)
    cfg.train.global_batch_size = 128
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "exaone4_1p2b_peft_1gpu_h100_bf16_config",
    "exaone4_1p2b_pretrain_1gpu_h100_bf16_config",
    "exaone4_1p2b_sft_1gpu_h100_bf16_config",
]
