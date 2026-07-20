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

"""GB200 pretraining recipe for Nemotron 3 Nano."""

from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotron_3_nano_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano BF16 pretraining config for eight GB200 GPUs.

    The recipe retains the established optimizer, scheduler, routing, and BF16
    contracts. It applies the validated GB200 TP1/EP8 HybridEP topology and
    uses a 4,096-token sequence length for the paired NeMo-CI convergence
    workload.

    Returns:
        GB200 BF16 pretraining configuration.
    """
    cfg = nemotron_3_nano_pretrain_config()

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False

    # Match the validated GB200 performance recipe's TE-scoped graph set.
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    # Retain performance-recipe parity. Nemotron 3 Nano uses no positional
    # embeddings, so this remains a no-op unless the architecture changes.
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.rerun_state_machine.check_for_nan_in_loss = False
    cfg.ddp.check_for_nan_in_grad = False

    # Keep BF16 compute while reducing gradients in BF16 instead of FP32.
    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # TP communication overlap requires TP > 1 and sequence parallelism.
    if cfg.comm_overlap is not None:
        cfg.comm_overlap.tp_comm_overlap = False

    return cfg


# NeMo-CI appends ``_pretrain_config`` to MODEL_RECIPE_NAME. This explicit
# alias lets the GB200 release case select the hardware recipe without changing
# the legacy ``nemotron_3_nano_pretrain_config`` default.
nemotron_3_nano_gb200_pretrain_config = nemotron_3_nano_pretrain_8gpu_gb200_bf16_config


__all__ = [
    "nemotron_3_nano_gb200_pretrain_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
]
