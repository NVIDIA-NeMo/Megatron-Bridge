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

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


GLM52_HF_MODEL_ID = "zai-org/GLM-5.2"


def glm52_753b_pretrain_512gpu_h100_bf16_config() -> ConfigContainer:
    """Return the H100 canonical pre-training config for GLM-5.2 753B MoE.

    The provider is derived from the official HF config through ``GLM5Bridge``.
    DSA requires CP=1 (the indexer selects top-k over the full sequence) and
    apply_rope_fusion=False (Megatron-Core asserts RoPE fusion is unsupported for
    the DSA attention variant).

    Recommended starting parallelism: TP=2, PP=16, EP=32 (512 GPUs).
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(GLM52_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Mock-data pretraining uses NullTokenizer while preserving the model vocab size.
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = getattr(cfg.model, "vocab_size", DEFAULT_NULL_TOKENIZER_VOCAB_SIZE)

    seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.seq_length = seq_length
    cfg.dataset.num_workers = 8
    if hasattr(cfg.dataset, "sequence_length"):
        cfg.dataset.sequence_length = seq_length
    cfg.model.seq_length = seq_length

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # GLM-5.2 has one MTP layer; keep it off for pretraining until MTP training
    # has been validated end-to-end (conversion is supported via GLM5Bridge).
    cfg.model.mtp_num_layers = None
    cfg.model.mtp_loss_scaling_factor = None

    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_shared_expert_overlap = True
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_quantization = False

    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 25
    cfg.train.manual_gc_eval = 25
    cfg.validation.eval_interval = 2000

    cfg.scheduler.lr_warmup_iters = 2000
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.min_lr = 1e-5
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"
    cfg.ddp.use_megatron_fsdp = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.dist.enable_megatron_core_experimental = True

    cfg.env_vars = {**COMMON_RECIPE_ENV_VARS}

    return cfg


__all__ = [
    "glm52_753b_pretrain_512gpu_h100_bf16_config",
]
