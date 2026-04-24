# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""DeepSeek-V3 pretraining recipe using `MambaModel` as the base class.

Mirror of :mod:`megatron.bridge.recipes.deepseek.deepseek_v3` but constructs
the provider via :class:`DeepSeekV3MambaBridge`, so the model is instantiated
as a Megatron-Core `MambaModel` (hybrid stack with MLA attention + MLP/MoE)
instead of `GPTModel`. See the bridge module for the `+-` / `+E` hybrid
layer layout that this recipe produces.
"""

import os

import torch
from transformers import AutoConfig

from megatron.bridge.models.deepseek.deepseek_v3_mamba_bridge import DeepSeekV3MambaBridge
from megatron.bridge.models.hf_pretrained.causal_lm import _ConfigOnlyPretrainedShim
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.deepseek.deepseek_v3 import set_deepseek_v3_pipeline_model_parallel_layout
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


#: Env var override for the HF model path (useful in sandboxed clusters that
#: cannot reach huggingface.co — point it at a local checkpoint directory).
_HF_PATH_ENV = "DEEPSEEK_V3_HF_PATH"


def _build_mamba_provider(hf_model_name: str | None = None):
    """Construct a :class:`DeepSeekV3MambaProvider` from an HF model id or path.

    `AutoBridge.from_hf_pretrained` would dispatch to the registered
    :class:`DeepSeekV3Bridge` (GPT-backed), so we bypass it and drive the
    bridge directly. ``hf_model_name`` defaults to the ``DEEPSEEK_V3_HF_PATH``
    environment variable, falling back to ``deepseek-ai/DeepSeek-V3``.
    """
    if hf_model_name is None:
        hf_model_name = os.environ.get(_HF_PATH_ENV, "deepseek-ai/DeepSeek-V3")
    hf_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
    shim = _ConfigOnlyPretrainedShim(hf_config)

    bridge = DeepSeekV3MambaBridge()
    bridge.hf_config = hf_config
    return bridge.provider_bridge(shim)


def deepseek_v3_mamba_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3 with a Mamba/Hybrid backbone.

    Parallelism defaults mirror :func:`deepseek_v3_pretrain_config`
    (TP=2, PP=16, EP=64), but note that the hybrid layer count is ``2 *
    num_hidden_layers`` (122 layers for DSv3), which may require revisiting
    the pipeline-parallel layout.
    """
    cfg = _pretrain_common()

    # Model config — built directly from our Mamba-backed bridge.
    cfg.model = _build_mamba_provider()

    # Tokenizer - uses NullTokenizer by default (no HF tokenizer download needed)
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # MTP (Multi-Token Prediction) configuration
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # Model-specific settings
    cfg.model.init_method_std = 0.006
    cfg.model.rotary_base = 10000.0
    cfg.model.rotary_scaling_factor = 40
    cfg.model.rotary_base = float(cfg.model.rotary_base)
    cfg.model.rotary_scaling_factor = int(cfg.model.rotary_scaling_factor)

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    cfg.model.num_layers_in_first_pipeline_stage = None
    cfg.model.num_layers_in_last_pipeline_stage = None

    # Set pipeline layout (derived from mtp_num_layers + pp/vp sizes).
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 1_000_000
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer settings
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = True

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg
