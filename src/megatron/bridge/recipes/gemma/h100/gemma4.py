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

"""Gemma 4 Dense (E4B) pre-training recipe."""

import os
from contextlib import contextmanager

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import ConfigContainer


_GEMMA4_E4B_HF_PATH = "google/gemma-4-E4B-it"


@contextmanager
def _gemma4_text_conversion_mode():
    previous_mode = os.environ.get("GEMMA4_CONVERSION_MODE")
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    try:
        yield
    finally:
        if previous_mode is None:
            os.environ.pop("GEMMA4_CONVERSION_MODE", None)
        else:
            os.environ["GEMMA4_CONVERSION_MODE"] = previous_mode


def gemma4_e4b_pretrain_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Gemma 4 E4B (Dense, ~3.8B parameters).

    Architecture (Gemma 4 E4B):
    - 42 layers, hidden_size=2560, ffn_hidden_size=10240
    - 8 attention heads, 2 KV heads (sliding), 2 KV heads (global, head_dim=512)
    - Sliding-window / global attention interleaved (skip_freq=6)
    - Dual RoPE: sliding θ=10 000, global θ=1 000 000 with 0.25 partial rotation
    - Per-Layer Embeddings (PLE, vocab=262144, dim=256)
    - Shared KV cache across the last 18 layers
    - Local (non-TE) transformer spec via ``get_gemma4_layer_spec``

    Default parallelism: TP=2, PP=1, seq_length=4096.
    Override at launch time with Hydra-style args, e.g.::

        checkpoint.pretrained_checkpoint=/path/to/megatron-ckpt
        checkpoint.save=/path/to/save
        train.train_iters=1000
        model.seq_length=4096
    """
    cfg = _pretrain_common()

    # gemma-4-E4B-it is a ConditionalGeneration HF model; force the text-only
    # Gemma4 bridge path so this pre-training recipe uses Gemma4DenseProvider.
    with _gemma4_text_conversion_mode():
        cfg.model = AutoBridge.from_hf_pretrained(_GEMMA4_E4B_HF_PATH).to_megatron_provider(load_weights=False)

    # Tokenizer — NullTokenizer for mock pre-training; override for real data
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset — mock data by default; override dataset.blend for real data
    cfg.dataset.blend = None
    cfg.dataset.seq_length = 4096

    # Parallelism: TP=2 to match the E4B parity / conversion setup
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096

    # Training
    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    cfg.validation.eval_interval = 200
    cfg.validation.eval_iters = 10

    cfg.scheduler.lr_warmup_iters = 100

    # Implementation — Dense E4B uses the local (non-TE) spec
    cfg.model.transformer_impl = "local"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel / fusion settings — disable TE-specific fusions for the local spec
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.masked_softmax_fusion = False
    cfg.model.gradient_accumulation_fusion = False

    # Memory saving (disabled; enable recompute for larger batches)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Optimizer precision
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # DDP
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "gemma4_e4b_pretrain_2gpu_h100_bf16_config",
]
