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

"""
Qwen3 recipe using the new flattened layout with _pretrain_common.

This file demonstrates the new pattern where recipes:
1. Call _pretrain_common() to get base config
2. Override model-specific settings directly on the returned config
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.config import ConfigContainer


def qwen3_600m_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 0.6B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-0.6B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-0.6B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8  # --num-workers for dataloader

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"  # default in mcore's transformer_config.py

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"  # default in mcore's transformer_config.py
    cfg.model.cuda_graph_scope = "full"  # default in mcore's transformer_config.py
    cfg.model.cuda_graph_warmup_steps = 3  # default in mcore's transformer_config.py

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.cross_entropy_loss_fusion = True # Defaults to True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading)
    # =========================================================================
    cfg.model.recompute_granularity = None  # Not enabled for 600M, defaults to None in mcore's transformer_config.py
    cfg.model.recompute_modules = None # Defaults to None in mcore's transformer_config.py
    cfg.model.fine_grained_activation_offloading = False # Defaults to False in mcore's transformer_config.py
    cfg.model.offload_modules = None # Defaults to None in mcore's transformer_config.py

    # =========================================================================
    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # =========================================================================
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False  # default in mcore's OptimizerConfig
    cfg.optimizer.main_grads_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.main_params_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_sq_dtype = torch.float32  # default in mcore's OptimizerConfig

    # =========================================================================
    # Checkpoint config (paths set in _pretrain_common)
    # =========================================================================
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_1p7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 1.7B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-1.7B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-1.7B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8  # --num-workers for dataloader

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"  # default in mcore's transformer_config.py

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"  # default in mcore's transformer_config.py
    cfg.model.cuda_graph_scope = "full"  # default in mcore's transformer_config.py
    cfg.model.cuda_graph_warmup_steps = 3  # default in mcore's transformer_config.py

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None  # None means auto selection
    cfg.model.cross_entropy_loss_fusion = True  # Defaults to True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading)
    # =========================================================================
    cfg.model.recompute_granularity = None  # Not enabled for 1.7B, defaults to None in mcore's transformer_config.py
    cfg.model.recompute_modules = None  # Defaults to None in mcore's transformer_config.py
    cfg.model.fine_grained_activation_offloading = False  # Defaults to False in mcore's transformer_config.py
    cfg.model.offload_modules = None  # Defaults to None in mcore's transformer_config.py

    # =========================================================================
    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # =========================================================================
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False  # default in mcore's OptimizerConfig
    cfg.optimizer.main_grads_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.main_params_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_dtype = torch.float32  # default in mcore's OptimizerConfig
    cfg.optimizer.exp_avg_sq_dtype = torch.float32  # default in mcore's OptimizerConfig

    # =========================================================================
    # Checkpoint config (paths set in _pretrain_common)
    # =========================================================================
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 4B.

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-4B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-4B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading)
    # =========================================================================
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 8B.

    Recommended parallelism: TP=4, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-8B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading)
    # =========================================================================
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_14b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 14B.

    Recommended parallelism: TP=8, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-14B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-14B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading)
    # =========================================================================
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False # Default is False in DistributedInitConfig
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen3_32b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 32B.

    Recommended parallelism: TP=8, PP=2 with recompute enabled for memory optimization.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-32B").to_megatron_provider(load_weights=False)

    # =========================================================================
    # Tokenizer (--tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-32B"

    # =========================================================================
    # Dataset config - mock data by default
    # =========================================================================
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8

    # =========================================================================
    # Model config (tensor_model_parallel_size, pipeline_model_parallel_size, etc.)
    # =========================================================================
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16  # Required for PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # =========================================================================
    # Training config
    # =========================================================================
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # =========================================================================
    # TE (Transformer Engine)
    # =========================================================================
    cfg.model.transformer_impl = "transformer_engine"

    # =========================================================================
    # CUDA Graph
    # =========================================================================
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # =========================================================================
    # Kernel selections
    # =========================================================================
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"

    # =========================================================================
    # Memory saving (recompute & offloading) - ENABLED for 32B
    # =========================================================================
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # Optimizer precision settings
    # =========================================================================
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # =========================================================================
    # DDP config
    # =========================================================================
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg
