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

"""Bounded GB200 training recipes for Muse Spark Open.

These are early recipes: they are functional but not yet performance-optimized. The pretraining and full-SFT
recipes fit the model with full activation recompute.

Each recipe is self-contained: it sets every model, data, and training option it relies on, even where that
repeats another recipe.
"""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import (
    _peft_common,
    _peft_common_vlm,
    _pretrain_common,
    _sft_common,
    _sft_common_vlm,
)
from megatron.bridge.recipes.utils.dataset_utils import (
    default_coderforge_config,
    default_peft_config,
    default_tulu3_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


_MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B"
_CORD_V2_REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"  # pragma: allowlist secret
_TULU3_REVISION = "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be"  # pragma: allowlist secret
_CODERFORGE_REVISION = "060fca96cf723b2ebab3181e9e59fafd273df3cb"  # pragma: allowlist secret


def muse_spark_open_pretrain_192gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step bounded text pretraining config on 192 GB200 GPUs.

    TP1/PP6/EP32 gives every GPU a unique slice of the experts. The data is
    text-only, so the perception towers stay frozen rather than receiving only
    weight decay. Pass the imported checkpoint with ``--pretrained_checkpoint``
    to start from natural expert routing; the recipe leaves it unset.

    The peak learning rate is 3e-5, decaying to 3e-6, instead of the bounded
    pretraining contract's from-scratch 3e-4: at 3e-4 the warm-started model's loss
    rises until the learning rate decays, while at 3e-5 it falls.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Decoder blocks per stage; the first and last stages also hold the embedding with the perception
    # towers and the output layer.
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (9, 11, 11, 11, 11, 9))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_shared_expert_overlap = False

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-5,
        min_lr=3e-6,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def muse_spark_open_pretrain_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step bounded text pretraining config on 256 GB200 GPUs with eight pipeline stages.

    TP1/PP8/EP32 keeps each stage's expert all-to-all within 32 GPUs and gives every
    GPU a unique slice of the experts. The data-parallel size is 32, as in the 192-GPU
    recipe, so both see the same batches. The data is text-only, so the perception
    towers stay frozen. Pass the imported checkpoint with ``--pretrained_checkpoint``.

    The peak learning rate is 3e-5, decaying to 3e-6, instead of the bounded
    pretraining contract's from-scratch 3e-4: at 3e-4 the warm-started model's loss
    rises until the learning rate decays, while at 3e-5 it falls.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Decoder blocks per stage; the first and last stages also hold the embedding with the perception
    # towers and the output layer.
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (7, 8, 8, 8, 8, 8, 8, 7))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8
    cfg.dataset.random_seed = 1234

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 1234
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=40,
        max_lr=3e-5,
        min_lr=3e-6,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_spark_open_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a one-epoch full SFT config on 100K packed Tulu 3 conversations with 192 GB200 GPUs.

    The first 100,000 Tulu 3 conversations pack into 10,951 4K sequences; 85 steps
    at GBS 128 make one epoch. TP4/PP6/EP32 leaves eight data-parallel replicas,
    each running 16 microbatches per step. Pipeline stages hold 9, 11, 11, 11, 11,
    and 9 decoder blocks; the first and last also hold the embedding with the
    perception towers and the output layer. Full activation recompute keeps every
    stage in memory.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (9, 11, 11, 11, 11, 9))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    # Text-only data never exercises the perception towers.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    # HybridEP: each EP32 group is one pipeline stage's 32 contiguous ranks (8 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_shared_expert_overlap = False
    # Packed sequences give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4.
    cfg.dataset = default_tulu3_config(seq_length=4096, enable_offline_packing=True, pad_seq_to_mult=4)
    cfg.dataset.hf_dataset.split = "train[:100000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/muse-spark-open/tulu3-100k-full-sft-gb200-4k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    # One epoch: caps the 10,951 packs at 85 full global batches, leaving 71 packs unused.
    cfg.dataset.max_train_samples = 85 * 128
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 4096 - 4

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 85
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 85
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 85
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def muse_spark_open_sft_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Return a 100-step full SFT config on packed Tulu 3 data with 256 GB200 GPUs and eight pipeline stages.

    Data is Tulu 3 train[:10000] packed to 8K at GBS 8, the cross-model bounded SFT
    contract; the 192-GPU recipe instead trains one epoch of 100K conversations at 4K
    and GBS 128. TP4/PP8/EP32 leaves eight data-parallel replicas, one packed sequence
    each. The first and last stages hold seven decoder blocks because they also own
    the embedding with the perception towers and the output layer; the middle stages
    hold eight. With one microbatch per replica the extra stages lower memory but do
    not shorten the step.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 8192
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (7, 8, 8, 8, 8, 8, 8, 7))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    # Text-only data never exercises the perception towers.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4. The prepared data is
    # shared with the 192-GPU recipe.
    cfg.dataset = default_tulu3_config(seq_length=8192, enable_offline_packing=True, pad_seq_to_mult=4)
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/muse-spark-open/tulu3-full-sft-gb200-8k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 8192 - 4

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_spark_open_sft_192gpu_gb200_bf16_long_context_config() -> ConfigContainer:
    """Return a 20-step packed 32K Tulu 3 full SFT config with context parallelism on 192 GB200 GPUs.

    TP4/PP6/CP2/EP32 with sequence parallelism leaves four data-parallel replicas.
    Context parallelism uses A2A because Transformer Engine rejects P2P for the
    sliding-window layers. Packing aligns each sequence to lcm(2*CP, TP*CP) = 8.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 32768
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (9, 11, 11, 11, 11, 9))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    # Text-only data never exercises the perception towers.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.dataset = default_tulu3_config(seq_length=32768, enable_offline_packing=True, pad_seq_to_mult=8)
    cfg.dataset.hf_dataset.split = "train[:10000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/muse-spark-open/tulu3-full-sft-gb200-32k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    cfg.dataset.max_train_samples = 10000
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 32768 - 8

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2,
        max_lr=1e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 20
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 20
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config() -> ConfigContainer:
    """Return a 20-step packed 64K CoderForge agent-trajectory full SFT config with CP4 on 192 GB200 GPUs.

    SWE-Rebench trajectories are long single sequences (median about 46K tokens), so
    they exercise the 2,048-token sliding window and long-range full attention under
    context parallelism, unlike packed short chat rows. TP4/PP6/CP4/EP32 with sequence
    parallelism keeps 4K tokens per GPU and leaves two data-parallel replicas. Packing
    aligns each sequence to lcm(2*CP, TP*CP) = 16. Only the SWE-Rebench files are
    downloaded; the dataset metadata lists every split, so split completeness is not
    verified.
    """
    cfg = _sft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 65536
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (9, 11, 11, 11, 11, 9))
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 4
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    # Text-only data never exercises the perception towers.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.dataset = default_coderforge_config(seq_length=65536, enable_offline_packing=True, pad_seq_to_mult=16)
    # 20 steps at GBS 8 consume 160 packs; slicing the split bounds offline tokenization.
    cfg.dataset.hf_dataset.split = "SWE_Rebench[:600]"
    cfg.dataset.hf_dataset.load_kwargs = {
        "revision": _CODERFORGE_REVISION,
        "data_files": {"SWE_Rebench": "trajectories/SWE_Rebench-*.parquet"},
        "verification_mode": "no_checks",
    }
    cfg.dataset.hf_output_root = "work/data/muse-spark-open/coderforge-full-sft-gb200-64k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.seed = 1234
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 65536 - 16

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 20
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=2,
        max_lr=1e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 20
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 20
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_spark_open_peft_64gpu_gb200_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a one-epoch LoRA config on 100K packed Tulu 3 conversations with 64 GB200 GPUs.

    LoRA rank 32 / alpha 32 adapts the attention projections, every routed expert
    (one adapter per expert), and the latent MoE projections. The frozen base runs at
    TP4/PP4/EP16 with stages of 15, 16, 16, and 15 decoder blocks, leaving four
    data-parallel replicas that each run 32 microbatches of GBS 128 packed 4K
    sequences; 85 steps make one epoch of the 10,951 packs from the first 100,000
    Tulu 3 conversations. The expert adapters keep the routed expert inputs for their
    weight gradients, which needs the four pipeline stages. DoRA has no per-expert
    adapter controls and keeps attention-only adapters.
    """
    cfg = _peft_common()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.tokenizer.tokenizer_model = _MODEL_ID
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Decoder blocks per stage; the first and last stages also hold the embedding with the perception
    # towers and the output layer.
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (15, 16, 16, 15))
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    # Text-only data never exercises the perception towers.
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]
    # HybridEP: each EP16 group is one pipeline stage's 16 contiguous ranks (4 nodes) in one NVL72 domain.
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_shared_expert_overlap = False
    # Packed sequences give EP ranks different token counts; HybridEP needs equal ones, so pad to the group max.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = True

    # Sequence parallelism over TP4 aligns each packed sequence to a multiple of 4.
    cfg.dataset = default_tulu3_config(seq_length=4096, enable_offline_packing=True, pad_seq_to_mult=4)
    cfg.dataset.hf_dataset.split = "train[:100000]"
    cfg.dataset.hf_dataset.load_kwargs = {"revision": _TULU3_REVISION}
    cfg.dataset.hf_output_root = "work/data/muse-spark-open/tulu3-100k-peft-gb200-4k"
    cfg.dataset.hf_rewrite = False
    cfg.dataset.hf_validation_proportion = None
    # One epoch: caps the 10,951 packs at 85 full global batches, leaving 71 packs unused.
    cfg.dataset.max_train_samples = 85 * 128
    cfg.dataset.seed = 1234
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.offline_packing_specs.max_single_sequence_length = 4096 - 4

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() == "lora":
        peft_cfg.target_modules = [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "fc1_latent_proj",
            "fc2_latent_proj",
        ]
        peft_cfg.dim = 32
        peft_cfg.alpha = 32
        peft_cfg.dropout = 0.0
        peft_cfg.share_expert_adapters = False
    elif isinstance(peft_scheme, str) and peft_scheme.lower() == "dora":
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 85
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 85
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.checkpoint.save_interval = 85
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return cfg


def muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a 100-step attention-projection LoRA config on image-text CORD-v2 data with 16 GB200 GPUs.

    The frozen base runs at TP4/EP16 with sequence parallelism. Context parallelism
    uses A2A because Transformer Engine rejects P2P for sliding-window layers.
    """
    cfg = _peft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_audio_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {"max_image_tokens": 256}
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in {"lora", "dora"}:
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.dropout = 0.0
    cfg.peft = peft_cfg

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=1e-4,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config() -> ConfigContainer:
    """Return a 100-step full SFT config on image-text CORD-v2 data with 192 GB200 GPUs.

    TP1/PP6/EP32 gives every GPU a unique slice of the experts. The first and last
    stages hold nine decoder blocks each because they also own the embedding with
    the perception towers and the output layer; the middle stages hold eleven. Full
    activation recompute and the distributed optimizer hold the complete parameter,
    gradient, and optimizer state. The audio tower stays frozen because the
    image-text dataset never exercises it.
    """
    cfg = _sft_common_vlm()

    cfg.model = AutoBridge.from_hf_pretrained(_MODEL_ID).get_model_config()
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.cp_comm_type = "a2a"
    cfg.model.sequence_parallel = False
    cfg.model.hybrid_layer_pattern = "|".join("*E" * blocks for blocks in (9, 11, 11, 11, 11, 9))
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_audio_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _MODEL_ID
    cfg.dataset.hf_processor_kwargs = {"max_image_tokens": 256}
    cfg.dataset.source.split = "train"
    cfg.dataset.source.load_kwargs = {"revision": _CORD_V2_REVISION}
    cfg.dataset.do_validation = False
    cfg.dataset.do_test = False
    cfg.dataset.num_workers = 4
    cfg.dataset.dataloader_type = "cyclic"
    cfg.dataset.pad_to_max_length = True
    cfg.dataset.enable_in_batch_packing = False

    cfg.mixed_precision = bf16_mixed()
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True
    cfg.rng.seed = 5678
    cfg.train.train_iters = 100
    # PP6 leaves 32 data-parallel replicas; two samples each per optimizer step.
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 10
    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        max_lr=5e-6,
        min_lr=0.0,
    )
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.load = None
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config",
    "muse_spark_open_peft_64gpu_gb200_bf16_config",
    "muse_spark_open_pretrain_192gpu_gb200_bf16_config",
    "muse_spark_open_pretrain_256gpu_gb200_bf16_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_long_context_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config",
    "muse_spark_open_sft_256gpu_gb200_bf16_config",
]
