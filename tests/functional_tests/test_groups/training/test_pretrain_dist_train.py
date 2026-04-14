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

import pytest
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import MockVLMConversationProvider
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLDistTrainConfig, Qwen3VLModelProvider
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.pretrain import pretrain
from tests.functional_tests.utils import initialize_distributed


class TestPretrainDistTrain:
    """
    Test end to end training with distributed training functionality.
    """

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_dist_train(self):
        """
        Qwen3-VL 8B mock pretrain with DistTrain (vision DP + language DP).
        Requires 8 processes (4 vision + 4 language ranks).
        Does not write checkpoints or TensorBoard/progress files (smoke run only).
        """
        initialize_distributed()
        if torch.distributed.get_world_size() != 8:
            pytest.skip("DistTrain layout expects world_size=8 (vision_world_size=4, language_world_size=4)")

        hf_path = "Qwen/Qwen3-VL-8B-Instruct"
        seq_length = 8192
        total_iters = 5
        global_batch_size = 32
        micro_batch_size = 2

        # Build Qwen3-VL provider from HF (same pattern as _qwen3_vl_common in qwen3_vl.py)
        bridge = AutoBridge.from_hf_pretrained(hf_path)
        model_cfg = bridge.to_megatron_provider(load_weights=False)
        assert isinstance(model_cfg, Qwen3VLModelProvider)

        model_cfg.tensor_model_parallel_size = 1
        model_cfg.pipeline_model_parallel_size = 2
        model_cfg.pipeline_dtype = torch.bfloat16
        model_cfg.virtual_pipeline_model_parallel_size = None
        model_cfg.context_parallel_size = 1
        model_cfg.sequence_parallel = False
        model_cfg.expert_model_parallel_size = 1
        model_cfg.seq_length = seq_length
        model_cfg.num_layers = 12
        model_cfg.deepstack_visual_indexes = [3, 6, 9]
        model_cfg.freeze_language_model = False
        model_cfg.freeze_vision_model = False
        model_cfg.freeze_vision_projection = False
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1
        model_cfg.cuda_graph_impl = "none"
        model_cfg.apply_rope_fusion = False
        model_cfg.variable_seq_lengths = True
        model_cfg.moe_token_dispatcher_type = "flex"
        model_cfg.deallocate_pipeline_outputs = False
        model_cfg.dist_train = Qwen3VLDistTrainConfig(
            use_dist_train=True,
            vision_to_llm_dp_ratio=2,
            vision_world_size=4,
            language_world_size=4,
            vision_tensor_model_parallel_size=1,
            vision_pipeline_model_parallel_size=1,
            vision_context_parallel_size=1,
            vision_expert_tensor_parallel_size=1,
            vision_expert_model_parallel_size=1,
        )

        optimizer_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=2,
            lr_decay_iters=total_iters,
            clip_grad=0.0,
        )

        ddp_cfg = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        )

        # CommOverlapConfig.setup() overwrites ddp overlap flags when use_distributed_optimizer
        # and data_parallel_size > 1 unless explicitly set here (see comm_overlap._get_optimizer_overlap_cfgs).
        comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        )

        dataset_cfg = MockVLMConversationProvider(
            seq_length=seq_length,
            hf_processor_path=hf_path,
            prompt="Describe this image.",
            num_workers=1,
            dataloader_type="cyclic",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            create_attention_mask=True,
            pad_to_max_length=True,
            num_images=8,
            image_size=(768, 768),
        )

        cfg = ConfigContainer(
            model=model_cfg,
            train=TrainingConfig(
                train_iters=total_iters,
                global_batch_size=global_batch_size,
                micro_batch_size=micro_batch_size,
                eval_interval=500,
                eval_iters=0,
                manual_gc=True,
                manual_gc_interval=100,
                manual_gc_eval=100,
                exit_signal_handler=True,
            ),
            validation=ValidationConfig(
                eval_interval=10_000,
                eval_iters=0,
            ),
            optimizer=optimizer_cfg,
            scheduler=scheduler_cfg,
            ddp=ddp_cfg,
            dataset=dataset_cfg,
            logger=LoggerConfig(
                log_interval=1,
                tensorboard_dir=None,
                log_timers_to_tensorboard=False,
                log_memory_to_tensorboard=False,
                log_throughput_to_tensorboard=False,
                log_progress=False,
            ),
            tokenizer=TokenizerConfig(
                tokenizer_type="NullTokenizer",
                vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
            ),
            checkpoint=CheckpointConfig(
                save_interval=0,
                save=None,
                load=None,
                async_save=False,
            ),
            rng=RNGConfig(seed=1234),
            comm_overlap=comm_overlap,
            mixed_precision="bf16_mixed",
            dist=DistributedInitConfig(
                use_decentralized_pg=True,
                use_gloo_process_groups=False,
            ),
        )

        pretrain(cfg, qwen3_vl_forward_step)

