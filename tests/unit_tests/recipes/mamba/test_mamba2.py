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

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from megatron.bridge.models.mamba import (
    MambaModelProvider1P3B,
    MambaModelProvider2P7B,
    MambaModelProvider130M,
    MambaModelProvider370M,
    MambaModelProvider780M,
    NVIDIAMambaHybridProvider8B,
    NVIDIAMambaModelProvider8B,
)
from megatron.bridge.recipes.mamba import (
    mamba2_1p3b_pretrain_config,
    mamba2_2p7b_pretrain_config,
    mamba2_8b_pretrain_config,
    mamba2_130m_pretrain_config,
    mamba2_370m_pretrain_config,
    mamba2_780m_pretrain_config,
    mamba2_hybrid_8b_pretrain_config,
)
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


RECIPE_CASES = [
    ("mamba2_130m", mamba2_130m_pretrain_config, MambaModelProvider130M, False),
    ("mamba2_370m", mamba2_370m_pretrain_config, MambaModelProvider370M, False),
    ("mamba2_780m", mamba2_780m_pretrain_config, MambaModelProvider780M, False),
    ("mamba2_1p3b", mamba2_1p3b_pretrain_config, MambaModelProvider1P3B, False),
    ("mamba2_2p7b", mamba2_2p7b_pretrain_config, MambaModelProvider2P7B, False),
    ("mamba2_8b", mamba2_8b_pretrain_config, NVIDIAMambaModelProvider8B, True),
    ("mamba2_hybrid_8b", mamba2_hybrid_8b_pretrain_config, NVIDIAMambaHybridProvider8B, True),
]

SMALL_PAR_COMBOS = [
    (1, 1, 1),
    (1, 4, 2),
    (1, 2, 4),
    (1, 2, 2),
    (1, 4, 1),
]

LARGE_PAR_COMBOS = [
    (1, 1, 1),
    (2, 1, 4),
    (4, 2, 2),
    (8, 2, 2),
]


class TestMamba2Recipes:
    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_default_parameters(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()

        assert isinstance(cfg, ConfigContainer)
        assert isinstance(cfg.model, provider)

        # Training configuration
        assert cfg.train.train_iters == 1_168_251
        assert cfg.train.global_batch_size == 8
        assert cfg.train.micro_batch_size == 1
        assert cfg.train.eval_interval == 100
        assert cfg.train.eval_iters == 32

        # Optimizer
        assert cfg.optimizer.optimizer == "adam"
        assert cfg.optimizer.lr == 3e-4
        assert cfg.optimizer.min_lr == 3e-5
        assert cfg.optimizer.weight_decay == 0.1
        assert cfg.optimizer.bf16 is True
        assert cfg.optimizer.fp16 is False

        # Dataset in mock mode by default
        assert cfg.dataset.sequence_length == 4096
        assert cfg.dataset.split == "1,1,1"
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None

        # Tokenizer
        if uses_null_tokenizer:
            assert cfg.tokenizer.tokenizer_type == "NullTokenizer"
            assert cfg.tokenizer.vocab_size == DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
        else:
            assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
            assert cfg.tokenizer.tokenizer_model == "EleutherAI/gpt-neox-20b"

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_custom_training_parameters(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func(
            train_iters=10000,
            global_batch_size=256,
            micro_batch_size=2,
            seq_length=4096,
            lr=1e-4,
            min_lr=1e-5,
            lr_warmup_iters=1000,
        )

        assert cfg.train.train_iters == 10000
        assert cfg.train.global_batch_size == 256
        assert cfg.train.micro_batch_size == 2
        assert cfg.dataset.sequence_length == 4096
        assert cfg.optimizer.lr == 1e-4
        assert cfg.optimizer.min_lr == 1e-5
        assert cfg.scheduler.lr_warmup_iters == 1000
        assert cfg.scheduler.lr_decay_iters is None

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_with_custom_directory(self, name, config_func, provider, uses_null_tokenizer):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = config_func(dir=temp_dir, name=f"{name}_run")

            expected_run_dir = os.path.join(temp_dir, f"{name}_run")
            expected_checkpoint_dir = os.path.join(expected_run_dir, "checkpoints")
            expected_tensorboard_dir = os.path.join(expected_run_dir, "tb_logs")

            assert cfg.checkpoint.save == expected_checkpoint_dir
            assert cfg.logger.tensorboard_dir == expected_tensorboard_dir

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_mock_toggle(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func(mock=True)
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None
        assert cfg.dataset.split == "1,1,1"

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_with_data_paths(self, name, config_func, provider, uses_null_tokenizer):
        data_paths = ["/path/to/data1", "/path/to/data2", "/path/to/data3"]
        cfg = config_func(data_paths=data_paths)
        assert cfg.dataset.split == "9999,8,2"
        assert cfg.dataset.blend is not None
        assert cfg.dataset.blend_per_split is None

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_with_train_valid_test_paths(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func(
            train_data_path=["/path/to/train1", "/path/to/train2", "/path/to/train3"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2", "/path/to/valid3"],
            test_data_path=["/path/to/test1", "/path/to/test2", "/path/to/test3"],
        )
        assert cfg.dataset.split is None
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_prioritizes_blend(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func(
            train_data_path=["/path/to/train1", "/path/to/train2"],
            valid_data_path=["/path/to/valid1", "/path/to/valid2"],
            test_data_path=["/path/to/test1", "/path/to/test2"],
            data_paths=["/path/to/data1", "/path/to/data2"],
        )
        assert cfg.dataset.split == "9999,8,2"
        assert cfg.dataset.blend is not None
        assert cfg.dataset.blend_per_split is None

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_checkpoint_configuration(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()
        assert cfg.checkpoint.save_interval == 2000
        assert cfg.checkpoint.ckpt_format == "torch_dist"
        assert cfg.checkpoint.fully_parallel_load is True

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_ddp_configuration(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()
        assert cfg.ddp.check_for_nan_in_grad is True
        assert cfg.ddp.grad_reduce_in_fp32 is True
        assert cfg.ddp.overlap_grad_reduce is True
        assert cfg.ddp.overlap_param_gather is True
        assert cfg.ddp.use_distributed_optimizer is True

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_comm_overlap(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()
        assert cfg.comm_overlap is None

        custom_overlap = CommOverlapConfig(
            tp_comm_overlap=True,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            data_parallel_size=1,
        )
        cfg2 = config_func(comm_overlap_config=custom_overlap)
        assert cfg2.comm_overlap is not None
        assert cfg2.comm_overlap.tp_comm_overlap is True

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_scheduler_configuration(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func(train_iters=50000)
        assert cfg.scheduler.start_weight_decay == 0.033
        assert cfg.scheduler.end_weight_decay == 0.033
        assert cfg.scheduler.weight_decay_incr_style == "constant"
        assert cfg.scheduler.lr_decay_style == "cosine"
        assert cfg.scheduler.lr_warmup_iters == 2000
        assert cfg.scheduler.lr_warmup_init == 0.0
        assert cfg.scheduler.lr_decay_iters is None
        assert cfg.scheduler.override_opt_param_scheduler is True

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_rng_and_dataset(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()
        assert cfg.rng.seed == 1234
        assert cfg.dataset.random_seed == 1234
        assert cfg.dataset.reset_attention_mask is False
        assert cfg.dataset.reset_position_ids is False
        assert cfg.dataset.eod_mask_loss is False
        assert cfg.dataset.num_dataset_builder_threads == 1
        assert cfg.dataset.data_sharding is True
        assert cfg.dataset.dataloader_type == "single"
        assert cfg.dataset.num_workers == 8

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_logger(self, name, config_func, provider, uses_null_tokenizer):
        cfg = config_func()
        assert cfg.logger.log_interval == 10
        assert "tb_logs" in cfg.logger.tensorboard_dir

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    def test_pretrain_config_parallelism_combinations(self, name, config_func, provider, uses_null_tokenizer):
        is_large = name in {"mamba2_8b", "mamba2_hybrid_8b"}
        combos = LARGE_PAR_COMBOS if is_large else SMALL_PAR_COMBOS
        for tp, pp, cp in combos:
            cfg = config_func(
                tensor_model_parallel_size=tp,
                pipeline_model_parallel_size=pp,
                context_parallel_size=cp,
                pipeline_dtype=torch.bfloat16,
                sequence_parallel=(is_large and tp > 1),
            )
            assert cfg.model.tensor_model_parallel_size == tp
            assert cfg.model.pipeline_model_parallel_size == pp
            assert cfg.model.context_parallel_size == cp

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    @pytest.mark.parametrize("global_batch_size,micro_batch_size", [(8, 1), (16, 2), (32, 4), (64, 8)])
    def test_pretrain_config_batch_sizes(
        self, name, config_func, provider, uses_null_tokenizer, global_batch_size, micro_batch_size
    ):
        cfg = config_func(global_batch_size=global_batch_size, micro_batch_size=micro_batch_size)
        assert cfg.train.global_batch_size == global_batch_size
        assert cfg.train.micro_batch_size == micro_batch_size

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    @pytest.mark.parametrize("seq_length", [1024, 2048, 4096, 8192, 16384])
    def test_pretrain_config_sequence_lengths(self, name, config_func, provider, uses_null_tokenizer, seq_length):
        cfg = config_func(seq_length=seq_length)
        assert cfg.dataset.sequence_length == seq_length

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_precision_recipes(self, name, config_func, provider, uses_null_tokenizer, precision):
        cfg = config_func(mixed_precision=precision) if False else config_func(precision_config=precision)
        assert cfg.mixed_precision == precision

    @pytest.mark.unit
    @pytest.mark.parametrize("name,config_func,provider,uses_null_tokenizer", RECIPE_CASES)
    @patch("megatron.bridge.recipes.utils.dataset_utils.get_blend_and_blend_per_split")
    def test_pretrain_config_fallback_to_mock_when_no_weights(
        self, mock_get_blend, name, config_func, provider, uses_null_tokenizer
    ):
        mock_get_blend.return_value = (None, None)
        cfg = config_func(data_paths=["/some/path"])
        assert cfg.dataset.blend is None
        assert cfg.dataset.blend_per_split is None
        assert cfg.dataset.split == "1,1,1"
