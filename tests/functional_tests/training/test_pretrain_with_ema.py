# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import os

import pytest
import torch

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.ema import EMACallback
from megatron.bridge.training.ema_checkpoint import EMA_DIRNAME
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_name
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


def _build_config(
    checkpoint_dir: str,
    tensorboard_dir: str,
    total_iters: int,
    save_interval: int,
    load_dir: str | None = None,
) -> ConfigContainer:
    seq_length = 512

    model_cfg = Llama32ModelProvider1B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        make_vocab_size_divisible_by=128,
        vocab_size=None,
        num_layers=1,
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=total_iters,
            global_batch_size=8,
            micro_batch_size=1,
            exit_signal_handler=True,
        ),
        validation=ValidationConfig(
            eval_interval=5,
            eval_iters=2,
        ),
        optimizer=OptimizerConfig(
            optimizer="adam",
            bf16=True,
            fp16=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            lr=3e-3,
            weight_decay=0.01,
            min_lr=1e-6,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.033,
            end_weight_decay=0.033,
            weight_decay_incr_style="constant",
            lr_decay_style="cosine",
            lr_warmup_iters=2,
            lr_warmup_init=0.0,
            lr_decay_iters=total_iters,
            override_opt_param_scheduler=True,
        ),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        logger=LoggerConfig(
            log_interval=5,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=10000,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=load_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
    )


def _ema_sidecar_path(checkpoint_dir: str, iteration: int, rank: int) -> str:
    checkpoint_name = get_checkpoint_name(checkpoint_dir, iteration, release=False)
    return os.path.join(
        checkpoint_name,
        EMA_DIRNAME,
        f"rank_{rank:05d}.pt",
    )


class TestPretrainWithEMA:
    @pytest.mark.run_only_on("GPU")
    def test_pretrain_with_ema_checkpoint(self, tmp_path):
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            total_iters = 6

            cfg = _build_config(
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                total_iters=total_iters,
                save_interval=total_iters,
            )

            callbacks = [
                EMACallback(
                    decay=0.95,
                    start_step=0,
                    store_on_cpu=True,
                    log_interval=5,
                )
            ]

            pretrain(cfg, forward_step, callbacks=callbacks)

            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, total_iters)

            rank = torch.distributed.get_rank()
            ema_path = _ema_sidecar_path(checkpoint_dir, total_iters, rank)

            assert os.path.exists(ema_path), f"EMA sidecar not found: {ema_path}"

            payload = torch.load(ema_path, map_location="cpu", weights_only=False)

            assert "ema_state" in payload
            assert "ema_updates" in payload
            assert "ema_skipped_iters" in payload
            assert isinstance(payload["ema_state"], dict)
            assert len(payload["ema_state"]) > 0
            assert payload["ema_updates"] > 0

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_pretrain_resume_with_ema(self, tmp_path):
        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            first_total_iters = 4
            resumed_total_iters = 8

            callbacks_first = [
                EMACallback(
                    decay=0.95,
                    start_step=0,
                    store_on_cpu=True,
                    log_interval=5,
                )
            ]

            cfg_first = _build_config(
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                total_iters=first_total_iters,
                save_interval=first_total_iters,
            )

            pretrain(cfg_first, forward_step, callbacks=callbacks_first)

            torch.distributed.barrier()

            rank = torch.distributed.get_rank()
            first_ema_path = _ema_sidecar_path(checkpoint_dir, first_total_iters, rank)
            assert os.path.exists(first_ema_path)

            first_payload = torch.load(first_ema_path, map_location="cpu", weights_only=False)
            first_updates = first_payload["ema_updates"]

            callbacks_resume = [
                EMACallback(
                    decay=0.95,
                    start_step=0,
                    store_on_cpu=True,
                    log_interval=5,
                )
            ]

            cfg_resume = _build_config(
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                total_iters=resumed_total_iters,
                save_interval=resumed_total_iters,
                load_dir=checkpoint_dir,
            )

            pretrain(cfg_resume, forward_step, callbacks=callbacks_resume)

            torch.distributed.barrier()
            verify_checkpoint_files(checkpoint_dir, resumed_total_iters)

            resumed_ema_path = _ema_sidecar_path(checkpoint_dir, resumed_total_iters, rank)
            assert os.path.exists(resumed_ema_path)

            resumed_payload = torch.load(resumed_ema_path, map_location="cpu", weights_only=False)

            assert resumed_payload["ema_updates"] >= first_updates
            assert len(resumed_payload["ema_state"]) > 0

        finally:
            clear_directories(tmp_path)