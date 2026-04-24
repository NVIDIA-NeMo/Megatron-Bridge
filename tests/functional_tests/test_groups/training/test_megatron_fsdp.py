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
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
    runtime_config_update,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import _finish_train
from megatron.bridge.training.train import train as run_training
from tests.functional_tests.utils import (
    broadcast_path,
    clear_directories,
    initialize_distributed,
    verify_checkpoint_files,
)


@dataclass
class Llama3ModelProviderFSDP145M(GPTModelProvider):
    """Small Llama3 model configuration for FSDP testing."""

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    num_query_groups: int = 8
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-05
    rotary_percent: float = 1.0
    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 2
    hidden_size: int = 768
    ffn_hidden_size: int = 2688
    num_attention_heads: int = 16
    vocab_size: int | None = None
    gradient_accumulation_fusion: bool = False


def create_fsdp_model_config(seq_length: int, bf16: bool = True, **kwargs) -> Llama3ModelProviderFSDP145M:
    """Create a standardized FSDP model configuration."""
    base_config = {
        "seq_length": seq_length,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "attention_softmax_in_fp32": True,
        "make_vocab_size_divisible_by": 128,
        "vocab_size": None,
    }
    if bf16:
        base_config.update(
            {
                "bf16": True,
                "pipeline_dtype": torch.bfloat16,
            }
        )
    base_config.update(kwargs)
    return Llama3ModelProviderFSDP145M(**base_config)


def create_base_training_config(
    train_iters: int, global_batch_size: int = 8, micro_batch_size: int = 1, **kwargs
) -> TrainingConfig:
    """Create a standardized training configuration."""
    base_config = {
        "train_iters": train_iters,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "exit_signal_handler": True,
    }
    base_config.update(kwargs)
    return TrainingConfig(**base_config)


def create_base_validation_config(train_iters: int, **kwargs) -> ValidationConfig:
    """Create a standardized validation configuration."""
    base_config = {
        "eval_interval": train_iters + 1,  # Disable evaluation to avoid hanging
        "eval_iters": 0,  # No evaluation iterations
    }
    base_config.update(kwargs)
    return ValidationConfig(**base_config)


def create_base_optimizer_config(**kwargs) -> OptimizerConfig:
    """Create a standardized optimizer configuration."""
    base_config = {
        "optimizer": "adam",
        "bf16": True,
        "fp16": False,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_eps": 1e-5,
        "use_distributed_optimizer": True,
        "clip_grad": 1.0,
        "lr": 3e-3,
        "weight_decay": 0.01,
        "min_lr": 1e-6,
    }
    base_config.update(kwargs)
    return OptimizerConfig(**base_config)


def create_base_scheduler_config(total_iters: int, **kwargs) -> SchedulerConfig:
    """Create a standardized scheduler configuration."""
    base_config = {
        "start_weight_decay": 0.033,
        "end_weight_decay": 0.033,
        "weight_decay_incr_style": "constant",
        "lr_decay_style": "cosine",
        "lr_warmup_iters": 2,
        "lr_warmup_init": 0.0,
        "lr_decay_iters": total_iters,
        "override_opt_param_scheduler": True,
    }
    base_config.update(kwargs)
    return SchedulerConfig(**base_config)


def create_base_ddp_config(**kwargs) -> DistributedDataParallelConfig:
    """Create a standardized DDP configuration for FSDP."""
    base_config = {
        "check_for_nan_in_grad": True,
        "grad_reduce_in_fp32": True,
        "overlap_grad_reduce": True,
        "overlap_param_gather": True,
        "average_in_collective": False,  # Required for FSDP
        "data_parallel_sharding_strategy": "optim_grads_params",  # For Megatron FSDP only
        "use_distributed_optimizer": True,
        "use_megatron_fsdp": True,  # Enable FSDP in DDP config too
    }
    base_config.update(kwargs)
    return DistributedDataParallelConfig(**base_config)


def create_base_dataset_config(seq_length: int, **kwargs) -> MockGPTDatasetConfig:
    """Create a standardized dataset configuration."""
    base_config = {
        "random_seed": 1234,
        "reset_attention_mask": False,
        "reset_position_ids": False,
        "eod_mask_loss": False,
        "seq_length": seq_length,
        "num_dataset_builder_threads": 1,
        "data_sharding": True,
        "dataloader_type": "single",
        "num_workers": 1,
    }
    base_config.update(kwargs)
    return MockGPTDatasetConfig(**base_config)


def create_base_logger_config(tensorboard_dir: Optional[str] = None, log_interval: int = 5, **kwargs) -> LoggerConfig:
    """Create a standardized logger configuration."""
    base_config = {
        "log_interval": log_interval,
        "log_params_norm": True,
    }
    if tensorboard_dir:
        base_config["tensorboard_dir"] = tensorboard_dir
    base_config.update(kwargs)
    return LoggerConfig(**base_config)


def create_base_tokenizer_config(**kwargs) -> TokenizerConfig:
    """Create a standardized tokenizer configuration."""
    base_config = {
        "tokenizer_type": "NullTokenizer",
        "vocab_size": 10000,
    }
    base_config.update(kwargs)
    return TokenizerConfig(**base_config)


def create_base_checkpoint_config(
    checkpoint_dir: Optional[str] = None, load_dir: Optional[str] = None, save_interval: Optional[int] = None, **kwargs
) -> CheckpointConfig:
    """Create a standardized checkpoint configuration."""
    base_config = {
        "ckpt_format": "fsdp_dtensor",  # Use FSDP DTensor format
        "fully_parallel_save": True,
        "async_save": False,  # Disable async save for testing
    }
    if checkpoint_dir:
        base_config["save"] = checkpoint_dir
    if load_dir:
        base_config["load"] = load_dir
    if save_interval:
        base_config["save_interval"] = save_interval
    base_config.update(kwargs)
    return CheckpointConfig(**base_config)


def create_fsdp_config_container(
    seq_length: int,
    train_iters: int,
    checkpoint_dir: Optional[str] = None,
    load_dir: Optional[str] = None,
    save_interval: Optional[int] = None,
    tensorboard_dir: Optional[str] = None,
    **overrides,
) -> ConfigContainer:
    """Create a complete FSDP configuration container with common defaults."""
    return ConfigContainer(
        model=create_fsdp_model_config(seq_length, **overrides.pop("model", {})),
        dist=DistributedInitConfig(use_megatron_fsdp=True),
        train=create_base_training_config(train_iters, **overrides.pop("train", {})),
        validation=create_base_validation_config(train_iters, **overrides.pop("validation", {})),
        optimizer=create_base_optimizer_config(**overrides.pop("optimizer", {})),
        scheduler=create_base_scheduler_config(train_iters, **overrides.pop("scheduler", {})),
        ddp=create_base_ddp_config(**overrides.pop("ddp", {})),
        dataset=create_base_dataset_config(seq_length, **overrides.pop("dataset", {})),
        logger=create_base_logger_config(tensorboard_dir, **overrides.pop("logger", {})),
        tokenizer=create_base_tokenizer_config(**overrides.pop("tokenizer", {})),
        checkpoint=create_base_checkpoint_config(
            checkpoint_dir, load_dir, save_interval, **overrides.pop("checkpoint", {})
        ),
        rng=RNGConfig(seed=1234, **overrides.pop("rng", {})),
    )


def _compute_forward_only_loss(forward_step_func, model, data_iterator, state, pg_collection):
    """Run one forward-only microbatch pass and return the reduced loss dict.

    Used to verify checkpoint correctness by comparing the loss from the same
    model state and data position before and after a save/load cycle.
    """
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.pipeline_parallel.utils import is_pp_last_stage
    from megatron.core.utils import get_model_config

    from megatron.bridge.training.utils.train_utils import prepare_forward_step_func

    model_config = get_model_config(model[0])
    wrapped_fwd = prepare_forward_step_func(forward_step_func, state)

    forward_backward_func = get_forward_backward_func(
        pp_size=pg_collection.pp.size(),
        vp_size=state.cfg.model.virtual_pipeline_model_parallel_size,
    )
    p2p_communicator = P2PCommunicator(pp_group=pg_collection.pp, config=model_config)

    for m in model:
        m.eval()

    with torch.no_grad():
        losses_reduced = forward_backward_func(
            forward_step_func=wrapped_fwd,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=state.cfg.model.seq_length,
            micro_batch_size=state.cfg.train.micro_batch_size,
            decoder_seq_length=state.cfg.model.seq_length,
            forward_only=True,
            p2p_communicator=p2p_communicator,
            pg_collection=pg_collection,
        )

    if is_pp_last_stage(pg_collection.pp):
        loss_dict = {}
        for key in losses_reduced[0]:
            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                val = torch.vstack(val).sum(dim=0)
                torch.distributed.all_reduce(val, group=pg_collection.dp_cp)
                loss_dict[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                loss_dict[key] = torch.cat(val).mean()
        return loss_dict
    return {}


def _get_inner_optimizer(megatron_optimizer):
    """Unwrap ChainedOptimizer / DistributedOptimizer to reach the raw torch optimizer."""
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    opt = megatron_optimizer
    if isinstance(opt, ChainedOptimizer):
        opt = opt.chained_optimizers[0]
    while hasattr(opt, "optimizer") and not isinstance(opt.optimizer, type(opt)):
        inner = opt.optimizer
        if inner is opt:
            break
        opt = inner
    return opt


class TestMegatronFSDP:
    """
    Test end to end training with Megatron FSDP and fsdp_dtensor checkpoint functionality.
    """

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_pretrain_basic(self, tmp_path):
        """
        Test basic FSDP training without checkpointing.
        """
        initialize_distributed()

        torch.distributed.barrier()

        try:
            seq_length = 512
            total_iters = 10

            cfg = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=total_iters,
            )

            # Run training
            pretrain(cfg, forward_step)

            torch.distributed.barrier()

        finally:
            clear_directories(tmp_path)

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_pretrain_save_resume(self, tmp_path):
        """
        Test FSDP checkpoint correctness by verifying that a model loaded from a
        checkpoint produces the same forward-pass loss as the original model.

        Phase 1: Train for N iterations, save checkpoint, compute a forward-only
                 loss with the trained model still in memory.
        Phase 2: Load the checkpoint into a fresh model, compute a forward-only
                 loss on the same data position.
        Assert the two losses are equal.
        """
        from megatron.bridge.data.utils import get_dataset_provider
        from megatron.bridge.training.setup import setup

        initialize_distributed()
        shared_base_dir = broadcast_path(tmp_path)

        checkpoint_dir = os.path.join(shared_base_dir, "checkpoints")
        tensorboard_dir = os.path.join(shared_base_dir, "tensorboard")

        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(tensorboard_dir, exist_ok=True)

        torch.distributed.barrier()

        try:
            seq_length = 512
            train_iters = 5

            # --- Phase 1: train, save, compute reference loss ----------------
            cfg_train = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=train_iters,
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
                save_interval=train_iters,
            )
            runtime_config_update(cfg_train)

            state = GlobalState()
            state.cfg = cfg_train
            setup_out = setup(state, get_dataset_provider(cfg_train.dataset))

            run_training(
                forward_step,
                setup_out.model,
                setup_out.optimizer,
                setup_out.scheduler,
                setup_out.train_data_iterator,
                setup_out.valid_data_iterator,
                setup_out.state,
                setup_out.checkpoint_manager,
                setup_out.pg_collection,
            )

            loss_before = _compute_forward_only_loss(
                forward_step,
                setup_out.model,
                setup_out.train_data_iterator,
                setup_out.state,
                setup_out.pg_collection,
            )

            _finish_train(setup_out.state, setup_out.checkpoint_manager)

            torch.distributed.barrier()

            verify_checkpoint_files(
                checkpoint_dir,
                train_iters,
                ckpt_format=cfg_train.checkpoint.ckpt_format,
                storage_writers_per_rank=cfg_train.checkpoint.storage_writers_per_rank,
            )

            torch.distributed.barrier()

            # --- Phase 2: load checkpoint, compute loss ----------------------
            cfg_load = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=train_iters,
                load_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir,
            )
            runtime_config_update(cfg_load)

            state2 = GlobalState()
            state2.cfg = cfg_load
            setup_out2 = setup(state2, get_dataset_provider(cfg_load.dataset))

            loss_after = _compute_forward_only_loss(
                forward_step,
                setup_out2.model,
                setup_out2.train_data_iterator,
                setup_out2.state,
                setup_out2.pg_collection,
            )

            _finish_train(setup_out2.state, setup_out2.checkpoint_manager)

            # --- Verify losses match -----------------------------------------
            assert loss_before, "No loss computed before checkpoint save"
            for key in loss_before:
                assert key in loss_after, f"Key '{key}' missing from loaded model loss"
                torch.testing.assert_close(
                    loss_before[key],
                    loss_after[key],
                    msg=f"Loss mismatch for key '{key}'",
                )

        finally:
            clear_directories(shared_base_dir)

    @pytest.mark.run_only_on("GPU")
    def test_fsdp_precision_aware_optimizer_decoupled_grad_consistency(self, tmp_path):
        """Verify that MFSDP + precision-aware optimizer keeps decoupled_grad
        consistent across ParamAndGradBuffer, FusedAdam, and clip_grad_norm.

        1. Build a 1-GPU MFSDP model with use_precision_aware_optimizer=True.
        2. Run a single training iteration (forward + backward + optimizer step).
        3. Assert:
           a) ParamAndGradBuffer.use_decoupled_grad is True
           b) The underlying FusedAdam was created with use_decoupled_grad=True
           c) clip_grad_norm succeeds and returns a finite grad norm
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp import MegatronFSDP

        from megatron.bridge.data.utils import get_dataset_provider
        from megatron.bridge.training.setup import setup

        initialize_distributed()

        torch.distributed.barrier()

        try:
            seq_length = 512
            train_iters = 1

            cfg = create_fsdp_config_container(
                seq_length=seq_length,
                train_iters=train_iters,
                optimizer={"use_precision_aware_optimizer": True},
            )
            runtime_config_update(cfg)

            state = GlobalState()
            state.cfg = cfg
            setup_out = setup(state, get_dataset_provider(cfg.dataset))

            model_list = setup_out.model
            optimizer = setup_out.optimizer

            # --- Run one training iteration ---
            run_training(
                forward_step,
                model_list,
                optimizer,
                setup_out.scheduler,
                setup_out.train_data_iterator,
                setup_out.valid_data_iterator,
                setup_out.state,
                setup_out.checkpoint_manager,
                setup_out.pg_collection,
            )

            # --- (a) MFSDP ParamAndGradBuffer has use_decoupled_grad=True ---
            mfsdp_model = model_list[0].module
            assert isinstance(mfsdp_model, MegatronFSDP), f"Expected MegatronFSDP, got {type(mfsdp_model)}"
            pgb = mfsdp_model.param_and_grad_buffer
            assert pgb.use_decoupled_grad is True, (
                "ParamAndGradBuffer.use_decoupled_grad should be True when use_precision_aware_optimizer is enabled"
            )

            # --- (b) Underlying FusedAdam has use_decoupled_grad=True ---
            inner_opt = _get_inner_optimizer(optimizer)
            assert getattr(inner_opt, "use_decoupled_grad", False), (
                f"FusedAdam.use_decoupled_grad should be True, "
                f"got {getattr(inner_opt, 'use_decoupled_grad', 'MISSING')} "
                f"on {type(inner_opt).__name__}"
            )

            # --- (c) clip_grad_norm does not segfault / has non-zero grad norm ---
            grad_norm = optimizer.clip_grad_norm(cfg.optimizer.clip_grad)
            assert torch.isfinite(torch.tensor(grad_norm)), f"clip_grad_norm returned non-finite grad_norm={grad_norm}"

            _finish_train(setup_out.state, setup_out.checkpoint_manager)

        finally:
            clear_directories(tmp_path)
