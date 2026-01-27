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

from typing import Optional

from megatron.bridge.training.config import OptimizerConfig, SchedulerConfig


def distributed_muon_with_cosine_annealing(
    precision: str = "bf16-mixed",
    muon_momentum: float = 0.95,
    muon_use_nesterov: bool = True,
    muon_scale_mode: str = "spectral",
    muon_fp32_matmul_prec: str = "medium",
    muon_num_ns_steps: int = 5,
    muon_tp_mode: str = "blockwise",
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int = 2000,
    weight_decay: float = 0.1,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
) -> tuple[OptimizerConfig, SchedulerConfig]:
    """
    Creates a distributed Muon optimizer configuration paired with a cosine annealing learning rate scheduler.

    Args:
        precision (str): Type of mixed precision to use ("bf16-mixed", "16-mixed", etc.).
        muon_momentum (float): Momentum value for the Muon optimizer.
        muon_use_nesterov (bool): Whether to use Nesterov momentum in Muon optimizer.
        muon_scale_mode (str): Mode to use for scaling within the Muon optimizer ("spectral", etc.).
        muon_fp32_matmul_prec (str): Matmul precision for Muon optimizer ("medium", etc.).
        muon_num_ns_steps (int): Number of no-step steps for Muon optimizer.
        muon_tp_mode (str): Tensor parallel mode for Muon optimizer ("blockwise", etc.).
        lr_warmup_iters (int): Number of warmup iterations for the learning rate scheduler.
        lr_decay_iters (int): Number of decay iterations for the learning rate scheduler.
        weight_decay (float): Amount of weight decay to apply.
        max_lr (float): Maximum learning rate.
        min_lr (Optional[float]): Minimum learning rate; if None, determined automatically.
        clip_grad (float): Gradient clipping threshold.

    Returns:
        tuple[OptimizerConfig, SchedulerConfig]: Configuration objects for the optimizer and scheduler.
    """

    scheduler = SchedulerConfig(
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=0.0,
        lr_decay_iters=lr_decay_iters,
        override_opt_param_scheduler=True,
    )
    optimizer = OptimizerConfig(
        optimizer="dist_muon",
        lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        muon_momentum=muon_momentum,
        muon_use_nesterov=muon_use_nesterov,
        muon_scale_mode=muon_scale_mode,
        muon_fp32_matmul_prec=muon_fp32_matmul_prec,
        muon_num_ns_steps=muon_num_ns_steps,
        muon_tp_mode=muon_tp_mode,
        clip_grad=clip_grad,
    )
    return optimizer, scheduler


def distributed_fused_adam_with_cosine_annealing(
    precision: str = "bf16-mixed",
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-5,
    weight_decay: float = 0.1,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
    start_weight_decay: float = 0.033,
    end_weight_decay: float = 0.033,
    weight_decay_incr_style: str = "constant",
    lr_decay_style: str = "cosine",
) -> tuple[OptimizerConfig, SchedulerConfig]:
    """
    Creates a distributed fused Adam optimizer with cosine annealing scheduler.

    Args:
        precision: Mixed precision type ("bf16-mixed", "16-mixed", etc.)
        lr_warmup_iters: Number of iterations for learning rate warmup
        lr_decay_iters: Number of iterations for learning rate decay. If None,
            defaults to train_iters during training.
        adam_beta1: Adam optimizer beta1 parameter
        adam_beta2: Adam optimizer beta2 parameter
        adam_eps: Adam optimizer epsilon parameter
        weight_decay: Weight decay coefficient
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate (defaults to 0.1 * max_lr)
        clip_grad: Gradient clipping value
        start_weight_decay: Initial weight decay value for scheduler
        end_weight_decay: Final weight decay value for scheduler
        weight_decay_incr_style: Weight decay increment style ("constant", "linear", etc.)
        lr_decay_style: Learning rate decay style ("cosine", "linear", etc.)

    Returns:
        Tuple of (OptimizerConfig, SchedulerConfig)
    """
    min_lr = min_lr if min_lr is not None else (0.1 * max_lr)
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
    )

    scheduler = SchedulerConfig(
        start_weight_decay=start_weight_decay,
        end_weight_decay=end_weight_decay,
        weight_decay_incr_style=weight_decay_incr_style,
        lr_decay_style=lr_decay_style,
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=0.0,
        lr_decay_iters=lr_decay_iters,
        override_opt_param_scheduler=True,
    )

    return optimizer, scheduler


def distributed_fused_adam_with_cosine_annealing_samples(
    precision: str = "bf16-mixed",
    lr_warmup_samples: Optional[int] = None,
    lr_decay_samples: Optional[int] = None,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-5,
    weight_decay: float = 0.1,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
) -> tuple[OptimizerConfig, SchedulerConfig]:
    """
    Creates a distributed fused Adam optimizer with cosine annealing scheduler for sample-based training.

    This is the sample-based equivalent of distributed_fused_adam_with_cosine_annealing().

    Args:
        precision: Mixed precision mode ("bf16-mixed", "16-mixed", etc.)
        lr_warmup_samples: Number of samples for learning rate warmup (None = auto from train_samples)
        lr_decay_samples: Number of samples for learning rate decay (None = auto from train_samples)
        adam_beta1: Adam optimizer beta1 parameter
        adam_beta2: Adam optimizer beta2 parameter
        adam_eps: Adam optimizer epsilon parameter
        weight_decay: Weight decay coefficient
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate (defaults to 0.1 * max_lr)
        clip_grad: Gradient clipping value

    Returns:
        A tuple of (OptimizerConfig, SchedulerConfig) configured for sample-based training
    """
    min_lr = min_lr if min_lr is not None else (0.1 * max_lr)
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_eps=adam_eps,
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
    )

    scheduler = SchedulerConfig(
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_samples=lr_warmup_samples,
        lr_warmup_init=0.0,
        lr_decay_samples=lr_decay_samples,
        override_opt_param_scheduler=True,
    )

    return optimizer, scheduler
