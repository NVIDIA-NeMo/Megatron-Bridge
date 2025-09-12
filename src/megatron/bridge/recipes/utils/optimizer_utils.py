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


def distributed_fused_adam_with_cosine_annealing(
    precision: str = "bf16-mixed",
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int = 2000,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.95,
    adam_eps: float = 1e-5,
    weight_decay: float = 0.1,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
) -> tuple[OptimizerConfig, SchedulerConfig]:
    """
    Creates a distributed fused Adam optimizer with cosine annealing scheduler.
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
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=0.0,
        lr_decay_iters=lr_decay_iters,
        override_opt_param_scheduler=True,
    )

    return optimizer, scheduler
