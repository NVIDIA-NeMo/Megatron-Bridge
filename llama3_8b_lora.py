#!/usr/bin/env python3
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
from typing import Any, Optional

import datasets
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.hub.data.builders.hf_dataset import HFDatasetConfig, ProcessExampleOutput
from megatron.hub.models.llama import Llama3ModelProvider8B
from megatron.hub.peft.lora import LoRA
from megatron.hub.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.hub.training.finetune import finetune
from megatron.hub.training.gpt_step import forward_step


def squad_process_example_fn(example: dict[str, Any], tokenizer: Optional[Any] = None) -> ProcessExampleOutput:
    """Format SQuAD examples into instruction-following format."""
    result = {}
    result["input"] = "Context: " + example["context"] + " Question: " + example["question"] + " Answer:"
    result["output"] = example["answers"]["text"][0]
    return result


def main():
    """
    This script finetunes a llama3 8b model on the SQuAD dataset.
    """

    # Hyperparameters
    max_steps = 1000
    seq_length = 2048
    global_batch_size = 128
    micro_batch_size = 1
    lr = 1e-4

    model_cfg = Llama3ModelProvider8B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=seq_length,
        bf16=True,
        params_dtype=torch.bfloat16,
        cross_entropy_loss_fusion=False,
    )

    squad_dataset = datasets.load_dataset("squad")

    pretrained_checkpoint = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B"

    dataset_config = HFDatasetConfig(
        dataset_name="squad",
        process_example_fn=squad_process_example_fn,
        dataset_dict=squad_dataset,
        seq_length=seq_length,
        seed=1234,
        dataloader_type="cyclic",
        num_workers=8,
        do_validation=False,
        do_test=False,
        val_proportion=None,
        delete_raw=True,
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=os.path.join(pretrained_checkpoint, "hf_assets"),
    )

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        use_distributed_optimizer=False,
        clip_grad=1.0,
        lr=lr,
        weight_decay=0.0,
        min_lr=0.0,
    )

    scheduler_config = SchedulerConfig(
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_decay_iters=max_steps,
        lr_warmup_iters=50,
        lr_warmup_init=0.0,
        override_opt_param_scheduler=True,
    )

    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=False,
    )

    checkpoint_dir = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/checkpoints/megatron_hub_peft"
    wandb_save_dir = "/nemo_run/wandb"
    tensorboard_dir = "/nemo_run/tensorboard"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(wandb_save_dir, exist_ok=True)

    logger_config = LoggerConfig(
        wandb_project="megatron-hub-custom-loop-peft",
        wandb_entity="nvidia",
        wandb_exp_name=f"mhub_squad_llama3_8b_model_provider_lora_gbs_{global_batch_size}_seq_length_{seq_length}_lr_{lr}",
        wandb_save_dir=wandb_save_dir,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        log_validation_ppl_to_tensorboard=True,
        tensorboard_log_interval=1,
        timing_log_level=2,
        log_progress=True,
        log_interval=10,
        logging_level="INFO",
    )

    checkpoint_config = CheckpointConfig(
        save_interval=200,
        save=None,
        load=None,
        async_save=True,
        fully_parallel_save=True,
        pretrained_checkpoint=pretrained_checkpoint,
    )

    rng_config = RNGConfig(seed=1234)
    lora_config = LoRA(dim=8, alpha=16)

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=max_steps,
            eval_interval=200,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            exit_signal_handler=True,
        ),
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        ddp=ddp_config,
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=tokenizer_config,
        checkpoint=checkpoint_config,
        rng=rng_config,
        peft=lora_config,
    )

    finetune(config=cfg, forward_step_func=forward_step)
    torch.distributed.destroy_process_group()


# def import_llama3_8b():
#     from pathlib import Path
#     from megatron.hub.converters.llama import HFLlamaImporter

#     output_dir = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/"
#     model_id = "meta-llama/Meta-Llama-3-8B"
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     importer = HFLlamaImporter(input_path=model_id, output_path=output_dir)
#     result_path = importer.apply()

if __name__ == "__main__":
    main()
