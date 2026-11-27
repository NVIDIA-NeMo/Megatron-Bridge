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
"""Two-rank optimizer resume using real Bridge and MLM checkpoint state dictionaries.

Run with uv run python -m torch.distributed.run --standalone --nproc_per_node=2
-m pytest tests/unit_tests/training/test_optimizer_lr_resume_distributed.py.
"""

import os
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, ParamKey
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.training import checkpointing as mlm_checkpointing

from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training import checkpointing
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfigOverrideProvider,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.checkpoint_utils import get_checkpoint_train_state_filename


pytestmark = [pytest.mark.unit, pytest.mark.gpu]


class TwoWeights(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(128, 128, dtype=torch.bfloat16, device="cuda"))
        self.b = torch.nn.Parameter(torch.full_like(self.a, 2.0))
        self.config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True)

    def sharded_state_dict(self, **kwargs):
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, torch.distributed.get_rank()))
            for name, param in self.named_parameters()
        }


class Rates(OptimizerConfigOverrideProvider):
    def __init__(self, rates) -> None:
        self.rates = rates

    def build_config_overrides(self, context):
        return {
            ParamKey(name=(name, "*." + name)): {"max_lr": rate, "min_lr": rate / 10}
            for name, rate in zip(("a", "b"), self.rates)
        }


def _build(rates):
    raw = TwoWeights()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(raw.config, ddp_config, raw)
    optimizer_config = OptimizerConfig(
        lr=0.001,
        min_lr=0.0001,
        weight_decay=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        clip_grad=0.0,
    )
    schedule = SchedulerConfig(override_opt_param_scheduler=True, use_checkpoint_opt_param_scheduler=False)
    schedule.lr_warmup_steps = 2
    schedule.lr_decay_steps = 20
    schedule.wd_incr_steps = 20
    schedule.start_weight_decay = schedule.end_weight_decay = 0.0
    optimizer, scheduler = setup_optimizer(
        optimizer_config, schedule, [model], optimizer_config_override_provider=Rates(rates)
    )
    while isinstance(optimizer, ChainedOptimizer):
        assert len(optimizer.chained_optimizers) == 1
        optimizer = optimizer.chained_optimizers[0]
    cfg = ConfigContainer(
        train=TrainingConfig(),
        model=GPTModelProvider(num_layers=1, hidden_size=128, num_attention_heads=1, seq_length=8, bf16=True),
        optimizer=optimizer_config,
        scheduler=schedule,
        ddp=ddp_config,
        dataset=GPTDatasetConfig(seq_length=8, random_seed=1234),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(
            save_rng=False, load_rng=False, fully_parallel_save=False, fully_parallel_load=False
        ),
    )
    return raw, model, optimizer, scheduler, cfg


def _step(model, optimizer, scheduler):
    for index, param in enumerate(model.parameters()):
        param.main_grad.fill_(0.25 + index * 0.5)
    assert optimizer.step()[0]
    scheduler.step(increment=1)


def _mlm_state(raw, optimizer, scheduler, cfg, metadata, *, loading=False):
    # The generator reads the active checkpoint policy and vocabulary from globals;
    # the tensor generation and checkpoint layout use its actual implementation.
    run = SimpleNamespace(checkpoint=cfg.checkpoint, tokenizer=SimpleNamespace(padded_vocab_size=128))
    with patch.object(mlm_checkpointing, "get_run_config", return_value=run):
        return mlm_checkpointing.generate_state_dict(
            SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
            [raw],
            optimizer,
            scheduler,
            None,
            iteration=3,
            optim_sd_kwargs={"metadata": metadata, "is_loading": loading},
        )


@pytest.fixture(scope="module")
def distributed_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available():
        pytest.skip("requires a two-GPU torch.distributed launch")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
    parallel_state.initialize_model_parallel()
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.fixture
def distributed_checkpoint_dir(distributed_group, tmp_path):
    shared = [str(tmp_path) if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(shared, src=0)
    directory = Path(shared[0]) / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    torch.distributed.barrier()
    return directory


@pytest.mark.parametrize("writer", ["bridge", "mlm"])
@pytest.mark.parametrize("format_name", ["dp_reshardable", "fully_reshardable"])
@pytest.mark.parametrize(
    "old_rates,new_rates",
    [((0.001, 0.002), (0.003, 0.001)), ((0.001, 0.002), (0.004, 0.004)), ((0.001, 0.001), (0.002, 0.003))],
)
def test_disk_checkpoint_resume_keeps_state_and_next_updates(
    distributed_checkpoint_dir, writer, format_name, old_rates, new_rates
):
    directory = distributed_checkpoint_dir
    raw, model, source, source_scheduler, source_cfg = _build(old_rates)
    for _ in range(3):
        _step(model, source, source_scheduler)
    metadata = {"distrib_optim_sharding_type": format_name, "checkpoint_version": 3.1}
    if writer == "bridge":
        saved = checkpointing.generate_state_dict(
            source_cfg.checkpoint,
            [raw],
            source,
            source_scheduler,
            None,
            iteration=3,
            optim_sd_kwargs={"metadata": metadata},
        )
    else:
        saved = _mlm_state(raw, source, source_scheduler, source_cfg, metadata)
    dist_checkpointing.save(saved, str(directory), content_metadata=metadata)
    metadata_error = [None]
    if torch.distributed.get_rank() == 0:
        try:
            source_cfg.to_yaml(str(directory / "run_config.yaml"))
            torch.save(
                TrainState(step=3, consumed_train_samples=3).state_dict(),
                get_checkpoint_train_state_filename(str(directory)),
            )
        except Exception as error:
            metadata_error[0] = repr(error)
    torch.distributed.broadcast_object_list(metadata_error, src=0)
    assert metadata_error[0] is None, metadata_error[0]

    resumed_raw, resumed_model, resumed, scheduler, cfg = _build(new_rates)
    state = SimpleNamespace(
        cfg=cfg, train_state=TrainState(), wandb_logger=None, mlflow_logger=None, comet_logger=None
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    with ExitStack() as stack:
        # Training-global bookkeeping and external telemetry are outside this component test.
        stack.enter_context(patch.object(checkpointing, "update_num_microbatches"))
        stack.enter_context(patch.object(checkpointing.fault_tolerance, "on_checkpoint_loaded"))
        for logger in (checkpointing.wandb_utils, checkpointing.mlflow_utils, checkpointing.comet_utils):
            stack.enter_context(patch.object(logger, "on_load_checkpoint_success"))
        iteration, _ = checkpointing._load_checkpoint_from_path(
            str(directory), state, [resumed_model], resumed, scheduler, pg_collection=groups
        )
    assert iteration == 3
    assert scheduler.num_steps == state.train_state.consumed_train_samples == 3
    assert sorted({group["max_lr"] for group in resumed.param_groups}) == sorted(set(new_rates))
    assert all(not group["default_config"] for group in resumed.param_groups)
    # The new schedule must already be applied before the first resumed update.
    for group in resumed.param_groups:
        assert group["lr"] == scheduler.get_lr(group)
    for name, param in raw.named_parameters():
        if param in source.model_param_group_index_map:
            expected = source._get_main_param_and_optimizer_states(param)
            actual = resumed._get_main_param_and_optimizer_states(getattr(resumed_raw, name))
            for key in ("param", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    # An MLM-generated request can read the same payload, including Bridge's writer.
    ref_raw, ref_model, reference, ref_scheduler, ref_cfg = _build(new_rates)
    loaded = dist_checkpointing.load(
        _mlm_state(ref_raw, reference, ref_scheduler, ref_cfg, metadata, loading=True), str(directory)
    )
    ref_raw.load_state_dict(loaded["model"])
    reference.load_state_dict(loaded["optimizer"], override_lr=True)
    ref_scheduler.load_state_dict(loaded["opt_param_scheduler"])
    for _ in range(2):
        _step(resumed_model, resumed, scheduler)
        _step(ref_model, reference, ref_scheduler)
        for name, param in ref_raw.named_parameters():
            torch.testing.assert_close(getattr(resumed_raw, name), param, rtol=0, atol=0)
            if param in reference.model_param_group_index_map:
                expected = reference._get_main_param_and_optimizer_states(param)
                actual = resumed._get_main_param_and_optimizer_states(getattr(resumed_raw, name))
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
