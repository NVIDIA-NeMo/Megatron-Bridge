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

from typing import Callable, Optional, Union

import torch
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.datasets.data_loaders import (
    build_train_valid_test_data_loaders as _mlm_build_train_valid_test_data_loaders,
)
from megatron.training.datasets.data_loaders import (
    build_train_valid_test_datasets,
    cyclic_iter,
    get_blend_and_blend_per_split,
    get_train_valid_test_num_samples,
    wrap_loaders_in_iterators,
)
from torch.utils.data import DataLoader

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.state import TrainState


__all__ = [
    "get_blend_and_blend_per_split",
    "cyclic_iter",
    "get_train_valid_test_num_samples",
    "build_train_valid_test_datasets",
    "build_train_valid_test_data_loaders",
    "build_train_valid_test_data_iterators",
    "setup_data_iterators",
]


def build_train_valid_test_data_loaders(
    cfg: ConfigContainer,
    train_state: TrainState,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build train, validation, and test data loaders.

    Delegates the standard GPT path to the consolidated, globals-free
    implementation in Megatron-LM, passing resumption offsets from ``train_state``.
    The DataLoader ``worker_init_fn`` (exit-signal handler) is now built inside the
    Megatron-LM core from ``cfg.train``, so it is no longer constructed here. The
    MegatronMIMO path stays here because it depends on Bridge-only providers. The
    ``do_*`` flags returned by the core are written back onto ``train_state``.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.
        dp_group: The data-parallel process group.

    Returns:
        A tuple (train_dataloader, valid_dataloader, test_dataloader).
    """
    # Check for MegatronMIMO path (Bridge-only data pipeline).
    from megatron.bridge.data.megatron_mimo.base_provider import MegatronMIMODatasetProvider
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider

    if isinstance(cfg.model, MegatronMIMOProvider):
        if not isinstance(cfg.dataset, MegatronMIMODatasetProvider):
            raise ValueError(
                "MegatronMIMO models require cfg.dataset to be a MegatronMIMODatasetProvider. "
                "Use HFMegatronMIMODatasetProvider, MockMegatronMIMOProvider, or a subclass of MegatronMIMODatasetProvider."
            )
        from megatron.bridge.data.megatron_mimo.loaders import build_megatron_mimo_data_loaders

        train_samples, valid_samples, test_samples = get_train_valid_test_num_samples(cfg)
        train_dataloader, valid_dataloader, test_dataloader = build_megatron_mimo_data_loaders(
            cfg=cfg,
            train_state=train_state,
            megatron_mimo_provider=cfg.dataset,
            train_samples=train_samples,
            valid_samples=valid_samples,
            test_samples=test_samples,
        )

        # Sync train_state flags across all ranks.
        # Use all_reduce(MAX) since some ranks may not have loaders in heterogeneous MegatronMIMO.
        do_train = train_dataloader is not None and cfg.train.train_iters > 0
        do_valid = valid_dataloader is not None and cfg.validation.eval_iters > 0
        do_test = test_dataloader is not None and cfg.validation.eval_iters > 0
        flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        train_state.do_train = flags[0].item()
        train_state.do_valid = flags[1].item()
        train_state.do_test = flags[2].item()

        return train_dataloader, valid_dataloader, test_dataloader

    (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        do_train,
        do_valid,
        do_test,
    ) = _mlm_build_train_valid_test_data_loaders(
        cfg=cfg,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
        dp_group=dp_group,
        consumed_train_samples=train_state.consumed_train_samples,
        consumed_valid_samples=train_state.consumed_valid_samples,
    )

    train_state.do_train = do_train
    train_state.do_valid = do_valid
    train_state.do_test = do_test

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
    cfg: ConfigContainer,
    train_state: TrainState,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
) -> tuple[Optional[RerunDataIterator], Optional[RerunDataIterator], Optional[RerunDataIterator]]:
    """Build train, validation, and test data iterators.

    Builds the data loaders (via :func:`build_train_valid_test_data_loaders`,
    which routes MegatronMIMO vs. the standard MLM path) and wraps them via the
    shared ``wrap_loaders_in_iterators`` helper in Megatron-LM, so the wrapping
    rules are single-sourced and apply uniformly to both loader kinds.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.
        dp_group: The data-parallel process group.

    Returns:
        A tuple (train_data_iterator, valid_data_iterator, test_data_iterator).
    """
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
        dp_group=dp_group,
    )

    return wrap_loaders_in_iterators(cfg, train_dataloader, valid_dataloader, test_dataloader)


def setup_data_iterators(
    cfg: ConfigContainer,
    train_state: TrainState,
    model_length: int,
    train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
) -> tuple[
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
    Union[Optional[RerunDataIterator], list[Optional[RerunDataIterator]]],
]:
    """Set up data iterators, handling virtual pipeline parallelism if enabled.

    Calls `build_train_valid_test_data_iterators` potentially multiple times
    if virtual pipeline parallelism is used, creating separate iterators for each
    virtual stage.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        model_length: The number of model chunks (used for virtual pipeline parallelism).
        train_valid_test_datasets_provider: A function to build the datasets.
        dp_group: The data-parallel process group.

    Returns:
        A tuple (train_data_iterator, valid_data_iterator, test_data_iterator).
        Each element can be a single iterator or a list of iterators if virtual
        pipeline parallelism is enabled.
    """
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        dp_group=dp_group,
    )

    return train_data_iterator, valid_data_iterator, test_data_iterator
