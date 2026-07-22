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

import json
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import torch
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import RerunDataIterator
from torch.utils.data import DataLoader

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.training.config import ConfigContainer, GPTDatasetConfig
from megatron.bridge.training.state import TrainState
from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler
from megatron.bridge.utils.common_utils import print_rank_0


def get_blend_and_blend_per_split(
    data_paths: Optional[list[str]] = None,
    data_args_path: Optional[str] = None,
    per_split_data_args_path: Optional[str] = None,
    train_data_paths: Optional[list[str]] = None,
    valid_data_paths: Optional[list[str]] = None,
    test_data_paths: Optional[list[str]] = None,
) -> tuple[Optional[list[str]], Optional[list[list[str]]]]:
    """Determine dataset blends from command-line arguments or config files.

    Parses different ways dataset paths/weights can be specified (single list,
    per-split lists, config files) and returns the blend information.

    Args:
        data_paths: List of paths/weights for a single blended dataset.
        data_args_path: Path to a file containing data paths/weights for a single blend.
        per_split_data_args_path: Path to a JSON file containing train/valid/test splits,
                                  each with its own list of paths/weights.
        train_data_paths: List of paths/weights specifically for the training split.
        valid_data_paths: List of paths/weights specifically for the validation split.
        test_data_paths: List of paths/weights specifically for the test split.

    Returns:
        A tuple (blend, blend_per_split):
        - blend: A list representing a single data blend, or None.
        - blend_per_split: A list containing blends for train, valid, test splits, or None.
                         Only one of `blend` or `blend_per_split` will be non-None.
    """
    use_data_path = data_paths is not None or data_args_path is not None
    use_per_split_data_path = (
        any(elt is not None for elt in [train_data_paths, valid_data_paths, test_data_paths])
        or per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if data_args_path is not None:
            assert data_paths is None
            with open(data_args_path, "r") as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert data_paths is not None
            blend = get_blend_from_list(data_paths)
    elif use_per_split_data_path:
        if per_split_data_args_path is not None:
            with open(per_split_data_args_path, "r") as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(train_data_paths),
                get_blend_from_list(valid_data_paths),
                get_blend_from_list(test_data_paths),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def cyclic_iter(iter: Iterable) -> Iterator:
    """Create an infinite iterator from a finite iterable."""
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples(cfg: ConfigContainer) -> tuple[int, int, int]:
    """Calculate the number of samples for train, validation, and test sets.

    Determines sample counts based on training mode either specified iterations or samples,
    global batch size, and evaluation interval/iterations specified in the config.

    Args:
        cfg: The main configuration container.

    Returns:
        A tuple (train_samples, valid_samples, test_samples).
    """

    # If train_samples is directly provided, use it
    if cfg.train.train_samples is not None:
        train_samples = cfg.train.train_samples
    else:
        # Otherwise fallback to calculating samples based on iterations and global batch size
        train_samples = cfg.train.train_iters * cfg.train.global_batch_size

    if cfg.validation.eval_interval:
        eval_iters = (cfg.train.train_iters // cfg.validation.eval_interval + 1) * cfg.validation.eval_iters
    else:
        eval_iters = 0
    test_iters = cfg.validation.eval_iters

    eval_gbs = (
        cfg.validation.eval_global_batch_size
        if cfg.validation.eval_global_batch_size is not None
        else cfg.train.global_batch_size
    )
    return (
        train_samples,
        eval_iters * eval_gbs,
        test_iters * eval_gbs,
    )


def build_train_valid_test_datasets(
    cfg: ConfigContainer, build_train_valid_test_datasets_provider: Callable
) -> tuple[Any, Any, Any]:
    """Build train, validation, and test datasets using a provider function.

    Args:
        cfg: The main configuration container.
        build_train_valid_test_datasets_provider: A function that takes
            train_val_test_num_samples and dataset_config and returns the datasets.

    Returns:
        A tuple (train_dataset, valid_dataset, test_dataset).
    """
    train_valid_test_num_samples = get_train_valid_test_num_samples(cfg)
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples, cfg.dataset)


def build_train_valid_test_datasets_for_num_epochs(
    cfg: ConfigContainer, build_train_valid_test_datasets_provider: Callable
) -> tuple[Any, Any, Any]:
    """Build a finite GPT SFT dataset and resolve epoch-based training iterations.

    This cannot use :func:`build_train_valid_test_datasets` because that function
    requires ``train_iters`` to already be resolved. GPT SFT dataset builders
    determine dataset sizes from the data source or ``max_train_samples`` and ignore
    the requested target sample counts, so zero placeholders are sufficient here.
    """
    if not isinstance(cfg.dataset, GPTSFTDatasetConfig):
        raise ValueError(
            "num_epochs is only supported for finite GPTSFTDatasetConfig datasets because other dataset "
            "providers may build a requested number of samples instead of exposing their true dataset size."
        )
    if cfg.dataset.dataloader_type != "batch":
        raise ValueError('num_epochs is currently supported only with dataloader_type="batch"')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider([0, 0, 0], cfg.dataset)
    if train_ds is None:
        raise ValueError("num_epochs requires a training dataset")

    try:
        train_dataset_size = len(train_ds)
    except (TypeError, NotImplementedError) as error:
        raise ValueError("num_epochs requires a training dataset with a finite length") from error

    cfg._resolve_num_epochs(train_dataset_size)
    return train_ds, valid_ds, test_ds


def build_train_valid_test_data_loaders(
    cfg: ConfigContainer,
    train_state: TrainState,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
    *,
    eval_dp_group: torch.distributed.ProcessGroup | None = None,
) -> tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build train, validation, and test data loaders.

    First builds the datasets using the provided provider function, then constructs
    PyTorch DataLoaders with appropriate sampling and configuration.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.
        dp_group: Data-parallel group used to shard the training dataset.
        eval_dp_group: Optional data-parallel group used to shard validation and test datasets.
            Defaults to ``dp_group``.

    Returns:
        A tuple (train_dataloader, valid_dataloader, test_dataloader).
    """
    # Check for MegatronMIMO path
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

    if cfg.train.num_epochs is not None and cfg.dataset.dataloader_type != "batch":
        raise ValueError('num_epochs is currently supported only with dataloader_type="batch"')

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Construct the data pipeline
    # Build datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg, build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider
    )

    drop_last = False if cfg.train.num_epochs is not None else cfg.dataset.drop_last
    if (
        train_ds is not None
        and cfg.dataset.dataloader_type == "batch"
        and not drop_last
        and len(train_ds) % cfg.train.global_batch_size != 0
        and not isinstance(cfg.dataset, GPTSFTDatasetConfig)
    ):
        raise ValueError(
            'dataloader_type="batch" with drop_last=False requires GPTSFTDatasetConfig because incomplete '
            "global batches use negative indices that only GPT SFT datasets convert to loss-masked padding. "
            "Use drop_last=True for other dataset providers."
        )

    # Check that the train dataset has at least one global batch of samples.
    if (
        train_ds is not None
        and cfg.dataset.dataloader_type != "external"
        and drop_last
        and len(train_ds) < cfg.train.global_batch_size
    ):
        raise RuntimeError(
            f"Not enough train samples for a single global batch: "
            f"train dataset size ({len(train_ds)}) < global batch size ({cfg.train.global_batch_size})."
        )

    exit_signal = cfg.train.exit_signal

    def worker_init_fn(_):
        DistributedSignalHandler(exit_signal).__enter__()

    maybe_worker_init_fn = worker_init_fn if cfg.train.exit_signal_handler_for_dataloader else None

    # Resolve train and eval DP ownership from their respective process groups.
    dp_rank = torch.distributed.get_rank(group=dp_group)
    dp_size = torch.distributed.get_world_size(group=dp_group)
    if eval_dp_group is None:
        eval_dp_group = dp_group
    eval_dp_rank = torch.distributed.get_rank(group=eval_dp_group)
    eval_dp_size = torch.distributed.get_world_size(group=eval_dp_group)
    # Text SFT configs call this field ``seed`` while Megatron GPT configs call
    # it ``random_seed``. Fall back to the unoffset config RNG seed so batch
    # sampling never depends on the pipeline-rank-specific torch global seed.
    sampler_seed = getattr(cfg.dataset, "seed", None)
    if sampler_seed is None:
        sampler_seed = getattr(cfg.dataset, "random_seed", None)
    if sampler_seed is None:
        sampler_seed = getattr(getattr(cfg, "rng", None), "seed", None)

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        train_ds,
        train_state.consumed_train_samples,
        cfg.dataset.dataloader_type,
        cfg.train.micro_batch_size,
        cfg.dataset.num_workers,
        cfg.dataset.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        global_batch_size=cfg.train.global_batch_size,
        drop_last=drop_last,
        seed=sampler_seed,
    )
    eval_gbs = (
        cfg.validation.eval_global_batch_size
        if cfg.validation.eval_global_batch_size is not None
        else cfg.train.global_batch_size
    )
    eval_mbs = (
        cfg.validation.eval_micro_batch_size
        if cfg.validation.eval_micro_batch_size is not None
        else cfg.train.micro_batch_size
    )
    if cfg.validation.skip_train and cfg.validation.eval_iters > 0:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            0,
            cfg.dataset.dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=eval_dp_rank,
            data_parallel_size=eval_dp_size,
            global_batch_size=eval_gbs,
            seed=sampler_seed,
        )
    elif cfg.validation.eval_iters > 0:
        val_dataloader_type = "cyclic" if isinstance(cfg.dataset, GPTDatasetConfig) else cfg.dataset.dataloader_type
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            train_state.consumed_valid_samples,
            val_dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=eval_dp_rank,
            data_parallel_size=eval_dp_size,
            global_batch_size=eval_gbs,
            seed=sampler_seed,
        )

    if cfg.validation.eval_iters > 0:
        test_dataloader = build_pretraining_data_loader(
            test_ds,
            0,
            cfg.dataset.dataloader_type,
            eval_mbs,
            cfg.dataset.num_workers,
            cfg.dataset.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
            collate_fn=test_ds.collate_fn if hasattr(test_ds, "collate_fn") else None,
            pin_memory=cfg.dataset.pin_memory,
            persistent_workers=cfg.dataset.persistent_workers,
            data_parallel_rank=eval_dp_rank,
            data_parallel_size=eval_dp_size,
            global_batch_size=eval_gbs,
            seed=sampler_seed,
        )

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and cfg.train.train_iters > 0
    do_valid = valid_dataloader is not None and cfg.validation.eval_iters > 0
    do_test = test_dataloader is not None and cfg.validation.eval_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    train_state.do_train = flags[0].item()
    train_state.do_valid = flags[1].item()
    train_state.do_test = flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
    cfg: ConfigContainer,
    train_state: TrainState,
    build_train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
    *,
    eval_dp_group: torch.distributed.ProcessGroup | None = None,
) -> tuple[Optional[RerunDataIterator], Optional[RerunDataIterator], Optional[RerunDataIterator]]:
    """Build train, validation, and test data iterators.

    Builds the data loaders first, then wraps them in appropriate iterators
    (e.g., RerunDataIterator, cyclic_iter) based on the configuration.

    Args:
        cfg: The main configuration container.
        train_state: The current training state.
        build_train_valid_test_datasets_provider: A function to build the datasets.
        dp_group: Data-parallel group used to shard the training dataset.
        eval_dp_group: Optional data-parallel group used to shard validation and test datasets.

    Returns:
        A tuple (train_data_iterator, valid_data_iterator, test_data_iterator).
    """

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
        dp_group=dp_group,
        eval_dp_group=eval_dp_group,
    )

    # Build iterators.
    dl_type = cfg.dataset.dataloader_type
    assert dl_type in ["single", "cyclic", "batch", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            # Single-pass iteration (no cycling)
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type in ("cyclic", "batch"):
            # Cycle for finetuning: allows train_iters > dataset size without raising StopIteration
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        val_dataloader_type = "cyclic" if isinstance(cfg.dataset, GPTDatasetConfig) else cfg.dataset.dataloader_type
        valid_data_iterator = _get_iterator(val_dataloader_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def setup_data_iterators(
    cfg: ConfigContainer,
    train_state: TrainState,
    model_length: int,
    train_valid_test_datasets_provider: Callable,
    dp_group: torch.distributed.ProcessGroup,
    *,
    eval_dp_group: torch.distributed.ProcessGroup | None = None,
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
        dp_group: Data-parallel group used to shard the training dataset.
        eval_dp_group: Optional data-parallel group used to shard validation and test datasets.

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
        eval_dp_group=eval_dp_group,
    )

    return train_data_iterator, valid_data_iterator, test_data_iterator
