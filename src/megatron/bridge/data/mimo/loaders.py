# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data loader utilities for MIMO training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from megatron.bridge.data.mimo.dp_utils import get_mimo_dp_info
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.bridge.utils.common_utils import print_rank_0


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.state import TrainState


def build_mimo_data_loaders(
    cfg: "ConfigContainer",
    train_state: "TrainState",
    mimo_provider: DatasetProvider,
    train_samples: int,
    valid_samples: int,
    test_samples: int,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Build MIMO data loaders using per-module data-parallel (DP) sampling derived from the MIMO parallelism configuration.
    
    Parameters:
        cfg: Configuration container whose `model` must be a `MimoModelProvider` with a populated `_grids` attribute and a non-`None` `mimo_parallelism_config`.
        mimo_provider: MIMO dataset provider that exposes `build_datasets(context)` and `get_collate_fn()`, and provides loader settings (`num_workers`, `pin_memory`, `drop_last`).
        train_samples (int): Number of training samples to request when building datasets.
        valid_samples (int): Number of validation samples to request when building datasets.
        test_samples (int): Number of test samples to request when building datasets.
    
    Returns:
        Tuple of `(train_loader, valid_loader, test_loader)`. Each element is a `torch.utils.data.DataLoader` configured with DP-aware `DistributedSampler`, or `None` when no dataset was built or when the current rank does not require data (in which case all three are `None`).
    
    Raises:
        ValueError: If `cfg.model` is not a `MimoModelProvider`, if `cfg.model.mimo_parallelism_config` is `None`, or if `cfg.model._grids` is `None` (indicating model infra has not been built).
    """
    from megatron.bridge.models.mimo.mimo_provider import MimoModelProvider

    if not isinstance(cfg.model, MimoModelProvider):
        raise ValueError("cfg.model must be MimoModelProvider for MIMO data loading.")

    if cfg.model.mimo_parallelism_config is None:
        raise ValueError("mimo_parallelism_config must be set for MIMO data loading.")

    if cfg.model._grids is None:
        raise ValueError(
            "MimoModelProvider._grids is None. Ensure build_model() is called before building data loaders."
        )

    print_rank_0("> building MIMO train, validation, and test datasets ...")

    # Use cached grids from build_model()
    grids = cfg.model._grids

    dp_rank, dp_size, needs_data, loader_module = get_mimo_dp_info(cfg.model.mimo_parallelism_config, grids)

    if not needs_data:
        return None, None, None

    # Build datasets
    context = DatasetBuildContext(
        train_samples=train_samples,
        valid_samples=valid_samples,
        test_samples=test_samples,
        tokenizer=None,
    )
    train_ds, valid_ds, test_ds = mimo_provider.build_datasets(context)

    print_rank_0(
        f"  Built datasets: train={len(train_ds) if train_ds else 0}, "
        f"valid={len(valid_ds) if valid_ds else 0}, "
        f"test={len(test_ds) if test_ds else 0}"
    )

    # Build data loaders with DP-aware sampling
    collate_fn = mimo_provider.get_collate_fn()
    micro_batch_size = cfg.train.micro_batch_size

    def _make_loader(dataset, shuffle: bool = True) -> Optional[DataLoader]:
        """
        Create a DataLoader for the given dataset configured with a DistributedSampler for the current MIMO data-parallel replica.
        
        Parameters:
            dataset: The dataset to load; if `None`, no loader is created and `None` is returned.
            shuffle (bool): Whether the distributed sampler should shuffle sample order for this split.
        
        Returns:
            DataLoader or `None`: A DataLoader using the configured `DistributedSampler` and provider settings, or `None` when `dataset` is `None`.
        """
        if dataset is None:
            return None
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=micro_batch_size,
            sampler=sampler,
            num_workers=mimo_provider.num_workers,
            collate_fn=collate_fn,
            pin_memory=mimo_provider.pin_memory,
            drop_last=mimo_provider.drop_last,
        )

    train_loader = _make_loader(train_ds, shuffle=True)
    valid_loader = _make_loader(valid_ds, shuffle=False)
    test_loader = _make_loader(test_ds, shuffle=False)

    return train_loader, valid_loader, test_loader
