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
    """Build MIMO data loaders with per-module DP settings.

    Creates data loaders with DP-aware sampling based on the MIMO parallelism
    configuration. Only ranks that need data (first/last PP stage) will get
    non-None loaders.

    Args:
        cfg: Configuration container with MimoModelProvider as cfg.model.
        train_state: Current training state.
        mimo_provider: MIMO dataset provider (e.g., MockMimoDatasetProvider)
            with get_collate_fn() method.
        train_samples: Number of training samples.
        valid_samples: Number of validation samples.
        test_samples: Number of test samples.

    Returns:
        Tuple of (train_loader, valid_loader, test_loader).
        Returns (None, None, None) if this rank doesn't need data.

    Raises:
        ValueError: If cfg.model is not MimoModelProvider or mimo_parallelism_config is None.

    Example:
        >>> from megatron.bridge.data.mimo import MockMimoProvider, build_mimo_data_loaders
        >>> provider = MockMimoProvider(
        ...     seq_length=2048,
        ...     processor_paths={"vision": "openai/clip-vit-large-patch14"},
        ...     tokenizer_path="meta-llama/Llama-2-7b-hf",
        ...     special_token_ids={"vision": 32000},
        ...     modality_configs={"vision": {"type": "image", "width": 224, "height": 224}},
        ... )
        >>> train_loader, valid_loader, test_loader = build_mimo_data_loaders(
        ...     cfg, train_state, provider,
        ...     train_samples=10000, valid_samples=1000, test_samples=1000,
        ... )
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

    print_rank_0(
        f"  MIMO DP info: dp_rank={dp_rank}, dp_size={dp_size}, needs_data={needs_data}, loader_module={loader_module}"
    )

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

    return (
        _make_loader(train_ds, shuffle=True),
        _make_loader(valid_ds, shuffle=False),
        _make_loader(test_ds, shuffle=False),
    )
