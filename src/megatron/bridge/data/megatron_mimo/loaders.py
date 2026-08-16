# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Data loader utilities for MegatronMIMO training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from torch.utils.data import DataLoader

from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.data.megatron_mimo.canonical_sampler import build_canonical_mimo_data_loader
from megatron.bridge.data.megatron_mimo.dp_utils import get_megatron_mimo_sampling_info
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.utils.common_utils import print_rank_0


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.state import TrainState


def build_megatron_mimo_data_loaders(
    cfg: "ConfigContainer",
    train_state: "TrainState",
    megatron_mimo_provider: DatasetProvider,
    train_samples: int,
    valid_samples: int,
    test_samples: int,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build MegatronMIMO data loaders with globally consistent sampling.

    By default all data-loading ranks receive identical global micro-batches (the
    sampler uses dp_size=1) and per-module DP sub-sharding is deferred to
    ``slice_batch_for_megatron_mimo`` in the forward step, ensuring consistency with
    the BridgeCommunicator's fan-in/fan-out routing for asymmetric DP configs.
    With ``dataset.megatron_mimo_scalable_dp`` each rank instead reads only its
    canonical-grid shard (see ``canonical_sampler``) and the forward step does not
    slice. Only ranks that need data (first/last PP stage) will get non-None loaders.

    Args:
        cfg: Configuration container with MegatronMIMOProvider as cfg.model.
        train_state: Current training state.
        megatron_mimo_provider: MegatronMIMO dataset provider (e.g., MockMegatronMIMOProvider)
            with get_collate_fn() method.
        train_samples: Number of training samples.
        valid_samples: Number of validation samples.
        test_samples: Number of test samples.

    Returns:
        Tuple of (train_loader, valid_loader, test_loader).
        Returns (None, None, None) if this rank doesn't need data.

    Raises:
        ValueError: If cfg.model is not MegatronMIMOProvider or megatron_mimo_parallelism_config is None.

    Example:
        >>> from megatron.bridge.data.megatron_mimo import MockMegatronMIMOProvider, build_megatron_mimo_data_loaders
        >>> provider = MockMegatronMIMOProvider(
        ...     seq_length=2048,
        ...     processor_paths={"vision": "openai/clip-vit-large-patch14"},
        ...     tokenizer_path="meta-llama/Llama-2-7b-hf",
        ...     special_token_ids={"vision": 32000},
        ...     modality_configs={"vision": {"type": "image", "width": 224, "height": 224}},
        ... )
        >>> train_loader, valid_loader, test_loader = build_megatron_mimo_data_loaders(
        ...     cfg, train_state, provider,
        ...     train_samples=10000, valid_samples=1000, test_samples=1000,
        ... )
    """
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOProvider

    if not isinstance(cfg.model, MegatronMIMOProvider):
        raise ValueError("cfg.model must be MegatronMIMOProvider for MegatronMIMO data loading.")

    if cfg.model.megatron_mimo_parallelism_config is None:
        raise ValueError("megatron_mimo_parallelism_config must be set for MegatronMIMO data loading.")

    if cfg.model._grids is None:
        raise ValueError(
            "MegatronMIMOProvider._grids is None. Ensure build_model() is called before building data loaders."
        )

    # Validate that micro_batch_size is divisible by every module's DP size.
    # slice_batch_for_megatron_mimo divides the micro-batch contiguously by the module's
    # DP size in forward_step; a non-divisible MBS would leave a remainder.
    micro_batch_size = cfg.train.micro_batch_size
    for mod_name, mod_cfg in cfg.model.megatron_mimo_parallelism_config.module_parallelisms.items():
        dp = mod_cfg.data_parallel_size
        if micro_batch_size % dp != 0:
            raise ValueError(
                f"micro_batch_size ({micro_batch_size}) must be divisible by "
                f"data_parallel_size ({dp}) of module '{mod_name}'. "
                f"slice_batch_for_megatron_mimo requires an evenly divisible micro-batch."
            )

    print_rank_0("> building MegatronMIMO train, validation, and test datasets ...")

    # Use cached grids from build_model()
    grids = cfg.model._grids

    scalable_dp = bool(getattr(getattr(cfg, "dataset", None), "megatron_mimo_scalable_dp", False))
    sampler_dp_rank, sampler_dp_size, needs_data = get_megatron_mimo_sampling_info(
        cfg.model.megatron_mimo_parallelism_config, grids, scalable_dp=scalable_dp
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
    train_ds, valid_ds, test_ds = megatron_mimo_provider.build_datasets(context)

    print_rank_0(
        f"  Built datasets: train={len(train_ds) if train_ds else 0}, "
        f"valid={len(valid_ds) if valid_ds else 0}, "
        f"test={len(test_ds) if test_ds else 0}"
    )

    # Default mode builds via the shared standard-path helper (sampler_dp_size=1: all
    # data-loading ranks see the same global batches; per-module DP sub-sharding is
    # deferred to slice_batch_for_megatron_mimo in the forward step). Scalable mode
    # shards reads on the canonical grid instead and skips the forward-step slice.
    collate_fn = megatron_mimo_provider.get_collate_fn()
    micro_batch_size = cfg.train.micro_batch_size

    def _make_loader(dataset, consumed_samples: int) -> Optional[DataLoader]:
        if dataset is None:
            return None
        if scalable_dp:
            # Shard reads on the canonical grid (LCM of the module DP sizes) so every
            # module materializes the same ordered global micro-batch under any sampler.
            module_dps = [
                p.data_parallel_size for p in cfg.model.megatron_mimo_parallelism_config.module_parallelisms.values()
            ]
            return build_canonical_mimo_data_loader(
                dataset,
                consumed_samples=consumed_samples,
                dataloader_type=megatron_mimo_provider.dataloader_type,
                micro_batch_size=micro_batch_size,
                module_dp_sizes=module_dps,
                dp_rank=sampler_dp_rank,
                dp_size=sampler_dp_size,
                data_sharding=megatron_mimo_provider.data_sharding,
                drop_last=megatron_mimo_provider.drop_last,
                num_workers=megatron_mimo_provider.num_workers,
                pin_memory=megatron_mimo_provider.pin_memory,
                collate_fn=collate_fn,
                persistent_workers=megatron_mimo_provider.persistent_workers,
            )
        return build_pretraining_data_loader(
            dataset=dataset,
            consumed_samples=consumed_samples,
            dataloader_type=megatron_mimo_provider.dataloader_type,
            micro_batch_size=micro_batch_size,
            num_workers=megatron_mimo_provider.num_workers,
            data_sharding=megatron_mimo_provider.data_sharding,
            collate_fn=collate_fn,
            pin_memory=megatron_mimo_provider.pin_memory,
            persistent_workers=megatron_mimo_provider.persistent_workers,
            data_parallel_rank=sampler_dp_rank,
            data_parallel_size=sampler_dp_size,
            drop_last=megatron_mimo_provider.drop_last,
        )

    train_loader = _make_loader(train_ds, consumed_samples=train_state.consumed_train_samples)
    valid_loader = _make_loader(valid_ds, consumed_samples=0)
    test_loader = _make_loader(test_ds, consumed_samples=0)

    return train_loader, valid_loader, test_loader
