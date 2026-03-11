# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Base class for MIMO dataset providers."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from torch.utils.data import Dataset

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class MimoDatasetProvider(DatasetProvider):
    """Abstract base class for MIMO dataset providers.

    All MIMO dataset providers must inherit from this class and implement
    the required methods. This ensures a consistent interface for MIMO
    data loading.

    Required methods:
        - build_datasets: Build train/valid/test datasets
        - get_collate_fn: Return the collate function for batching

    Example:
        >>> class MyMimoProvider(MimoDatasetProvider):
        ...     def build_datasets(self, context):
        ...         # Build and return datasets
        ...         return train_ds, valid_ds, test_ds
        ...
        ...     def get_collate_fn(self):
        ...         # Return collate function
        ...         return my_collate_fn
    """

    @abstractmethod
    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Build train, validation, and test datasets.

        Args:
            context: Build context with sample counts.

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset).
            Any element can be None if not needed.
        """
        ...

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """Return the collate function for batching.

        The collate function should handle the modality_inputs dict
        and batch them appropriately for the model.

        Returns:
            Callable that takes a list of samples and returns a batch dict.
        """
        ...
