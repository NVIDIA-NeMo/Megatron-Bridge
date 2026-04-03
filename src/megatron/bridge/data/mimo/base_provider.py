# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
        """
        Construct the provider's train, validation, and test datasets.
        
        Parameters:
            context (DatasetBuildContext): Build context containing sample counts used during dataset construction.
        
        Returns:
            Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]: A 3-tuple (train_dataset, valid_dataset, test_dataset); any element may be `None` if that split is not produced.
        """
        ...
    
    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """
        Provide the callable used to collate a list of samples into a batched `modality_inputs` dictionary.
        
        Returns:
            Callable[[List[Any]], Dict[str, Any]]: A callable that accepts a list of samples and returns a dictionary mapping modality keys to their batched tensors/structures.
        """
        ...
