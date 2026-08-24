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

import importlib
import logging
from typing import Any, Callable, Literal, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.energon import Sample, WorkerConfig, get_savable_loader, get_train_dataset
from megatron.energon import dataset_config as _energon_dataset_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.default_generic_webdataset import DefaultGenericWebdatasetFactory
from megatron.energon.typed_converter import JsonParser


logger = logging.getLogger(__name__)

_ORIGINAL_METHOD_ATTRIBUTE = "__megatron_bridge_original_method__"
_TRUSTED_DATASET_FACTORY_MODULE_PREFIXES = (
    "megatron.bridge.data.energon",
    "megatron.energon",
)
_TRUSTED_DATASET_FACTORY_MODULE_ALIASES = frozenset(
    {
        "megatron.bridge.models.qwen_vl.data.energon",
    }
)
_energon_factory_init = getattr(
    DefaultGenericWebdatasetFactory.__init__,
    _ORIGINAL_METHOD_ATTRIBUTE,
    DefaultGenericWebdatasetFactory.__init__,
)
_energon_load_config = getattr(
    _energon_dataset_config.load_config,
    _ORIGINAL_METHOD_ATTRIBUTE,
    _energon_dataset_config.load_config,
)


def _validate_energon_dataset_metadata(value: Any, *, root: bool = True) -> None:
    """Reject executable object references in untrusted dataset factory metadata."""
    if isinstance(value, list):
        for item in value:
            _validate_energon_dataset_metadata(item, root=False)
        return
    if not isinstance(value, dict):
        return

    module_name = value.get("__module__")
    function_name = value.get("__function__")
    class_name = value.get("__class__")
    if function_name is not None:
        raise ValueError(
            "Energon dataset metadata cannot resolve serialized Python functions. "
            "Use declarative configuration or pass a Python callable from trusted application code."
        )
    if class_name is not None:
        trusted_class = (
            isinstance(module_name, str)
            and isinstance(class_name, str)
            and (
                module_name in _TRUSTED_DATASET_FACTORY_MODULE_ALIASES
                or any(
                    module_name == prefix or module_name.startswith(f"{prefix}.")
                    for prefix in _TRUSTED_DATASET_FACTORY_MODULE_PREFIXES
                )
            )
        )
        if trusted_class:
            module = importlib.import_module(module_name)
            referenced_type = getattr(module, class_name, None)
            required_base = DefaultGenericWebdatasetFactory if root else Sample
            trusted_class = isinstance(referenced_type, type) and issubclass(referenced_type, required_base)
        if not trusted_class:
            raise ValueError(
                "Energon dataset metadata cannot instantiate serialized Python classes. "
                "Only packaged Energon dataset factory and sample classes are allowed."
            )

    for item in value.values():
        _validate_energon_dataset_metadata(item, root=False)


def _secure_energon_load_config(
    path: EPath | dict[str, Any],
    *,
    default_type: type,
    default_kwargs: dict[str, Any] | None = None,
    parser: JsonParser = JsonParser(strict=True),
) -> Any:
    """Validate dataset metadata before Energon resolves serialized objects."""
    is_dataset_factory = isinstance(default_type, type) and issubclass(default_type, DefaultGenericWebdatasetFactory)
    if not is_dataset_factory:
        return _energon_load_config(
            path,
            default_type=default_type,
            default_kwargs=default_kwargs,
            parser=parser,
        )

    if isinstance(path, dict):
        data = path
    else:
        with path.open("rb") as config_file:
            data = _energon_dataset_config.load_yaml(config_file)
    _validate_energon_dataset_metadata(data)
    return _energon_load_config(
        data,
        default_type=default_type,
        default_kwargs=default_kwargs,
        parser=parser,
    )


def _secure_energon_factory_init(
    self: DefaultGenericWebdatasetFactory,
    path: EPath,
    *,
    subflavors: dict[str, Any] | None = None,
    field_map: dict[str, str] | None = None,
    sample_loader: str | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    part_filter: str | list[str] | Callable[[str], bool] | None = None,
    **kwargs: Any,
) -> None:
    """Reject dataset-local Python hooks before Energon resolves their files."""
    executable_fields = [
        name
        for name, value in (("sample_loader", sample_loader), ("part_filter", part_filter))
        if isinstance(value, str)
    ]
    if executable_fields:
        raise ValueError(
            "Energon dataset metadata cannot load Python files through "
            f"{', '.join(executable_fields)}. Use a declarative field_map instead."
        )
    _energon_factory_init(
        self,
        path,
        subflavors=subflavors,
        field_map=field_map,
        sample_loader=sample_loader,
        part_filter=part_filter,
        **kwargs,
    )


# Energon constructs this factory internally after reading dataset.yaml, so
# Bridge must install the guard before calling get_train_dataset().
setattr(_secure_energon_load_config, _ORIGINAL_METHOD_ATTRIBUTE, _energon_load_config)
_energon_dataset_config.load_config = _secure_energon_load_config
setattr(_secure_energon_factory_init, _ORIGINAL_METHOD_ATTRIBUTE, _energon_factory_init)
DefaultGenericWebdatasetFactory.__init__ = _secure_energon_factory_init


class EnergonMultiModalDataModule:
    """
    A DataModule for handling multimodal datasets with images and text.

    This data module is designed to work with multimodal datasets that involve both images and text.
    It provides a seamless interface to load training and validation data, manage batching, and handle
    the state of the data pipeline across training epochs. The module integrates with the Megatron-Energon
    framework for efficient data handling in large-scale distributed training.

    Attributes:
    path (str): Path to the energon dataset.
    tokenizer (Tokenizer): The tokenizer used for processing text.
    seq_length (int): The maximum sequence length for tokenized text.
    micro_batch_size (int): The batch size for training and validation.
    num_workers (int): Number of workers for data loading.
    pin_memory (bool): Whether to pin memory in the DataLoader.
    multimodal_sample_config (MultiModalSampleConfig): Configuration object for multimodal samples.
    task_encoder (MultiModalTaskEncoder): Encoder responsible for encoding and batching samples.
    init_global_step (int): The initial global step for the trainer, used for resuming training.
    train_dataloader_object (Optional): The DataLoader object for training data.
    val_dataloader_object (Optional): The DataLoader object for validation data.
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        seq_length: int = 2048,
        micro_batch_size: int = 1,
        num_workers: int = 1,
        num_val_workers: int | None = None,
        pin_memory: bool = True,
        shuffle_buffer_size: int = 100,
        max_samples_per_sequence: int | None = None,
        multimodal_sample_config: Optional[Any] = None,
        task_encoder: Optional[Any] = None,
        decoder_seq_length: Optional[int] = None,
        packing_buffer_size: Optional[int] = None,
        validation_task_encoder: Optional[Any] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the EnergonMultiModalDataModule.

        Parameters:
        path (str): Path to the dataset.
        tokenizer (Tokenizer): The tokenizer used for processing text.
        seq_length (int, optional): The maximum sequence length for tokenized text. Defaults to 2048.
        micro_batch_size (int, optional): The batch size for training and validation. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 1.
        num_val_workers (int, optional): Number of workers for validation data loading. Defaults to num_workers.
        pin_memory (bool, optional): Whether to pin memory in the DataLoader. Defaults to True.
        multimodal_sample_config (MultiModalSampleConfig, optional): Configuration object for multimodal samples.
        Defaults to MultiModalSampleConfig().
        shuffle_buffer_size (int, optional): Size of the shuffle buffer. Defaults to 100.
        max_samples_per_sequence (int, optional): Maximum number of samples per sequence to load from memory.
        Defaults to None (loads the whole tar file at once).
        task_encoder (MultiModalTaskEncoder, optional): Encoder responsible for encoding and batching samples.
        If not provided, a default (MultimodalTaskEncoder) encoder will be created. Defaults to None.
        decoder_seq_length (int, optional): The max sequence length for the decoder. Used in encoder-decoder models
        packing_buffer_size (int, optional): Size of the packing buffer for batched samples. Defaults to None.
        validation_task_encoder (MultiModalTaskEncoder, optional): Encoder responsible for encoding
        and batching samples for validation. Defaults to None and will be the same as task_encoder.
        pg_collection (ProcessGroupCollection, optional): Process group collection for distributed training.
        If provided, used instead of the global parallel_state. Defaults to None.
        **kwargs: Additional keyword arguments. Will be passed to get_train_dataset() of Energon
        """

        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.decoder_seq_length = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multimodal_sample_config = multimodal_sample_config
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_samples_per_sequence = max_samples_per_sequence
        self.task_encoder = task_encoder
        self.init_global_step = 0
        self.train_dataloader_object = None
        self.val_dataloader_object = None
        self.packing_buffer_size = packing_buffer_size
        self.validation_task_encoder = validation_task_encoder or self.task_encoder
        self.num_val_workers = self.num_workers if num_val_workers is None else num_val_workers
        self.pg_collection = pg_collection
        self.kwargs = kwargs

    def _build_worker_config(self, num_workers: int, split: str = "train") -> WorkerConfig:
        """Build a WorkerConfig using pg_collection, falling back to default_worker_config.

        NOTE: We intentionally use the pure DP rank (pg_collection.dp)
        rather than the combined DP-CP rank. With Megatron's rank ordering
        (default "tp-cp-ep-dp-pp"), all CP ranks within the same DP replica
        already share the same pure DP rank. This ensures that CP ranks
        processing different sequence portions of the same batch receive
        identical data from the dataloader.
        Using dp_cp would be INCORRECT here — it would assign each CP rank
        a unique rank, causing them to read different data shards.
        """
        if self.pg_collection is None or self.pg_collection.dp is None:
            logger.info(
                f"Multimodal {split} data loader pg_collection is not available, "
                f"using default worker config with num_workers {num_workers}"
            )
            return WorkerConfig.default_worker_config(num_workers)

        rank = self.pg_collection.dp.rank()
        world_size = self.pg_collection.dp.size()
        data_parallel_group = self.pg_collection.dp
        cp_rank = self.pg_collection.cp.rank() if self.pg_collection.cp is not None else 0
        cp_size = self.pg_collection.cp.size() if self.pg_collection.cp is not None else 1

        logger.info(
            f"Multimodal {split} dataloader initializing with "
            f"dp_rank {rank} dp_world_size {world_size} "
            f"cp_rank {cp_rank} cp_size {cp_size} "
            f"data_parallel_group {data_parallel_group}"
        )
        return WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            data_parallel_group=data_parallel_group,
            worker_debug_path=None,
            worker_log_level=0,
        )

    def datasets_provider(self, worker_config, split: Literal["train", "val"] = "val"):
        """
        Provide the dataset for training or validation.

        This method retrieves the dataset for the specified split (either 'train' or 'val') and configures
        it according to the worker configuration.

        Parameters:
        worker_config: Configuration for the data loader workers.
        split (Literal['train', 'val'], optional): The data split to retrieve ('train' or 'val'). Defaults to 'val'.

        Returns:
        Dataset: The dataset configured for the specified split.
        """

        if split not in {"train", "val"}:
            raise ValueError("Invalid value for split. Allowed values are 'train' or 'val'.")

        if split == "train":
            task_encoder = self.task_encoder
        else:
            task_encoder = self.validation_task_encoder

        _dataset = get_train_dataset(
            self.path,
            batch_size=self.micro_batch_size,
            task_encoder=task_encoder,
            worker_config=worker_config,
            packing_buffer_size=self.packing_buffer_size,
            split_part=split,
            shuffle_buffer_size=self.shuffle_buffer_size,
            max_samples_per_sequence=self.max_samples_per_sequence,
            **self.kwargs,
        )

        return _dataset

    def build(self):
        return self.train_dataloader(), self.val_dataloader()

    def train_dataloader(self) -> Any:
        """
        Initialize and return the training DataLoader.

        Returns:
        TRAIN_DATALOADERS: The DataLoader for the training dataset.
        """

        logger.info(f"Multimodal train dataloader initializing with init_global_step {self.init_global_step}")
        if self.train_dataloader_object:
            return self.train_dataloader_object
        worker_config = self._build_worker_config(self.num_workers, split="train")
        train_dataset = self.datasets_provider(worker_config, split="train")
        energon_dataloader = get_savable_loader(train_dataset, worker_config=worker_config)
        self.train_dataloader_object = energon_dataloader
        return EnergonDataloader(self.train_dataloader_object)

    def val_dataloader(self):
        """
        Initialize and return the validation DataLoader.

        This method initializes the DataLoader for the validation dataset. It ensures that the parallel state
        is initialized correctly for distributed training and returns a configured DataLoader object.

        Returns:
        EVAL_DATALOADERS: The DataLoader for the validation dataset.
        """
        if self.val_dataloader_object is None:
            worker_config = self._build_worker_config(self.num_val_workers, split="val")
            val_dataset = self.datasets_provider(worker_config, split="val")
            energon_loader = get_savable_loader(val_dataset, worker_config=worker_config)
            self.val_dataloader_object = energon_loader
        return EnergonDataloader(self.val_dataloader_object)

    def test_dataloader(self) -> None:
        """
        Return None as test dataset split does not exist.

        This method overrides the test_dataloader method and returns None since the test dataset split
        is not defined or used in this module.

        Returns:
        None
        """
        logger.warning("Multimodal dataloader test dataset split does not exist")
        return None


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()

    def restore_state(self, state):
        """Restore the underlying Energon loader to a saved stream position and rebuild the iterator.

        Must run before iteration starts (no workers spawned yet), which holds when called right
        after dataloader construction on resume.
        """
        self._dataloader.restore_state_rank(state)
        self._iter = iter(cyclic_iter(self._dataloader))


def cyclic_iter(iter):
    """
    Create a cyclic iterator that loops over the given iterable indefinitely.

    Args:
        iter (iterable): The input iterable to cycle through.

    Yields:
        Any: The next item from the iterable, looping back to the start when exhausted.
    """
    while True:
        for x in iter:
            yield x
