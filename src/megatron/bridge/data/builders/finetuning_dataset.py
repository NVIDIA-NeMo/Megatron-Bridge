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

import logging
import os
import socket
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import torch
from megatron.core.msc_utils import MultiStorageClientFeature
from megatron.core.tokenizers.text.libraries import HuggingFaceTokenizer

from megatron.bridge.data.datasets.packed_parquet import is_packed_parquet_spec, resolve_packed_parquet_paths
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.data.datasets.sft import create_sft_dataset
from megatron.bridge.utils.common_utils import get_rank_safe, print_rank_0


logger = logging.getLogger(__name__)


class _PathVisibility(TypedDict):
    rank: int
    local_rank: str
    hostname: str
    error: Optional[str]
    paths: dict[str, "_PathProbe"]


class _PathProbe(TypedDict):
    path: str
    exists: bool
    error: Optional[str]
    required_if: Optional[str]


class _PathRequirement(TypedDict):
    label: str
    path: Union[str, Path]
    required_if: Optional[str]


class FinetuningDatasetBuilder:
    """Builder class for fine-tuning datasets.

    This class provides methods to build datasets for fine-tuning large language models.
    It follows a builder pattern similar to BlendedMegatronDatasetBuilder but adapted for
    fine-tuning scenarios.

    Args:
        dataset_root (Union[str, Path]): The root directory containing training, validation, and test data.
        tokenizer: The tokenizer to use for preprocessing text.
        is_built_on_rank (Callable): Function that returns True if the dataset should be built on current rank.
        seq_length (int, optional): The maximum sequence length. Defaults to 2048.
        seed (int, optional): Random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): Number of worker processes for memmap datasets. Defaults to 1.
        max_train_samples (int, optional): Maximum number of training samples. Defaults to None.
        enable_offline_packing (bool, optional): Whether to prepare and load offline packed sequences.
            Defaults to False.
        offline_packing_specs (Optional[PackedSequenceSpecs], optional): Specifications for offline packed
            sequences. Required when enable_offline_packing is True. Defaults to None.
        dataset_kwargs (Optional[dict[str, Any]], optional): Additional dataset creation arguments. Defaults to None.
        do_validation (bool, optional): Whether to build the validation dataset. Defaults to True.
        do_test (bool, optional): Whether to build the test dataset. Defaults to True.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        tokenizer,
        seq_length: int = 2048,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: Optional[int] = None,
        enable_offline_packing: bool = False,
        offline_packing_specs: Optional[PackedSequenceSpecs] = None,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        do_validation: bool = True,
        do_test: bool = True,
    ):
        if enable_offline_packing and offline_packing_specs is None:
            raise ValueError("offline_packing_specs must be set when enable_offline_packing=True.")
        if offline_packing_specs is not None and not enable_offline_packing:
            raise ValueError("enable_offline_packing must be True when offline_packing_specs is set.")
        if enable_offline_packing:
            assert offline_packing_specs is not None
            if offline_packing_specs.packed_sequence_size <= 0:
                raise ValueError("offline_packing_specs.packed_sequence_size must be greater than 0.")

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            self.dataset_root = msc.Path(dataset_root)
        else:
            self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        self.memmap_workers = memmap_workers
        self.max_train_samples = max_train_samples
        self.enable_offline_packing = enable_offline_packing
        self.offline_packing_specs = offline_packing_specs
        self.packed_sequence_size = -1 if not offline_packing_specs else offline_packing_specs.packed_sequence_size
        self.dataset_kwargs = dataset_kwargs or {}
        self._pad_cu_seqlens = False if not offline_packing_specs else offline_packing_specs.pad_cu_seqlens
        self._pad_seq_to_mult = None if not offline_packing_specs else offline_packing_specs.pad_seq_to_mult
        self._num_tokenizer_workers = -1 if not offline_packing_specs else offline_packing_specs.num_tokenizer_workers

        self.do_validation = do_validation
        self.do_test = do_test

        print_rank_0(f"Building FinetuningDatasetBuilder with root={self.dataset_root}")

        if self.packed_sequence_size > 0:
            print_rank_0(f"Using packed sequences with size {self.packed_sequence_size}")

    def prepare_data(self) -> None:
        """Prepare data if needed."""
        self.prepare_packed_data()

    def prepare_packed_data(self) -> None:
        """Prepare packed sequence data files if configured.

        Skips preparation if:
        - packed_sequence_size <= 0 (packing disabled)
        - packed data files already exist (parquet or legacy .npy)
        """
        if self.packed_sequence_size <= 0:
            return

        self._prepare_packed_split(
            split_name="training",
            packed_path=self.train_path_packed,
            input_path=self.train_path,
        )

        if not self.do_validation:
            return

        self._prepare_packed_split(
            split_name="validation",
            packed_path=self.validation_path_packed,
            input_path=self.validation_path,
        )

    def _prepare_packed_split(
        self,
        split_name: str,
        packed_path: Union[str, Path],
        input_path: Path,
    ) -> None:
        """Prepare a single packed data split if it doesn't already exist.

        Args:
            split_name: Name of the split (for logging).
            packed_path: Output path for the packed data.
            input_path: Input path to the raw dataset.
        """
        from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data

        if self._packed_path_exists(packed_path):
            print_rank_0(f"Skipping packed {split_name} data preparation - already exists: {packed_path}")
            return

        packed_path_str = str(packed_path)
        if packed_path_str.lower().endswith(".npy"):
            warnings.warn(
                "Automatic .npy packed sequence preparation is deprecated and will be removed in the next release. "
                "Please use packed parquet format instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            return

        print_rank_0(f"Preparing packed {split_name} data at {packed_path}")
        prepare_packed_sequence_data(
            input_path=input_path,
            output_path=packed_path,
            output_metadata_path=self.pack_metadata,
            packed_sequence_size=self.packed_sequence_size,
            tokenizer=self.tokenizer,
            max_seq_length=self.seq_length,
            seed=self.seed,
            dataset_kwargs=self.dataset_kwargs,
            pad_seq_to_mult=self._pad_seq_to_mult,
            num_tokenizer_workers=self._num_tokenizer_workers,
        )

    def _packed_path_exists(self, path: Union[str, Path]) -> bool:
        """Check if a packed data path exists.

        For .npy files: check file exists
        For packed parquet specs: check if resolution returns non-empty

        Args:
            path: The path to check

        Returns:
            True if the packed data exists
        """
        path_str = str(path)

        # For packed parquet specs, check if resolution returns files
        if is_packed_parquet_spec(path_str):
            try:
                resolved = resolve_packed_parquet_paths(path_str)
                return len(resolved) > 0
            except ValueError:
                return False

        # For .npy or other files, check existence
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return msc.Path(path_str).is_file()
        else:
            return Path(path_str).is_file()

    def _path_exists(self, path: Union[str, Path]) -> bool:
        """Check whether a dataset path resolves on the current rank."""
        path_str = str(path)
        if is_packed_parquet_spec(path_str):
            try:
                return len(resolve_packed_parquet_paths(path_str)) > 0
            except ValueError:
                return False

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            return bool(msc.Path(path_str).exists())
        return Path(path_str).exists()

    def _effective_metadata_path(
        self,
        path: Union[str, Path],
        pack_metadata_path: Optional[Union[str, Path]],
    ) -> Optional[Union[str, Path]]:
        """Resolve metadata required to load a packed dataset path."""
        if self.packed_sequence_size <= 0:
            return None
        if self._pad_cu_seqlens:
            return pack_metadata_path
        if is_packed_parquet_spec(str(path)):
            return None
        return pack_metadata_path

    def _dataset_path_requirements(self) -> list[_PathRequirement]:
        """Return all paths that must be probed before any dataset is constructed."""
        is_packing = self.packed_sequence_size > 0
        train_path = self.train_path_packed if is_packing else self.train_path
        pack_metadata_path = self.pack_metadata if is_packing else None

        requirements: list[_PathRequirement] = [{"label": "training data", "path": train_path, "required_if": None}]
        train_metadata_path = self._effective_metadata_path(train_path, pack_metadata_path)
        if train_metadata_path is not None:
            requirements.append(
                {
                    "label": "training packed metadata",
                    "path": train_metadata_path,
                    "required_if": "training data",
                }
            )

        if self.do_validation:
            validation_path = self.validation_path_packed if is_packing else self.validation_path
            requirements.append({"label": "validation data", "path": validation_path, "required_if": None})
            validation_metadata_path = self._effective_metadata_path(validation_path, pack_metadata_path)
            if validation_metadata_path is not None:
                requirements.append(
                    {
                        "label": "validation packed metadata",
                        "path": validation_metadata_path,
                        "required_if": "validation data",
                    }
                )

        if self.do_test:
            requirements.append({"label": "test data", "path": self.test_path, "required_if": None})
        return requirements

    @staticmethod
    def _describe_path_states(
        states: Sequence[tuple[_PathVisibility, Optional[_PathProbe]]],
        *,
        limit: int = 8,
    ) -> str:
        """Format a bounded sample of per-rank path states."""
        descriptions = []
        for state, probe in states[:limit]:
            path = "not configured" if probe is None else probe["path"]
            error = state["error"] if probe is None else probe["error"]
            detail = "" if error is None else f", error {error}"
            descriptions.append(
                f"rank {state['rank']} (local rank {state['local_rank']}, host {state['hostname']}, "
                f"path {path}{detail})"
            )
        omitted = len(states) - limit
        if omitted > 0:
            descriptions.append(f"... and {omitted} more rank(s)")
        return "; ".join(descriptions)

    def _preflight_dataset_paths(self) -> dict[str, bool]:
        """Probe every dataset path once and validate distributed visibility before loading."""
        is_distributed = torch.distributed.is_initialized()
        local_state: _PathVisibility = {
            "rank": torch.distributed.get_rank() if is_distributed else get_rank_safe(),
            "local_rank": os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "unknown")),
            "hostname": socket.gethostname(),
            "error": None,
            "paths": {},
        }

        try:
            requirements = self._dataset_path_requirements()
        except Exception as error:
            requirements = []
            local_state["error"] = f"{type(error).__name__}: {error}"

        for requirement in requirements:
            path_str = str(requirement["path"])
            try:
                path_exists = self._path_exists(path_str)
                probe_error = None
            except Exception as error:
                path_exists = False
                probe_error = f"{type(error).__name__}: {error}"
            local_state["paths"][requirement["label"]] = {
                "path": path_str,
                "exists": path_exists,
                "error": probe_error,
                "required_if": requirement["required_if"],
            }

        if is_distributed and torch.distributed.get_world_size() > 1:
            visibility: list[_PathVisibility] = [local_state.copy() for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(visibility, local_state)
        else:
            visibility = [local_state]

        issues = []
        state_errors = [(state, None) for state in visibility if state["error"] is not None]
        if state_errors:
            issues.append(f"Path configuration failed on: {self._describe_path_states(state_errors)}.")

        labels = sorted({label for state in visibility for label in state["paths"]})
        for label in labels:
            states_and_probes = [(state, state["paths"].get(label)) for state in visibility]
            unconfigured = [(state, probe) for state, probe in states_and_probes if probe is None]
            if unconfigured:
                issues.append(
                    f"Path requirement '{label}' is not configured on every rank: "
                    f"{self._describe_path_states(unconfigured)}."
                )
                continue

            probe_errors = [
                (state, probe)
                for state, probe in states_and_probes
                if probe is not None and probe["error"] is not None
            ]
            if probe_errors:
                issues.append(
                    f"Path requirement '{label}' could not be probed on: {self._describe_path_states(probe_errors)}."
                )
                continue

            visible = [(state, probe) for state, probe in states_and_probes if probe is not None and probe["exists"]]
            missing = [
                (state, probe) for state, probe in states_and_probes if probe is not None and not probe["exists"]
            ]
            if visible and missing:
                issues.append(
                    f"Dataset path visibility is inconsistent for '{label}'. "
                    f"Visible on {len(visible)} rank(s): {self._describe_path_states(visible)}. "
                    f"Missing on {len(missing)} rank(s): {self._describe_path_states(missing)}."
                )
                continue

            required_if = states_and_probes[0][1]["required_if"] if states_and_probes[0][1] is not None else None
            if required_if is not None and not visible:
                parent_visible = any(
                    (parent_probe := state["paths"].get(required_if)) is not None and parent_probe["exists"]
                    for state in visibility
                )
                if parent_visible:
                    issues.append(
                        f"Required path '{label}' is missing on every rank while '{required_if}' exists. "
                        f"Missing on {len(missing)} rank(s): {self._describe_path_states(missing)}."
                    )

        if issues:
            current_paths = [(local_state, probe) for probe in local_state["paths"].values()]
            current_description = self._describe_path_states(current_paths) if current_paths else "none"
            raise RuntimeError(
                "Dataset path preflight failed across distributed ranks after rank-0 preparation. "
                + " ".join(issues)
                + f" Current rank state: {current_description}. "
                "Set NEMO_HOME, NEMO_DATASETS_CACHE, or dataset_root to shared storage mounted at the same path "
                "on every node, or prepare the dataset on every node."
            )

        return {label: probe["exists"] for label, probe in local_state["paths"].items()}

    def build(self) -> list[Optional[Any]]:
        """Build train, validation, and test datasets.

        This method creates the necessary datasets based on the configuration.
        It first ensures data preparation (e.g., packing) is done (on rank 0),
        then builds the datasets potentially using the prepared files.

        Returns:
            A list containing the train, validation, and test datasets.
            Elements can be None if the corresponding data file doesn't exist
            or if dataset building is skipped for the split.
        """
        # Prepare packed data if needed
        if get_rank_safe() == 0:
            self.prepare_data()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        path_exists = self._preflight_dataset_paths()

        # This needs to be called on all ranks
        datasets: list[Optional[Any]] = self._build_datasets(path_exists)
        return datasets

    def _build_datasets(self, path_exists: dict[str, bool]) -> list[Optional[Any]]:
        """Internal method to build all datasets.

        Returns:
            list[Optional[Any]]: The train, validation, and test datasets.
        """
        train_ds = self._create_dataset(
            self.train_path if self.packed_sequence_size <= 0 else self.train_path_packed,
            pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
            path_exists=path_exists["training data"],
            max_num_samples=self.max_train_samples,
            **self.dataset_kwargs,
        )

        if self.do_validation:
            valid_ds = self._create_dataset(
                self.validation_path if self.packed_sequence_size <= 0 else self.validation_path_packed,
                pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
                path_exists=path_exists["validation data"],
                is_test=True,
                **self.dataset_kwargs,
            )
        else:
            valid_ds = None

        if self.do_test:
            test_ds = self._create_dataset(
                self.test_path,
                path_exists=path_exists["test data"],
                is_test=True,
                **self.dataset_kwargs,
            )
        else:
            test_ds = None

        return [train_ds, valid_ds, test_ds]

    def _create_dataset(
        self,
        path: Union[str, Path],
        pack_metadata_path: Optional[Union[str, Path]] = None,
        is_test: bool = False,
        path_exists: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Create a single dataset instance (train, validation, or test).

        Args:
            path: Path to the dataset file or packed parquet spec
            pack_metadata_path: Path to the packed sequence metadata
            is_test: Whether this is a test dataset
            path_exists: Result from the pre-construction path preflight, if available
            **kwargs: Additional arguments to pass to the dataset constructor

        Returns:
            The created dataset
        """
        if path_exists is None:
            path_exists = self._path_exists(path)

        if not path_exists:
            print_rank_0(f"Warning: Dataset path {path} does not exist")
            return None

        is_not_packing = self.packed_sequence_size <= 0

        # For packed parquet from external sources, only pass metadata if pad_cu_seqlens is True
        # This avoids "missing metadata" errors when using externally prepared packed data
        effective_metadata_path = self._effective_metadata_path(path, pack_metadata_path)

        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=(self.seq_length if is_not_packing else self.packed_sequence_size),
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            is_test=is_test,
            pack_metadata_file_path=effective_metadata_path,
            pad_cu_seqlens=False if is_not_packing else self._pad_cu_seqlens,
            pad_seq_to_mult=1 if is_not_packing else self._pad_seq_to_mult,
            **kwargs,
        )

    @property
    def train_path(self) -> Path:
        """Path to the training dataset file (training.jsonl)."""
        return self.dataset_root / "training.jsonl"

    @property
    def default_pack_path(self) -> Path:
        """The default directory path for storing packed sequence files.

        Constructed based on the dataset root and tokenizer model name.
        Creates the directory if it doesn't exist.

        Returns:
            The Path object for the default packing directory.
        """
        tokenizer_model_name = self._extract_tokenizer_model_name()
        default_pack_path = (
            self.dataset_root / "packed" / f"{tokenizer_model_name}_pad_seq_to_mult{self._pad_seq_to_mult}"
        )
        if not default_pack_path.exists():
            try:
                # Shared filesystems can expose stale parent-dir state despite exist_ok=True.
                default_pack_path.mkdir(parents=True, exist_ok=True)
            except (FileExistsError, FileNotFoundError):
                pass
            logger.info(f"Using default path for packing files: {str(default_pack_path)}")

        return default_pack_path

    @property
    def pack_metadata(self) -> Path:
        """Path to the metadata file for packed sequences.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed sequence metadata file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_metadata_path is not None:
                return self.offline_packing_specs.packed_metadata_path
            return self.default_pack_path / f"{self.packed_sequence_size}_metadata.jsonl"
        else:
            raise ValueError("pack_metadata invalid since packed sequence size is not specified.")

    @property
    def train_path_packed(self) -> Path:
        """Path to the packed training dataset file.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed training data file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_train_data_path is not None:
                return self.offline_packing_specs.packed_train_data_path
            return self.default_pack_path / f"training_{self.packed_sequence_size}.idx.parquet"
        else:
            raise ValueError("`train_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path_packed(self) -> Path:
        """Path to the packed validation dataset file.

        Determined by `offline_packing_specs` or defaults based on the
        `default_pack_path` and `packed_sequence_size`.

        Returns:
            The Path object for the packed validation data file.

        Raises:
            ValueError: If packed sequences are not configured.
        """
        if self.packed_sequence_size > 0:
            if self.offline_packing_specs.packed_val_data_path is not None:
                return self.offline_packing_specs.packed_val_data_path
            return self.default_pack_path / f"validation_{self.packed_sequence_size}.idx.parquet"
        else:
            raise ValueError("`validation_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path(self) -> Path:
        """Path to the validation dataset file (validation.jsonl)."""
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        """Path to the test dataset file (test.jsonl)."""
        return self.dataset_root / "test.jsonl"

    def _extract_tokenizer_model_name(self) -> str:
        """Automatically get the model name from model path."""
        # Legacy tokenizer compatibility
        tokenizer_cls = HuggingFaceTokenizer
        tokenizer_instance = self.tokenizer._tokenizer

        if self.offline_packing_specs and self.offline_packing_specs.tokenizer_model_name is not None:
            return self.offline_packing_specs.tokenizer_model_name
        elif isinstance(tokenizer_instance, tokenizer_cls):
            name = self.tokenizer.path

            if name.endswith("context/nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/context/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-4:-2])
            elif name.endswith("nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-3:-1])
            else:
                # hf_org/hf_model => hf_org--hf_model
                tokenizer_model_name = name.replace("/", "--")
            return tokenizer_model_name
        else:
            return f"unknown_tokenizer_{hash(self.tokenizer)}"
