# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Offline materialization of packed GPT SFT artifacts."""

import json
import logging
import resource
from collections.abc import Callable
from multiprocessing import Pool
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from megatron.core.msc_utils import MultiStorageClientFeature
from tqdm import tqdm

from megatron.bridge.data.packing.algorithms import (
    create_hist,
    create_packing_strategy,
    fill_packing_strategy,
    iter_packing_strategy,
)
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


logger = logging.getLogger(__name__)

_shared_dataset = None
_PACKING_ITEM_KEYS = ("input_ids", "loss_mask", "answer_start_idx")


def _storage_path(path: str | Path) -> Any:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.Path(str(path))
    return Path(path)


def _temporary_sibling(path: str | Path) -> Any:
    destination = _storage_path(path)
    suffix = destination.suffix
    stem = destination.name[: -len(suffix)] if suffix else destination.name
    return destination.with_name(f".{stem}.{uuid4().hex}.tmp{suffix}")


def _publish_staged_path(staged_path: Any, destination_path: str | Path) -> None:
    destination = _storage_path(destination_path)
    if MultiStorageClientFeature.is_enabled():
        staged_path.rename(destination)
    else:
        staged_path.replace(destination)


def _remove_staged_path(staged_path: Any | None) -> None:
    if staged_path is not None:
        staged_path.unlink(missing_ok=True)


def _select_packing_item_fields(item):
    if not isinstance(item, dict):
        return item
    selected = {}
    for key in _PACKING_ITEM_KEYS:
        if key not in item:
            continue
        value = item[key]
        if key in {"input_ids", "loss_mask"}:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.asarray(value)
        selected[key] = value
    return selected


def _get_shared_dataset_item(i):
    try:
        return _select_packing_item_fields(_shared_dataset[i])
    except Exception as error:
        error.add_note(f"Failed to prepare packed-SFT dataset index {i}.")
        raise


def _init_shared_dataset_worker(dataset):
    global _shared_dataset
    _shared_dataset = dataset


def _materialize_dataset_items(dataset, num_workers):
    if num_workers <= 1:
        return np.array([_select_packing_item_fields(dataset[i]) for i in tqdm(range(len(dataset)))])

    # File-backed tensor sharing avoids one descriptor per returned tensor; the pool still needs descriptors.
    previous_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
    torch.multiprocessing.set_sharing_strategy("file_system")

    previous_nofile_limit = None
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit != hard_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
            previous_nofile_limit = (soft_limit, hard_limit)
    except (ValueError, OSError) as error:
        logger.warning("Unable to raise the file-descriptor limit for tokenizer workers: %s", error)

    try:
        with Pool(num_workers, initializer=_init_shared_dataset_worker, initargs=(dataset,)) as pool:
            items = tqdm(pool.imap(_get_shared_dataset_item, range(len(dataset))), total=len(dataset))
            return np.array(list(items))
    finally:
        if previous_nofile_limit is not None:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, previous_nofile_limit)
            except (ValueError, OSError) as error:
                logger.warning("Unable to restore the file-descriptor limit after tokenization: %s", error)
        torch.multiprocessing.set_sharing_strategy(previous_sharing_strategy)


def _pre_pad_data_point(data: dict, max_seq_length: int, max_stored_length_to_pad: int, pad_id: int) -> None:
    """Pad a single data point so its runtime segment length is divisible by the requested multiple.

    Pads ``input_ids``/``context_ids`` with ``pad_id`` and ``loss_mask`` with ``0`` (no loss on
    pad positions). The chat preprocessing path (``_chat_preprocess``) returns ``torch`` tensors;
    those and NumPy arrays retain their compact representation so corpus-scale preparation does not
    expand every token into a Python object. Plain-list inputs remain lists. ``loss_mask`` stays the
    same length as ``input_ids`` so grouped samples cannot produce ragged packing inputs.

    Args:
        data: A single tokenized example. Mutated in place.
        max_seq_length: Hard upper bound for the runtime sequence length after next-token shifting.
        max_stored_length_to_pad: Stored target length to pad/truncate to. This is the divisible runtime
            target plus one token because packed SFT labels are derived by shifting ``input_ids``.
        pad_id: Token id used to pad ``input_ids``/``context_ids``.
    """
    assert max_seq_length + 1 >= max_stored_length_to_pad
    # loss_mask must be padded too (with 0), otherwise samples that round to the same padded
    # input_ids length but had different original lengths keep mismatched loss_mask lengths.
    pad_values = {"input_ids": pad_id, "context_ids": pad_id, "loss_mask": 0}
    for key, pad_value in pad_values.items():
        if key not in data:
            continue
        val = data[key]
        sequence_length = len(val)
        if sequence_length > max_stored_length_to_pad:
            if sequence_length > max_seq_length + 1:
                logger.info(
                    "Sequence length %d exceeds max_seq_length %d; truncating for packing.",
                    sequence_length,
                    max_seq_length,
                )
            if torch.is_tensor(val):
                val = val[:max_stored_length_to_pad].clone()
            elif isinstance(val, np.ndarray):
                val = val[:max_stored_length_to_pad].copy()
            else:
                val = list(val[:max_stored_length_to_pad])
        elif sequence_length < max_stored_length_to_pad:
            padding_length = max_stored_length_to_pad - sequence_length
            if torch.is_tensor(val):
                val = torch.cat((val, val.new_full((padding_length,), pad_value)))
            elif isinstance(val, np.ndarray):
                val = np.concatenate((val, np.full(padding_length, pad_value, dtype=val.dtype)))
            else:
                val = list(val) + [pad_value] * padding_length
        data[key] = val
    return


def tokenize_dataset(
    path: Path,
    tokenizer: MegatronTokenizer,
    max_seq_length: int,
    seed: int,
    dataset_kwargs: dict | None = None,
    pad_seq_to_mult: int | None = 1,
    num_tokenizer_workers: int = -1,
    *,
    dataset_builder: Callable[..., Any],
):
    """
    Tokenizes a dataset from the provided path using the specified tokenizer
    and prepares it for further processing.

    Args:
        path (Path): Path to the dataset file.
        tokenizer (MegatronTokenizer): The tokenizer to use for tokenization.
        max_seq_length (int): Maximum sequence length for the tokens.
        seed (int): Random seed for shuffling the dataset.
        dataset_kwargs (dict | None): Additional GPT SFT dataset construction options.
            Can include 'chat', 'use_hf_tokenizer_chat_template', 'tool_schemas', etc.
        pad_seq_to_mult (int | None): Optional multiple to pad each sequence to during packing
            preparation (e.g., set to 2 * context_parallel_size for THD CP).
        num_tokenizer_workers: Number of worker processes used to materialize tokenized samples.
            Values less than or equal to 1 run serially.
        dataset_builder: Builder-owned callable that constructs one unpacked GPT SFT split.

    Returns:
        np.ndarray: A NumPy array containing the tokenized data.
    """
    if not dataset_kwargs:
        dataset_kwargs = {}
    # Handle tool_schemas - convert to JSON string if needed
    ts = dataset_kwargs.get("tool_schemas")
    if ts and not isinstance(ts, str):
        dataset_kwargs["tool_schemas"] = json.dumps(ts)

    # Handle chat_template - set it on tokenizer if provided
    chat_template = dataset_kwargs.pop("chat_template", None)
    if chat_template:
        # This is called during packing preparation (rank 0 only).
        # The chat template is only needed to create the packed .npy files.
        # Once created, all ranks load the pre-tokenized .npy files.
        if hasattr(tokenizer, "_tokenizer"):
            tokenizer._tokenizer.chat_template = chat_template

    if pad_seq_to_mult is not None and pad_seq_to_mult <= 0:
        raise ValueError("pad_seq_to_mult must be a positive integer when provided.")

    # Keep the historical minimum of 16 unless a larger multiple is requested.
    pad_seq_length_to_mult = 1 if pad_seq_to_mult is None else max(1, pad_seq_to_mult)
    runtime_max_seq_length = max_seq_length
    max_runtime_pad_cap = None
    if pad_seq_length_to_mult > 1:
        # Runtime segments, not stored input_ids, must be divisible for THD+CP. Stored samples
        # carry one extra token because packed SFT labels are derived by shifting input_ids.
        max_runtime_pad_cap = (runtime_max_seq_length // pad_seq_length_to_mult) * pad_seq_length_to_mult
        if max_runtime_pad_cap == 0:
            raise ValueError(
                f"max_seq_length ({runtime_max_seq_length}) must be at least the effective padding multiple "
                f"({pad_seq_length_to_mult})."
            )
    stored_max_seq_length = runtime_max_seq_length + 1 if max_runtime_pad_cap is not None else runtime_max_seq_length

    dataset = dataset_builder(
        path,
        tokenizer=tokenizer,
        seq_length=stored_max_seq_length,
        memmap_workers=2,
        seed=seed,
        packed_sequence_size=-1,
        is_test=True,
        dataset_kwargs={"pad_seq_length_to_mult": pad_seq_length_to_mult, **dataset_kwargs},
    )
    if dataset is None:
        raise FileNotFoundError(f"GPT SFT input path does not exist: {path}")

    pad_id = dataset.tokenizer.eod
    pad_seq_length_to_mult = dataset.pad_seq_length_to_mult
    max_seq_length = runtime_max_seq_length

    dataset = _materialize_dataset_items(dataset, num_tokenizer_workers)

    if max_runtime_pad_cap is not None:

        def ceil_to_nearest(n, m):
            return (n + m - 1) // m * m

        for data in dataset:
            runtime_len = max(len(data["input_ids"]) - 1, 0)
            runtime_length_to_pad = min(max_runtime_pad_cap, ceil_to_nearest(runtime_len, pad_seq_length_to_mult))
            max_stored_length_to_pad = runtime_length_to_pad + 1
            _pre_pad_data_point(data, max_seq_length, max_stored_length_to_pad, pad_id)

    return dataset


def prepare_gpt_sft_packed_data(
    input_path: Path,
    output_path: Path,
    output_metadata_path: Path,
    packed_sequence_size: int,
    tokenizer: MegatronTokenizer,
    max_seq_length: int,
    seed: int | None = 0,
    packing_algorithm: str = "first_fit_shuffle",
    dataset_kwargs: dict | None = None,
    pad_seq_to_mult: int | None = 1,
    num_tokenizer_workers: int = -1,
    stream_packed_parquet: bool = False,
    *,
    dataset_builder: Callable[..., Any],
):
    """
    Prepares a packed sequence dataset from a given input file and saves it to an output file.

    Args:
        input_path (Path): Path to the input dataset file.
        output_path (Path): Path to save the packed sequence data.
        output_metadata_path (Path): Path to save packing metadata.
        packed_sequence_size (int): The maximum size for each packed sequence.
        tokenizer (MegatronTokenizer): The tokenizer to use for tokenization.
        max_seq_length (int): Maximum sequence length for the tokens.
        seed (int | None): Random seed for shuffling (optional).
        packing_algorithm (str): The algorithm used for packing sequences
                currently supports "first_fit_shuffle" and "first_fit_decreasing".
        dataset_kwargs (dict | None): Additional GPT SFT dataset construction options.
            Enables packing with chat templates, tool schemas, etc.
        pad_seq_to_mult (int | None): Optional multiple to pad each sequence to during packing
            preparation (e.g., set to 2 * context_parallel_size for THD CP).
        num_tokenizer_workers: Number of worker processes used to materialize tokenized samples.
            Values less than or equal to 1 run serially.
        stream_packed_parquet: Fill and write Parquet row groups incrementally instead of retaining
            corpus-sized Python token lists. This option is not supported for legacy NumPy output.
        dataset_builder: Builder-owned callable that constructs one unpacked GPT SFT split.

    Returns:
        None: Saves the packed sequence data to the specified output path.
    """
    output_path_str = str(output_path)
    is_parquet_output = output_path_str.lower().endswith((".parquet", ".pq"))
    if stream_packed_parquet and not is_parquet_output:
        raise ValueError("stream_packed_parquet requires a Parquet output path.")
    logger.info(f"Preparing packed sequence from {input_path}")
    dataset = tokenize_dataset(
        input_path,
        tokenizer,
        max_seq_length,
        seed,
        dataset_kwargs,
        pad_seq_to_mult=pad_seq_to_mult,
        num_tokenizer_workers=num_tokenizer_workers,
        dataset_builder=dataset_builder,
    )
    sequences, histogram = create_hist(dataset, max_seq_length)

    random_state = np.random.get_state()
    np.random.seed(seed)
    staged_output_path = None
    try:
        assignments, packing_metadata = create_packing_strategy(histogram, packed_sequence_size, packing_algorithm)

        if stream_packed_parquet:
            from megatron.bridge.data.packing.parquet import write_packed_parquet_streaming

            staged_output_path = _temporary_sibling(output_path)
            output_rows = iter_packing_strategy(assignments, sequences, packed_sequence_size, tokenizer.eos_id)
            try:
                write_packed_parquet_streaming(
                    output_rows,
                    staged_output_path,
                    _publish_on_completion=False,
                )
            except Exception:
                _remove_staged_path(staged_output_path)
                raise
        else:
            output_data = fill_packing_strategy(assignments, sequences, packed_sequence_size, tokenizer.eos_id)
    finally:
        np.random.set_state(random_state)

    if not stream_packed_parquet:
        # Keep every materialized output hidden until its metadata is ready,
        # matching the streaming path's retry-safe publication contract.
        staged_output_path = _temporary_sibling(output_path)
        try:
            if is_parquet_output:
                from megatron.bridge.data.packing.parquet import write_packed_parquet

                write_packed_parquet(output_data, staged_output_path)
            else:
                # Legacy .npy format. The staged sibling retains the .npy
                # suffix so NumPy does not silently append a second suffix.
                if MultiStorageClientFeature.is_enabled():
                    msc = MultiStorageClientFeature.import_package()
                    msc.numpy.save(staged_output_path, output_data)
                else:
                    np.save(staged_output_path, output_data)
        except Exception:
            _remove_staged_path(staged_output_path)
            raise

    # Save packing metadata. Streaming output stays at a temporary sibling
    # until metadata is complete, so a failed preparation cannot leave a
    # discoverable packed dataset that the builder would accept on retry.
    staged_metadata_path = None
    metadata_backup_path = None
    metadata_path = None
    metadata_published = False
    try:
        if output_metadata_path is not None:
            metadata_path = _storage_path(output_metadata_path)
            staged_metadata_path = _temporary_sibling(output_metadata_path)
            metadata_existed = True
            try:
                with metadata_path.open(mode="r") as f:
                    packing_metadata_file = json.load(f)
                    # 'packing_metadata_file' is expected to be a list of dicts: List[Dict[str, int]]
                    # Each dict corresponds to a packed dataset. Typically there will be two dicts,
                    # one each for the packed val and train datasets.
                    # Each dict records two values: 'max_samples_per_bin', the max
                    # number of samples per packed sequence, and 'dataset_max_seqlen', the max
                    # sequence length per sample in the packed dataset.
                    assert isinstance(packing_metadata_file, list), "invalid packing_metadata_file!"
            except FileNotFoundError:
                metadata_existed = False
                packing_metadata_file = []

            packing_metadata_file.append(packing_metadata)
            with staged_metadata_path.open(mode="w") as f:
                json.dump(packing_metadata_file, f)

            # Keep the previous metadata recoverable until the packed dataset
            # is published. The dataset path is the builder's completion gate,
            # so metadata must be rolled back if that final publication fails.
            if metadata_existed:
                metadata_backup_path = _temporary_sibling(output_metadata_path)
                _publish_staged_path(metadata_path, metadata_backup_path)
            try:
                _publish_staged_path(staged_metadata_path, output_metadata_path)
            except Exception:
                if metadata_backup_path is not None:
                    _publish_staged_path(metadata_backup_path, output_metadata_path)
                    metadata_backup_path = None
                raise
            staged_metadata_path = None
            metadata_published = True

        if staged_output_path is not None:
            try:
                _publish_staged_path(staged_output_path, output_path)
            except Exception:
                if metadata_published:
                    if metadata_backup_path is not None:
                        _publish_staged_path(metadata_backup_path, output_metadata_path)
                        metadata_backup_path = None
                    else:
                        metadata_path.unlink(missing_ok=True)
                raise
            staged_output_path = None
    except Exception:
        _remove_staged_path(staged_metadata_path)
        _remove_staged_path(staged_output_path)
        raise

    if metadata_backup_path is not None:
        try:
            _remove_staged_path(metadata_backup_path)
        except Exception as error:
            logger.warning("Unable to remove packed metadata backup %s: %s", metadata_backup_path, error)

    logger.info(f"Packed sequence is prepared and saved to {output_path}")
