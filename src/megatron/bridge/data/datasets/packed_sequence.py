# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from typing import Literal, List

import numpy as np
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.datasets.packing_utils import create_hist, create_packing_strategy, fill_packing_strategy
from megatron.bridge.data.datasets.sft import create_sft_dataset
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


logger = logging.getLogger(__name__)

_shared_dataset = None
def _tokenize_get_item(i):
    return _shared_dataset[i]

def _tokenize_init_worker(dataset):
    global _shared_dataset
    _shared_dataset = dataset

def _retrieve_tokenized(dataset, num_workers):
    if num_workers == 1:
        return np.array([dataset[i] for i in tqdm(range(len(dataset)))])
    num_workers = num_workers if num_workers > 0 else mp.cpu_count()
    with Pool(num_workers, initializer=_tokenize_init_worker, initargs=(dataset,)) as pool:
        return np.array(list(tqdm(pool.imap(_tokenize_get_item, range(len(dataset))), total=len(dataset))))

def tokenize_dataset(
    path: Path,
    tokenizer: MegatronTokenizer,
    max_seq_length: int,
    seed: int,
    dataset_kwargs: dict | None = None,
    pad_seq_to_mult: int | None = 1,
    num_tokenizer_workers: int = -1,
):
    """
    Tokenizes a dataset from the provided path using the specified tokenizer
    and prepares it for further processing.

    Args:
        path (Path): Path to the dataset file.
        tokenizer (MegatronTokenizer): The tokenizer to use for tokenization.
        max_seq_length (int): Maximum sequence length for the tokens.
        seed (int): Random seed for shuffling the dataset.
        dataset_kwargs (dict | None): Additional keyword arguments to pass to create_sft_dataset.
            Can include 'chat', 'use_hf_tokenizer_chat_template', 'tool_schemas', etc.
        pad_seq_to_mult (int | None): Optional multiple to pad each sequence to during packing
            preparation (e.g., set to 2 * context_parallel_size for THD CP).

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

    dataset = create_sft_dataset(
        path=path,
        tokenizer=tokenizer,
        seq_length=max_seq_length,
        seed=seed,
        is_test=True,
        pad_seq_length_to_mult=pad_seq_length_to_mult,
        **dataset_kwargs,
    )

    pad_id = dataset.tokenizer.eod
    pad_seq_length_to_mult = dataset.pad_seq_length_to_mult
    max_seq_length = dataset.max_seq_length
    dataset = _retrieve_tokenized(dataset, num_tokenizer_workers)

    if pad_seq_to_mult > 1:

        def pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id):
            """
            Pad each individual data point to the length of max_length_to_pad.
            This keeps packed samples divisible by the requested multiple (used for CP/THD).
            """
            assert max_seq_length >= max_length_to_pad
            for key, val in data.items():
                if key in {"input_ids", "context_ids"}:
                    if len(val) <= max_length_to_pad:
                        # input_ids are truncated by 1 for labels; add 1 extra pad token
                        val = val + [pad_id] * (max_length_to_pad - len(val) + 1)
                    elif len(val) > max_seq_length:
                        logging.info(
                            "Sequence length %d is larger than max_seq_length %d; truncating for packing.",
                            len(val),
                            max_seq_length,
                        )
                        val = val[:max_seq_length]
                    data[key] = val
            return

        ceil_to_nearest = lambda n, m: (n + m - 1) // m * m
        for data in dataset:
            max_length_to_pad = min(max_seq_length, ceil_to_nearest(len(data["input_ids"]), pad_seq_length_to_mult))
            pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id)

    return dataset

def save_packed_dataset(packed_data: List[dict], output_path: Path, save_format: Literal["npy", "mmap"] = "npy"):
    logging.info(f'Saving packed_sequence index to {output_path} in {save_format} format...')

    """
    Saves the packed dataset to the specified output path in the given format.

    Args:
        packed_data (List[dict]): The packed dataset to be saved.
        output_path (Path): The path where the packed dataset should be saved.
        save_format (Literal["npy", "mmap"]): The file format to save the packed dataset.
            Can be either "npy" or "mmap". Defaults to "npy".

    Returns:
        None: The function saves the packed dataset to the specified output path.
    """
    if save_format == "npy":
        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            msc.numpy.save(output_path, packed_data)
        else:
            np.save(output_path, packed_data)
    elif save_format == "mmap":
        if not output_path.parent.exists():
            # the msc.numpy.memmap doesn't work properly with mode="w+"
            raise NotImplementedError("Packed sequence dataset in 'mmap' format is not supported on non-local filesystems. Please use 'npy' format or save to a local directory.")

        """Convert npy to memmap format with .idx metadata and .bin data"""
        # Compute metadata
        print('Computing metadata...')
        input_lens = np.array([len(item["input_ids"]) for item in packed_data], dtype=np.int32)
        loss_lens = np.array([len(item["loss_mask"]) for item in packed_data], dtype=np.int32)
        seq_lens = np.array([len(item["seq_start_id"]) for item in packed_data], dtype=np.int32)

        # Compute byte offsets for each sample
        bytes_per_sample = (input_lens + loss_lens + seq_lens) * 4  # int32 = 4 bytes
        offsets = np.concatenate([[0], np.cumsum(bytes_per_sample)[:-1]])

        # Save index file
        idx_file = output_path.with_suffix(output_path.suffix + '.idx.npy')
        np.save(idx_file, {
            'offsets': offsets,
            'input_lens': input_lens,
            'loss_lens': loss_lens,
            'seq_lens': seq_lens
        })

        # Write data file
        bin_file = output_path.with_suffix(output_path.suffix + '.bin')

        total_ints = int(np.sum(bytes_per_sample) // 4)
        mmap = np.memmap(bin_file, dtype=np.int32, mode='w+', shape=(total_ints,))
        pos = 0
        for item in tqdm(packed_data):
            inp = np.array(item["input_ids"], dtype=np.int32)
            loss = np.array(item["loss_mask"], dtype=np.int32)
            seq = np.array(item["seq_start_id"], dtype=np.int32)

            chunk_len = len(inp) + len(loss) + len(seq)
            mmap[pos:pos+len(inp)] = inp
            mmap[pos+len(inp):pos+len(inp)+len(loss)] = loss
            mmap[pos+len(inp)+len(loss):pos+chunk_len] = seq
            pos += chunk_len
        mmap.flush()
        del mmap
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

def prepare_packed_sequence_data(
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
    save_format: Literal["npy", "mmap"] = "npy",
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
        dataset_kwargs (dict | None): Additional keyword arguments to pass to create_sft_dataset.
            Enables packing with chat templates, tool schemas, etc.
        pad_seq_to_mult (int | None): Optional multiple to pad each sequence to during packing
            preparation (e.g., set to 2 * context_parallel_size for THD CP).

    Returns:
        None: Saves the packed sequence data to the specified output path.
    """
    logger.info(f"Preparing packed sequence from {input_path}")
    dataset = tokenize_dataset(
        input_path,
        tokenizer,
        max_seq_length,
        seed,
        dataset_kwargs,
        pad_seq_to_mult=pad_seq_to_mult,
        num_tokenizer_workers=num_tokenizer_workers
    )
    sequences, histogram = create_hist(dataset, max_seq_length)

    assignments, packing_metadata = create_packing_strategy(histogram, packed_sequence_size, packing_algorithm)
    output_data = fill_packing_strategy(assignments, sequences, packed_sequence_size, tokenizer.eos_id)

    # save output data
    save_packed_dataset(output_data, output_path, save_format)

    # save packing metadata, packing_metadata is appended to the packing file if it exists
    if output_metadata_path is not None:
        try:
            with output_metadata_path.open(mode="r") as f:
                packing_metadata_file = json.load(f)
                # 'packing_metadata_file' is expected to be a list of dicts: List[Dict[str, int]]
                # Each dict corresponds to a packed dataset. Typically there will be two dicts,
                # one each for the packed val and train datasets.
                # Each dict records two values: 'max_samples_per_bin', the max
                # number of samples per packed sequence, and 'dataset_max_seqlen', the max
                # sequence length per sample in the packed dataset.
                assert isinstance(packing_metadata_file, list), "invalid packing_metadata_file!"
        except FileNotFoundError:
            packing_metadata_file = []

        packing_metadata_file.append(packing_metadata)
        with output_metadata_path.open(mode="w") as f:
            json.dump(packing_metadata_file, f)

    logger.info(f"Packed sequence is prepared and saved to {output_path}")


@dataclass
class PackedSequenceSpecs:
    """
    Configuration class for packed sequence datasets.

    This class holds parameters related to sequence packing, including the size of the packed sequences,
    tokenizer information, paths to packed data files, and other related settings.
    """

    packed_sequence_size: int = -1
    """
    If a positive integer, this arg enables training with sequence packing and specifies the pack size
    If less than or equal to 0, sequence packing is disabled. Defaults to -1.
    Note: This arg is distinct from `seq_length` because `seq_length` specifies the maximum length
    of the original sequence (i.e. the length to truncate long sequences in the input data).
    """

    packed_sequence_format: Literal["npy", "mmap"] = "npy"
    """
    The file format for the packed sequence dataset. Can be either "npy" (default) or "mmap".
    "mmap" produces two files: .mmap.idx.npy and .mmap.bin. The `packed_train_data_path` and `packed_val_data_path` should point
    to the prefix of the .idx.npy and .bin files (i.e. without the .idx.npy or .bin suffix).
    """

    tokenizer_model_name: str = None
    """
    Keep track of tokenizer model name, since each tokenizer produces a different packed sequence dataset file.
    This field is set by llm.finetune api.
    """

    num_tokenizer_workers: int = -1
    """
    The number of worker processes to use for tokenization when preparing the packed sequence dataset.
    If -1, the number of workers will be set to the number of CPU cores available
    """

    packed_train_data_path: str = None
    """
    If specified, use this file for the packed training dataset instead of the default path.
    """

    packed_val_data_path: str = None
    """
    If specified, use this file for the packed validation dataset instead of the default path.
    """

    packed_metadata_path: str = None
    """
    If specified, use this file for the training and validation packing metadata file instead of the default path.
    """

    pad_cu_seqlens: bool = False
    """
    If True, pad cu_seqlens to a constant size, which is required for use with cudagraphs.
    """
    pad_seq_to_mult: int | None = 1
    """
    Optional multiple to pad each sample to when generating packed datasets.
    For THD/context parallel, set to (context_parallel_size * 2) to keep samples divisible.
    """

    def __post_init__(self):
        if self.packed_train_data_path is not None:
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                self.packed_train_data_path = msc.Path(self.packed_train_data_path)
            else:
                self.packed_train_data_path = Path(self.packed_train_data_path)
            assert self.packed_train_data_path.suffix == ".npy", (
                f"packed training data file must be a .npy file: {self.packed_train_data_path}"
            )
            assert self.packed_train_data_path.exists(), (
                f"packed training data file does not exist: {self.packed_train_data_path}"
            )

        if self.packed_val_data_path is not None:
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                self.packed_val_data_path = msc.Path(self.packed_val_data_path)
            else:
                self.packed_val_data_path = Path(self.packed_val_data_path)
            assert self.packed_val_data_path.suffix == ".npy", (
                f"packed validation data file must be a .npy file: {self.packed_val_data_path}"
            )
            assert self.packed_val_data_path.exists(), (
                f"packed validation data file does not exist: {self.packed_val_data_path}"
            )

        if self.pad_seq_to_mult is not None and self.pad_seq_to_mult <= 0:
            raise ValueError("pad_seq_to_mult must be a positive integer when provided.")
