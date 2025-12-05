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

import numpy as np
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.datasets.packing_utils import create_hist, create_packing_strategy, fill_packing_strategy
from megatron.bridge.data.datasets.sft import create_sft_dataset
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer


logger = logging.getLogger(__name__)


def tokenize_dataset(
    path: Path,
    tokenizer: MegatronTokenizer,
    max_seq_length: int,
    seed: int,
    dataset_kwargs: dict | None = None,
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

    Returns:
        np.ndarray: A NumPy array containing the tokenized data.
    """
    # dataset = create_sft_dataset(
    #     path=path,
    #     tokenizer=tokenizer,
    #     seq_length=max_seq_length,
    #     seed=seed,
    #     is_test=True,
    # )
    # return np.array([dataset[i] for i in range(len(dataset))])
    from megatron.bridge.data.datasets.sft import GPTSFTDataset

    pad_seq_length_to_mult = 16
    cp_size = 8 # TODO(liding): hardcoded for now for debugging

    if cp_size > 1:
        pad_seq_length_to_mult = max(pad_seq_length_to_mult, cp_size * 2)

    dataset = GPTSFTDataset(
        file_path=str(path),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        min_seq_length=1,
        pad_seq_length_to_mult=pad_seq_length_to_mult,
        sep_id=None,
        max_num_samples=None,
        seed=seed,
        label_key="output",
        answer_only_loss=True,
        truncation_field="input",
        truncation_method="right",
        prompt_template="{input} {output}",
        is_test=True,
        **dataset_kwargs,
    )

    max_seq_length = dataset.max_seq_length
    pad_id = dataset.tokenizer.eos_id
    pad_seq_length_to_mult = dataset.pad_seq_length_to_mult
    dataset = np.array([dataset[i] for i in range(len(dataset))])

    if cp_size > 1:
        def pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id):
            '''
            pad each individual data point to the length of max_length
            '''
            assert max_seq_length >= max_length_to_pad
            for key, val in data.items():
                if key in {'input_ids', 'context_ids'}:
                    if len(val) <= max_length_to_pad:
                        # because input_ids are truncated by 1 for inputs and labels,
                        # we add 1 extra padding here to make sure padded inputs and labels
                        # are is a multiple of (cp_size * 2)
                        val = val + [pad_id] * (max_length_to_pad - len(val) + 1)
                        data[key] = val
                    elif len(val) > max_seq_length:
                        logging.info(
                            f"""The current sequence length {len(val)} for packing is
                                        larger than the max_seq_length specified ({max_seq_length}).
                                        The current seqquence length is truncated to the size of max_seq_length.
                                        Please consider increase the sequence packing size"""
                        )
                        data[key] = val[:max_seq_length]
            return
        
        ceil_to_nearest = lambda n, m: (n + m - 1) // m * m
        for data in dataset:
            max_length_to_pad = min(max_seq_length, ceil_to_nearest(len(data['input_ids']), pad_seq_length_to_mult))
            pre_pad_dataset(data, max_seq_length, max_length_to_pad, pad_id)
    return dataset

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

    Returns:
        None: Saves the packed sequence data to the specified output path.
    """
    logger.info(f"Preparing packed sequence from {input_path}")
    dataset = tokenize_dataset(input_path, tokenizer, max_seq_length, seed, dataset_kwargs)
    sequences, histogram = create_hist(dataset, max_seq_length)

    assignments, packing_metadata = create_packing_strategy(histogram, packed_sequence_size, packing_algorithm)
    output_data = fill_packing_strategy(assignments, sequences, packed_sequence_size, tokenizer.eos_id)

    # save output data
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        msc.numpy.save(output_path, output_data)
    else:
        np.save(output_path, output_data)

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

    tokenizer_model_name: str = None
    """
    Keep track of tokenizer model name, since each tokenizer produces a different packed sequence dataset file.
    This field is set by llm.finetune api.
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
