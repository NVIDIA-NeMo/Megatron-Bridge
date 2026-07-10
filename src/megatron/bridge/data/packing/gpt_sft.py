# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Runtime dataset for legacy NumPy GPT SFT packed artifacts."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from megatron.core.msc_utils import MultiStorageClientFeature

from megatron.bridge.data.datasets.gpt_sft import GPTSFTDataset
from megatron.bridge.training.tokenizers.tokenizer import MegatronTokenizer
from megatron.bridge.utils.safe_pickle import safe_load_npy


logger = logging.getLogger(__name__)


def _safe_load_packed_npy(file_path: str | Path) -> np.ndarray:
    """Load a packed NumPy artifact with the restricted object unpickler."""
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        with msc.open(str(file_path), "rb") as input_file:
            data = input_file.read()
    else:
        with open(file_path, "rb") as input_file:
            data = input_file.read()
    return safe_load_npy(data)


class GPTSFTPackedDataset(GPTSFTDataset):
    """ """

    def __init__(
        self,
        file_path: str,
        tokenizer: MegatronTokenizer,
        return_cu_seqlen: bool = True,
        pad_cu_seqlens: bool = False,
        pad_seq_to_mult: int = 1,
        pack_metadata_file_path: str | None = None,
        **kwargs,
    ):
        """
        file_path: See `file_path` in the parent class.
        tokenizer: See `tokenizer` in the parent class.
        return_cu_seqlen: Whether to return `cu_seqlen` to pass to the model. Having `cu_seqlen` in the model input
                enables THD attention kernel, which is the correct format for training with packed sequence to prevent
                cross-sequence attention. This flag should be True unless you have a specific use case.
        pad_seq_to_mult: The multiple used for padding sequences during packing. When > 1, cu_seqlens_unpadded
                will be computed to support THD CP. When == 1 (no padding), cu_seqlens_unpadded is not computed.
        """
        np.random.seed(kwargs.get("seed", 1234))
        super().__init__(file_path, tokenizer, **kwargs)
        assert self.virtual_tokens == 0, "P-Tuning with packed sequence is not supported."
        self.return_cu_seqlen = return_cu_seqlen
        self._pad_seq_to_mult = pad_seq_to_mult

        self.pad_cu_seqlens = pad_cu_seqlens
        if self.pad_cu_seqlens:
            assert pack_metadata_file_path is not None, (
                "a metadata json file is required when pad_cu_seqlens is enabled"
            )
            assert self.pad_to_max_length is True, (
                "'pad_to_max_length=True' is required when pad_cu_seqlens is enabled"
            )

        self.pack_metadata = None
        if pack_metadata_file_path is not None:
            if MultiStorageClientFeature.is_enabled():
                msc = MultiStorageClientFeature.import_package()
                with msc.open(str(pack_metadata_file_path), "r") as f:
                    self.pack_metadata = json.load(f)
            else:
                with open(pack_metadata_file_path) as f:
                    self.pack_metadata = json.load(f)

    def __getitem__(self, idx):
        is_padding = idx < 0
        if self.samples_mapping is not None:
            # assert idx < len(self.samples_mapping)
            idx = self.samples_mapping[idx]

        input_ids = self.indexed_dataset[idx]["input_ids"]
        seq_boundaries = self.indexed_dataset[idx]["seq_start_id"] + [len(input_ids)]
        loss_mask = self.indexed_dataset[idx]["loss_mask"]
        if is_padding or idx < 0:
            loss_mask = [0] * len(loss_mask)
        return {"input_ids": input_ids, "seq_boundaries": seq_boundaries, "loss_mask": loss_mask}

    def _load_dataset(self):
        try:
            self.indexed_dataset = _safe_load_packed_npy(self.file_path)
        except Exception as e:
            logger.error(
                f"Failed to load packed dataset. The dataset should be a `.npy` file. "
                f"Please check if the packed dataset was prepared correctly. The original error was:\n {e}",
            )
            exit(1)

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            # custom samples mapping logic, following the format for unpacked sft dataset
            # Note: this is epoch-level shuffling, i.e. sampling without replacement until end of epoch, then repeat.
            # Unpacked dataset shuffles by sampling with replacement indefinitely.
            dataset_len = len(self.indexed_dataset)
            max_num_epochs = np.ceil(self.max_num_samples / dataset_len)
            indices = np.arange(dataset_len)[None, :].repeat(max_num_epochs, axis=0)
            [np.random.shuffle(x) for x in indices]
            self.samples_mapping = indices.reshape(1, -1).squeeze()[: self.max_num_samples]
        else:
            self.samples_mapping = None

    def _build_loss_mask(self, processed_example):
        seq_boundaries = processed_example["seq_boundaries"]
        if self.answer_only_loss:
            return np.concatenate(
                [
                    processed_example["loss_mask"][seq_boundaries[i] : seq_boundaries[i + 1] - 1]
                    for i in range(len(seq_boundaries) - 1)
                ]
            )
        return np.concatenate(
            [
                [
                    0 if x == self.tokenizer.eos_id else 1.0
                    for x in processed_example["input_ids"][seq_boundaries[i] : seq_boundaries[i + 1] - 1]
                ]
                for i in range(len(seq_boundaries) - 1)
            ]
        )

    def _maybe_cast_to_list(self, x):
        return [item.tolist() if isinstance(item, np.ndarray) else item for item in x]

    def collate_fn(self, batch):
        """
        Collates a list of packed sequence samples into a batch for the model.

        This method is specifically designed for `GPTSFTPackedDataset`. It takes a list
        of packed sequence items (as returned by `__getitem__`) and prepares a batch
        of tensors. This includes handling `cu_seqlens` which are crucial for the
        efficient processing of packed sequences with kernels like THD attention.

        Args:
            batch (List[dict]): A list of packed sequence samples.

        Returns:
            dict: A dictionary of batched tensors, including 'tokens', 'labels',
                  'loss_mask', 'position_ids', and potentially 'cu_seqlens',
                  'cu_seqlens_argmin', 'max_seqlen' if `return_cu_seqlen` is True.
        """
        input_ids = [
            np.concatenate(
                [
                    item["input_ids"][item["seq_boundaries"][i] : item["seq_boundaries"][i + 1] - 1]
                    for i in range(len(item["seq_boundaries"]) - 1)
                ]
            )
            for item in batch
        ]
        labels = [
            np.concatenate(
                [
                    item["input_ids"][item["seq_boundaries"][i] + 1 : item["seq_boundaries"][i + 1]]
                    for i in range(len(item["seq_boundaries"]) - 1)
                ]
            )
            for item in batch
        ]

        loss_mask = [self._build_loss_mask(item) for item in batch]

        token_count = [item.shape[0] for item in input_ids]

        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            # pad to the nearest multiple of 16 for FP8 training
            # for many datasets in practice, all packed sequence lengths are very close to the
            # target length (2048, 4096, 8192), so there is very minimal padding
            max_length = max(len(length) for length in input_ids)
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, self.pad_seq_length_to_mult))
        assert max_length <= self.max_seq_length

        position_ids: list[list[int]] = []
        cu_seqlens: list[list[int]] = []
        # Only compute cu_seqlens_unpadded when pad_seq_to_mult > 1 (actual padding for CP)
        cu_seqlens_unpadded: list[list[int]] | None = [] if self._pad_seq_to_mult > 1 else None
        for item in batch:
            position_ids.append([])
            cu_seqlens.append([0])
            if cu_seqlens_unpadded is not None:
                cu_seqlens_unpadded.append([0])
            seqlens = np.array(item["seq_boundaries"][1:]) - np.array(item["seq_boundaries"][:-1])
            for length in seqlens:
                # length minus 1 because input_ids is truncated by 1 for labels
                position_ids[-1].extend(list(range(length - 1)))
                cu_seqlens[-1].append(cu_seqlens[-1][-1] + length - 1)

            # the last seq needs to be the max seq len because rope and attn kernels expect no padding
            assert cu_seqlens[-1][-1] <= max_length

            # since data is prepadded when cp_size > 1, there may be some extra padding at the end
            # of the packed sequence. In this case, we need to add the max seq len to the end.
            if cu_seqlens[-1][-1] != max_length:
                cu_seqlens[-1].append(max_length)

            if cu_seqlens_unpadded is not None:
                for i in range(len(item["seq_boundaries"]) - 1):
                    current_seq = item["input_ids"][item["seq_boundaries"][i] : item["seq_boundaries"][i + 1] - 1]

                    # Stop unpadded lengths at the last non-eos token so padding eos are excluded.
                    current_seq_arr = np.array(current_seq)
                    non_eos_positions = np.where(current_seq_arr != self.tokenizer.eos_id)[0]
                    seqlen_unpadded = non_eos_positions[-1] + 1 if non_eos_positions.size > 0 else 0
                    cu_seqlens_unpadded[-1].append(cu_seqlens_unpadded[-1][-1] + seqlen_unpadded)

                # if extra paddings are added in the packed sequence, they can't be counted as
                # actual tokens for training
                if len(cu_seqlens[-1]) > len(cu_seqlens_unpadded[-1]):
                    cu_seqlens_unpadded[-1].append(cu_seqlens_unpadded[-1][-1])

            if self.pad_cu_seqlens:
                # pad cu_seqlens to a constant shape with zero length sequences
                max_samples_per_bin = max(p["max_samples_per_bin"] for p in self.pack_metadata)
                # plus 2 since cu_seqlens additionally contains 0 and may append max_length
                pad_num = max_samples_per_bin - len(cu_seqlens[-1]) + 2
                cu_seqlens[-1].extend([max_length] * pad_num)

        assert len(input_ids[0]) == len(position_ids[0]), (
            "Dataset problem: input_ids and position_ids lengths don't match"
        )

        input_ids = self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        labels = self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id)
        loss_mask = self._collate_item(loss_mask, max_length=max_length, pad_id=0)
        position_ids = self._collate_item(position_ids, max_length=max_length, pad_id=0)

        tokens = torch.LongTensor(input_ids)
        loss_mask = torch.LongTensor(loss_mask)
        # drop any padding/eos tokens from contributing to the loss
        loss_mask[tokens == self.tokenizer.eos_id] = 0

        processed_batch = {
            "tokens": tokens,
            "labels": torch.LongTensor(labels),
            "loss_mask": loss_mask,
            "position_ids": torch.LongTensor(position_ids),
            "token_count": token_count,
        }

        if self.return_cu_seqlen:
            cu_seqlens = self._collate_item(
                cu_seqlens, max_length=max(len(length) for length in cu_seqlens) + 1, pad_id=-1
            )
            # Pre-generate `cu_seqlens_argmin` and `max_seqlen` as CPU tensor to avoid device-to-host copies.
            cu_seqlens = torch.IntTensor(cu_seqlens)
            cu_seqlens_argmin = torch.argmin(cu_seqlens, dim=1, keepdim=True)
            seqlens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
            max_seqlen, _ = seqlens.max(dim=1, keepdim=True)

            if self.pad_cu_seqlens:
                # If padding, use the global max seqlen, so that 'pad_cu_seqlens' is the same
                # across all batches. This is maintly used compatiblity with megatron's implementation
                # of cudagraphs, which uses the same cudagraphs over all batches.
                dataset_max_seqlen = max(p["dataset_max_seqlen"] for p in self.pack_metadata)
                min_pack_seq_len = min(p["min_packed_seqlen"] for p in self.pack_metadata)
                padding_gap = max_length - min_pack_seq_len

                # Use the larger of the two values to avoid NaN issues with attention kernel
                safe_max_seqlen = max(dataset_max_seqlen, padding_gap)
                max_seqlen = torch.IntTensor([safe_max_seqlen] * len(cu_seqlens))
            else:
                seqlens = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
                max_seqlen, _ = seqlens.max(dim=1, keepdim=True)

            cu_seqlens_batch = {
                "attention_mask": None,  # no attention mask is needed for packed seq
                "cu_seqlens": torch.IntTensor(cu_seqlens),  # cu_seqlens_q must be in dtype torch.int32
                "cu_seqlens_argmin": cu_seqlens_argmin,  # only required for perf
                "max_seqlen": max_seqlen,  # only required for perf
            }

            # Only include cu_seqlens_unpadded when pad_seq_to_mult > 1 (actual CP padding)
            if cu_seqlens_unpadded is not None:
                cu_seqlens_unpadded = self._collate_item(
                    cu_seqlens_unpadded, max_length=max(len(length) for length in cu_seqlens_unpadded) + 1, pad_id=-1
                )
                cu_seqlens_unpadded = torch.IntTensor(cu_seqlens_unpadded)
                cu_seqlens_unpadded_argmin = torch.argmin(cu_seqlens_unpadded, dim=1, keepdim=True)
                cu_seqlens_batch["cu_seqlens_unpadded"] = cu_seqlens_unpadded
                cu_seqlens_batch["cu_seqlens_unpadded_argmin"] = cu_seqlens_unpadded_argmin

            processed_batch.update(cu_seqlens_batch)
        else:
            attention_mask = [self._create_attention_mask(max_length) for _ in batch]
            processed_batch.update(
                {
                    "attention_mask": torch.stack(attention_mask),
                }
            )

        return processed_batch
