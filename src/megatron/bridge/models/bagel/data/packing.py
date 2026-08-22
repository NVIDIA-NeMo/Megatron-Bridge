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

import logging
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Self

import numpy as np
import torch

from megatron.bridge.models.bagel.data.energon import BagelSample


logger = logging.getLogger(__name__)


def _patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchify an image with BAGEL's channel-last patch ordering."""
    channels, height, width = image.shape
    image = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
    return torch.einsum("chpwq->hwpqc", image).reshape(-1, patch_size**2 * channels)


def _position_ids(height: int, width: int, patch_size: int, max_patches_per_side: int) -> torch.Tensor:
    """Build BAGEL's extrapolated flattened image position IDs."""
    rows = torch.arange(height // patch_size)
    columns = torch.arange(width // patch_size)
    return (rows[:, None] * max_patches_per_side + columns).flatten()


def _attention_mask(split_lens: list[int], attn_modes: list[str]) -> torch.Tensor:
    """Build one official-style nested attention mask."""
    sample_len = sum(split_lens)
    mask = torch.zeros((sample_len, sample_len), dtype=torch.bool)
    offset = 0
    for split_len, mode in zip(split_lens, attn_modes):
        block = torch.ones((split_len, split_len))
        mask[offset : offset + split_len, offset : offset + split_len] = block.tril() if mode == "causal" else block
        mask[offset : offset + split_len, :offset] = 1
        offset += split_len
    offset = 0
    for split_len, mode in zip(split_lens, attn_modes):
        if mode == "noise":
            mask[:, offset : offset + split_len] = 0
            mask[offset : offset + split_len, offset : offset + split_len] = 1
        offset += split_len
    return torch.zeros_like(mask, dtype=torch.float).masked_fill_(~mask, float("-inf"))


class BagelPacker:
    """Pack cooked Energon samples in BAGEL's official streaming order."""

    def __init__(
        self,
        group_iters: Sequence[Iterator[BagelSample]],
        group_weights: Sequence[float],
        is_mandatory: Sequence[bool],
        special_tokens: Mapping[str, int],
        *,
        expected_num_tokens: int = 32768,
        max_num_tokens_per_sample: int = 16384,
        max_num_tokens: int = 36864,
        prefer_buffer_before: int = 16384,
        max_buffer_size: int = 50,
        text_cond_dropout_prob: float = 0.1,
        vit_cond_dropout_prob: float = 0.4,
        vae_cond_dropout_prob: float = 0.1,
        vae_image_downsample: int = 16,
        max_latent_size: int = 32,
        vit_patch_size: int = 14,
        max_num_patch_per_side: int = 70,
    ) -> None:
        """Configure the official non-Flex BAGEL packing path."""
        if not (len(group_iters) == len(group_weights) == len(is_mandatory)):
            raise ValueError("BAGEL group iterators, weights, and mandatory flags must align")
        total_weight = sum(group_weights)
        if total_weight <= 0:
            raise ValueError("BAGEL group weights must have a positive sum")

        self.group_iters = group_iters
        self.group_cumprobs = [sum(group_weights[: index + 1]) / total_weight for index in range(len(group_weights))]
        self.is_mandatory = is_mandatory
        self.bos_token_id = special_tokens["bos_token_id"]
        self.eos_token_id = special_tokens["eos_token_id"]
        self.start_of_image = special_tokens["start_of_image"]
        self.end_of_image = special_tokens["end_of_image"]
        self.expected_num_tokens = expected_num_tokens
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.max_num_tokens = max_num_tokens
        self.prefer_buffer_before = prefer_buffer_before
        self.max_buffer_size = max_buffer_size
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vae_image_downsample = vae_image_downsample
        self.max_latent_size = max_latent_size
        self.vit_patch_size = vit_patch_size
        self.max_num_patch_per_side = max_num_patch_per_side
        self._status = self._new_status()
        self._source_ids: list[dict[str, object]] = []
        self._buffer: list[BagelSample] = []

    @staticmethod
    def _new_status() -> dict[str, Any]:
        """Create one empty official packing accumulator."""
        return {
            "curr": 0,
            "sample_lens": [],
            "packed_position_ids": [],
            "nested_attention_masks": [],
            "packed_text_ids": [],
            "packed_text_indexes": [],
            "packed_label_ids": [],
            "ce_loss_indexes": [],
            "ce_loss_weights": [],
            "vae_image_tensors": [],
            "packed_latent_position_ids": [],
            "vae_latent_shapes": [],
            "packed_vae_token_indexes": [],
            "packed_timesteps": [],
            "mse_loss_indexes": [],
            "packed_vit_tokens": [],
            "vit_token_seqlens": [],
            "packed_vit_position_ids": [],
            "packed_vit_token_indexes": [],
        }

    @staticmethod
    def _source_id(sample: BagelSample) -> dict[str, object]:
        """Return the canonical source coordinate carried by a cooked sample."""
        return {"dataset_name": sample.metadata["dataset_group"], "source": sample.metadata["source"]}

    def __iter__(self) -> Self:
        """Return this stateful packed-batch iterator."""
        return self

    def __next__(self) -> dict[str, object]:
        """Return the next batch with official mandatory and FIFO-buffer behavior."""
        while True:
            if self._status["curr"] == 0:
                for group_index, group_iter in enumerate(self.group_iters):
                    if self.is_mandatory[group_index]:
                        while True:
                            sample = next(group_iter)
                            num_tokens = sample.num_tokens + 2 * len(sample.sequence_plan)
                            if num_tokens < self.max_num_tokens_per_sample:
                                self._pack_sequence(sample, self._status)
                                self._source_ids.append(self._source_id(sample))
                                break
                            logger.warning("Skipping BAGEL mandatory sample with length %d", num_tokens)

            if self._status["curr"] < self.prefer_buffer_before and self._buffer:
                sample = self._buffer.pop(0)
                sample_from_buffer = True
            else:
                draw = random.random()
                group_index = 0
                for index, probability in enumerate(self.group_cumprobs):
                    if draw < probability:
                        group_index = index
                        break
                sample = next(self.group_iters[group_index])
                sample_from_buffer = False

            num_tokens = sample.num_tokens + 2 * len(sample.sequence_plan)
            if num_tokens > self.max_num_tokens_per_sample:
                logger.warning("Skipping BAGEL sample with length %d", num_tokens)
                continue

            if self._status["curr"] + num_tokens > self.max_num_tokens:
                if len(self._buffer) < self.max_buffer_size and not sample_from_buffer:
                    self._buffer.append(sample)
                else:
                    batch = self._to_tensor(self._status)
                    batch["source_ids"] = self._source_ids
                    self._status = self._new_status()
                    self._source_ids = []
                    return batch
                continue

            self._pack_sequence(sample, self._status)
            self._source_ids.append(self._source_id(sample))
            if self._status["curr"] >= self.expected_num_tokens:
                batch = self._to_tensor(self._status)
                batch["source_ids"] = self._source_ids
                self._status = self._new_status()
                self._source_ids = []
                return batch

    def state_dict(self) -> dict[str, object]:
        """Capture packing, buffering, and process RNG state at a batch boundary."""
        return {
            "status": {key: list(value) if isinstance(value, list) else value for key, value in self._status.items()},
            "source_ids": list(self._source_ids),
            "buffer": list(self._buffer),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore packing, buffering, and process RNG state."""
        status = state["status"]
        if not isinstance(status, dict):
            raise TypeError("BAGEL packer status must be a dictionary")
        self._status = {key: list(value) if isinstance(value, list) else value for key, value in status.items()}
        self._source_ids = list(state["source_ids"])
        self._buffer = list(state["buffer"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])

    def _pack_sequence(self, sample: BagelSample, status: dict[str, Any]) -> None:
        """Append one cooked sample using BAGEL's dropout and timestep calls."""
        image_tensors = list(sample.image_tensor_list)
        text_ids_list = list(sample.text_ids_list)
        split_lens: list[int] = []
        attn_modes: list[str] = []
        curr = status["curr"]
        curr_rope_id = 0
        sample_len = 0
        timestep: float

        for item in sample.sequence_plan:
            split_start = item.get("split_start", True)
            if split_start:
                curr_split_len = 0

            if item["type"] == "text":
                text_ids = text_ids_list.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.text_cond_dropout_prob:
                    continue
                shifted_ids = [self.bos_token_id, *text_ids]
                status["packed_text_ids"].extend(shifted_ids)
                status["packed_text_indexes"].extend(range(curr, curr + len(shifted_ids)))
                if item["loss"] == 1:
                    status["ce_loss_indexes"].extend(range(curr, curr + len(shifted_ids)))
                    status["ce_loss_weights"].extend([1 / (len(shifted_ids) ** 0.5)] * len(shifted_ids))
                    status["packed_label_ids"].extend([*text_ids, self.eos_token_id])
                curr += len(shifted_ids)
                curr_split_len += len(shifted_ids)
                status["packed_text_ids"].append(self.eos_token_id)
                status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    status["ce_loss_indexes"].append(curr)
                    status["ce_loss_weights"].append(1.0)
                    status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1
                attn_modes.append("causal")
                status["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
                curr_rope_id += curr_split_len

            elif item["type"] == "vit_image":
                image = image_tensors.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.vit_cond_dropout_prob:
                    curr_rope_id += 1
                    continue
                status["packed_text_ids"].append(self.start_of_image)
                status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1
                vit_tokens = _patchify(image, self.vit_patch_size)
                num_image_tokens = vit_tokens.shape[0]
                status["packed_vit_token_indexes"].extend(range(curr, curr + num_image_tokens))
                curr += num_image_tokens
                curr_split_len += num_image_tokens
                status["packed_vit_tokens"].append(vit_tokens)
                status["vit_token_seqlens"].append(num_image_tokens)
                status["packed_vit_position_ids"].append(
                    _position_ids(image.size(1), image.size(2), self.vit_patch_size, self.max_num_patch_per_side)
                )
                status["packed_text_ids"].append(self.end_of_image)
                status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    status["ce_loss_indexes"].append(curr)
                    status["ce_loss_weights"].append(1.0)
                    status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1
                attn_modes.append("full")
                status["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
                curr_rope_id += 1

            elif item["type"] == "vae_image":
                image = image_tensors.pop(0)
                if item["enable_cfg"] == 1 and random.random() < self.vae_cond_dropout_prob:
                    curr_rope_id += 1
                    continue
                status["packed_text_ids"].append(self.start_of_image)
                status["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1
                status["vae_image_tensors"].append(image)
                status["packed_latent_position_ids"].append(
                    _position_ids(image.size(1), image.size(2), self.vae_image_downsample, self.max_latent_size)
                )
                height, width = image.shape[1:]
                latent_height = height // self.vae_image_downsample
                latent_width = width // self.vae_image_downsample
                status["vae_latent_shapes"].append((latent_height, latent_width))
                num_image_tokens = latent_width * latent_height
                status["packed_vae_token_indexes"].extend(range(curr, curr + num_image_tokens))
                if item["loss"] == 1:
                    status["mse_loss_indexes"].extend(range(curr, curr + num_image_tokens))
                    if split_start:
                        timestep = np.random.randn()
                else:
                    timestep = float("-inf")
                status["packed_timesteps"].extend([timestep] * num_image_tokens)
                curr += num_image_tokens
                curr_split_len += num_image_tokens
                status["packed_text_ids"].append(self.end_of_image)
                status["packed_text_indexes"].append(curr)
                if item["special_token_loss"] == 1:
                    status["ce_loss_indexes"].append(curr)
                    status["ce_loss_weights"].append(1.0)
                    status["packed_label_ids"].append(item["special_token_label"])
                curr += 1
                curr_split_len += 1
                if split_start:
                    attn_modes.append("noise" if item["loss"] == 1 and "frame_delta" not in item else "full")
                status["packed_position_ids"].extend([curr_rope_id] * (num_image_tokens + 2))
                if "frame_delta" in item:
                    curr_rope_id += item["frame_delta"]
                elif item["loss"] == 0:
                    curr_rope_id += 1

            if item.get("split_end", True):
                split_lens.append(curr_split_len)
                sample_len += curr_split_len

        status["curr"] = curr
        status["sample_lens"].append(sample_len)
        status["nested_attention_masks"].append(_attention_mask(split_lens, attn_modes))

    @staticmethod
    def _to_tensor(status: dict[str, Any]) -> dict[str, object]:
        """Convert one completed accumulator to official packed tensors."""
        batch: dict[str, object] = {
            "sequence_length": sum(status["sample_lens"]),
            "sample_lens": status["sample_lens"],
            "packed_text_ids": torch.tensor(status["packed_text_ids"]),
            "packed_text_indexes": torch.tensor(status["packed_text_indexes"]),
            "packed_position_ids": torch.tensor(status["packed_position_ids"]),
            "nested_attention_masks": status["nested_attention_masks"],
        }
        if status["vae_image_tensors"]:
            images = status["vae_image_tensors"]
            max_shape = [max(dimension) for dimension in zip(*(image.shape for image in images))]
            padded_images = torch.zeros((len(images), *max_shape))
            for index, image in enumerate(images):
                padded_images[index, :, : image.shape[1], : image.shape[2]] = image
            batch["padded_images"] = padded_images
            batch["patchified_vae_latent_shapes"] = status["vae_latent_shapes"]
            batch["packed_latent_position_ids"] = torch.cat(status["packed_latent_position_ids"], dim=0)
            batch["packed_vae_token_indexes"] = torch.tensor(status["packed_vae_token_indexes"])
        if status["packed_vit_tokens"]:
            batch["packed_vit_tokens"] = torch.cat(status["packed_vit_tokens"], dim=0)
            batch["packed_vit_position_ids"] = torch.cat(status["packed_vit_position_ids"], dim=0)
            batch["packed_vit_token_indexes"] = torch.tensor(status["packed_vit_token_indexes"])
            batch["vit_token_seqlens"] = torch.tensor(status["vit_token_seqlens"])
        if status["packed_timesteps"]:
            batch["packed_timesteps"] = torch.tensor(status["packed_timesteps"])
            batch["mse_loss_indexes"] = torch.tensor(status["mse_loss_indexes"])
        if status["packed_label_ids"]:
            batch["packed_label_ids"] = torch.tensor(status["packed_label_ids"])
            batch["ce_loss_indexes"] = torch.tensor(status["ce_loss_indexes"])
            batch["ce_loss_weights"] = torch.tensor(status["ce_loss_weights"])
        return batch
