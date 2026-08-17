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

"""Runtime dataset for direct SFT examples."""

import inspect
from collections.abc import Callable
from typing import Any

import torch


def _collate_kwargs_for_impl(
    collate_impl: Callable[..., dict[str, torch.Tensor]],
    collate_kwargs: dict[str, Any],
    *,
    require_packing_support: bool,
) -> dict[str, Any]:
    try:
        parameters = inspect.signature(collate_impl).parameters
    except (TypeError, ValueError):
        return collate_kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return collate_kwargs

    supported_kwargs = {key: value for key, value in collate_kwargs.items() if key in parameters}
    if require_packing_support and "enable_in_batch_packing" not in supported_kwargs:
        raise ValueError(
            f"Collate function {getattr(collate_impl, '__name__', collate_impl)} "
            "does not accept enable_in_batch_packing=True. Use a collate that supports in-batch packing."
        )
    return supported_kwargs


class DirectSFTDataset(torch.utils.data.Dataset):
    """Repeating wrapper over normalized SFT examples.

    - Examples may use structured conversations or paired prompt-completion
      text, as selected by the owning dataset config. Optional modality fields
      are passed through and consumed by the collate function.
    - Dataset length is set to a target length and indexes wrap around the
      underlying list to meet the requested size.
    - A `collate_fn` attribute is exposed so the framework can pass it to the
      DataLoader.
    """

    def __init__(
        self,
        base_examples: list[dict[str, Any]],
        target_length: int,
        processor: Any,
        collate_impl: Callable[..., dict[str, torch.Tensor]] | None = None,
        sequence_length: int | None = None,
        pad_to_max_length: bool = False,
        pad_to_multiple_of: int = 128,
        enable_in_batch_packing: bool = False,
        defer_in_batch_packing_to_step: bool = False,
        in_batch_packing_pad_to_multiple_of: int = 1,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        explicit_collate_impl = collate_impl is not None
        if collate_impl is None:
            from megatron.bridge.data.collators.registry import resolve_model_collate

            collate_impl = resolve_model_collate(collate_key)
        assert collate_impl is not None

        collate_kwargs: dict[str, Any] = {
            "sequence_length": sequence_length,
            "pad_to_max_length": pad_to_max_length,
            "pad_to_multiple_of": pad_to_multiple_of,
            # Active deferral user: Qwen3-VL. Other VLM/HF collates should pack
            # here when enable_in_batch_packing is set.
            "enable_in_batch_packing": enable_in_batch_packing and not defer_in_batch_packing_to_step,
            "in_batch_packing_pad_to_multiple_of": in_batch_packing_pad_to_multiple_of,
        }
        if explicit_collate_impl:
            collate_kwargs = _collate_kwargs_for_impl(
                collate_impl,
                collate_kwargs,
                require_packing_support=bool(collate_kwargs["enable_in_batch_packing"]),
            )

        def _bound_collate(batch: list) -> dict[str, torch.Tensor]:
            return collate_impl(batch, self._processor, **collate_kwargs)

        self.collate_fn = _bound_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        return base
