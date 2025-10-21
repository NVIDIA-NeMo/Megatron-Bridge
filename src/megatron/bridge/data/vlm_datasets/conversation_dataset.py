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

"""
Core dataset types for conversation-style VLM examples.
"""

from typing import Any, Callable

import torch

from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS


class VLMConversationDataset(torch.utils.data.Dataset):
    """Repeating wrapper over a list of HF-style conversation examples.

    - Each base example is expected to contain a "conversation" key following
      processor.apply_chat_template conventions. Optional modality fields like
      "audio" are passed through and consumed by the collate function.
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
        collate_impl: Callable[[list, Any], dict[str, torch.Tensor]] | None = None,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        selected_impl = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])  # type: ignore[index]

        def _bound_collate(batch: list) -> dict[str, torch.Tensor]:
            return selected_impl(batch, self._processor)  # type: ignore[call-arg]

        self.collate_fn = _bound_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        return base
