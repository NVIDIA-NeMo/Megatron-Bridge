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

import random
from typing import Any, Callable, Dict, List, Optional

import torch

from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS


class VLMConversationDataset(torch.utils.data.Dataset):
    """Repeating wrapper over a list of HF-style conversation examples.

    - Each base example is expected to contain a "conversation" key following
      processor.apply_chat_template conventions. Optional modality fields like
      "audio" are passed through and consumed by the collate function.
    - Dataset length is set to a target length and indexes wrap around the
      underlying list to meet the requested size.
    - Examples are shuffled on construction to ensure diverse batches when
      used with sequential samplers (e.g. MegatronPretrainingSampler).
    - A `collate_fn` attribute is exposed so the framework can pass it to the
      DataLoader.
    """

    def __init__(
        self,
        base_examples: List[Dict[str, Any]],
        target_length: int,
        processor: Any,
        collate_impl: Optional[Callable[[list, Any], Dict[str, torch.Tensor]]] = None,
        shuffle: bool = True,
        seed: int = 42,
        pack_sequences: bool = False,
    ) -> None:
        assert isinstance(base_examples, list) and len(base_examples) > 0, "base_examples must be a non-empty list"
        if shuffle:
            base_examples = list(base_examples)
            random.Random(seed).shuffle(base_examples)
        self._base_examples = base_examples
        self._length = int(max(0, target_length))
        self._processor = processor
        # Choose collate implementation by processor type name when not provided
        collate_key = type(processor).__name__ if processor is not None else "default"
        selected_impl = collate_impl or COLLATE_FNS.get(collate_key, COLLATE_FNS["default"])  # type: ignore[index]

        # If packing requested, only collates that advertise `pack_sequences` support it;
        # bind via functools.partial so the DataLoader just calls f(batch, processor).
        if pack_sequences:
            import inspect
            from functools import partial

            sig = inspect.signature(selected_impl)
            if "pack_sequences" in sig.parameters:
                selected_impl = partial(selected_impl, pack_sequences=True)
            else:
                raise ValueError(
                    f"Collate function {getattr(selected_impl, '__name__', selected_impl)} "
                    f"does not accept pack_sequences=True. Use a collate that supports packing "
                    f"(e.g. nemotron_omni_collate_fn)."
                )

        def _bound_collate(batch: list) -> Dict[str, torch.Tensor]:
            return selected_impl(batch, self._processor)  # type: ignore[call-arg]

        self.collate_fn = _bound_collate

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._length == 0:
            raise IndexError("Empty dataset")
        base = self._base_examples[idx % len(self._base_examples)]
        return base
