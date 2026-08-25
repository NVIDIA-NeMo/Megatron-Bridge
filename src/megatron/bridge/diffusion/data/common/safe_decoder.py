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

"""Safe decoding for Bridge diffusion WebDataset fields."""

import io
import re
from typing import Any

import torch
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from webdataset.autodecode import basichandlers

from megatron.bridge.utils.safe_pickle import safe_torch_tensor_pickle_loads


_PICKLE_EXTENSIONS = frozenset({"pickle", "pkl", "pyd"})


def _safe_basic_handler(key: str, data: bytes) -> Any:
    """Decode tensor fields safely and delegate non-executable formats."""
    extension = re.sub(r".*[.]", "", key).lower()
    if extension in _PICKLE_EXTENSIONS:
        return safe_torch_tensor_pickle_loads(data)
    if extension == "pth":
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    return basichandlers(key, data)


class SafeDiffusionSampleDecoder(SampleDecoder):
    """Energon decoder that never uses WebDataset's unrestricted pickle handlers."""

    def __init__(self) -> None:
        super().__init__()
        replaced = False
        for index, handler in enumerate(self._decoder.handlers):
            if handler is basichandlers:
                self._decoder.handlers[index] = _safe_basic_handler
                replaced = True
        if not replaced:
            raise RuntimeError("Energon SampleDecoder no longer exposes the expected WebDataset basic handler.")
