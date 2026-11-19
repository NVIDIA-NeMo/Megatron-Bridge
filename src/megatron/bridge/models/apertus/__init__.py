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

from megatron.bridge.models.apertus.apertus_bridge import ApertusBridge


# Apertus1p5ForConditionalGeneration is on transformers main (huggingface/transformers#47662)
# but not yet in a release. Without it the Apertus 1.5 bridge is not registered, so importing
# megatron.bridge.models keeps working for every other model.
try:
    from transformers import Apertus1p5ForConditionalGeneration  # noqa: F401
except ImportError:
    APERTUS1P5_AVAILABLE = False
else:
    APERTUS1P5_AVAILABLE = True

__all__ = ["APERTUS1P5_AVAILABLE", "ApertusBridge"]

if APERTUS1P5_AVAILABLE:
    from megatron.bridge.models.apertus.apertus1p5_bridge import Apertus1p5Bridge

    __all__.append("Apertus1p5Bridge")
