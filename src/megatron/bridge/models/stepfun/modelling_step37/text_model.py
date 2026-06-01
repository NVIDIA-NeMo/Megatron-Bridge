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

"""Step3.7 text decoder wrapper.

Mirrors ``qwen_vl/modelling_qwen3_vl/text_model.py``: a thin subclass of
``megatron.core.models.gpt.GPTModel`` that fixes Step3.7-specific defaults
without changing the GPTModel forward signature. The actual decoder layer
type (Step-3.5's hybrid full/sliding ``Step35DecoderLayer``) is selected by
the layer spec passed in by :class:`Step37ModelProvider`.
"""

from megatron.core.models.gpt.gpt_model import GPTModel


class Step37GPTModel(GPTModel):
    """GPTModel subclass used as Step3.7's language tower.

    Currently this class only carries a Step3.7-specific name (so error /
    state-dict messages name the right model) and exists to mirror
    ``Qwen3VLGPTModel`` for structural parity. All forward behaviour is
    inherited from :class:`megatron.core.models.gpt.GPTModel`.
    """


__all__ = ["Step37GPTModel"]
