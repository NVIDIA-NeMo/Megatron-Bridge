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

"""Muse Glimmer model, builder, configuration, and bridge exports."""

import logging

import transformers

from megatron.bridge.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel
from megatron.bridge.models.muse_glimmer.muse_glimmer_bridge import MuseGlimmerBridge
from megatron.bridge.models.muse_glimmer.muse_glimmer_builder import MuseGlimmerModelBuilder
from megatron.bridge.models.muse_glimmer.muse_glimmer_config import (
    MuseGlimmerModelConfig,
    MuseGlimmerTransformerConfig,
    MuseGlimmerVisionModelConfig,
)


logger = logging.getLogger(__name__)


__all__ = [
    "MuseGlimmerBridge",
    "MuseGlimmerModel",
    "MuseGlimmerModelBuilder",
    "MuseGlimmerModelConfig",
    "MuseGlimmerTransformerConfig",
    "MuseGlimmerVisionModelConfig",
]

try:
    from transformers import (
        MuseGlimmerConfig,
        MuseGlimmerTextConfig,
        MuseGlimmerVisionConfig,
    )
except ImportError:
    logger.warning(
        "Muse Glimmer Hugging Face classes are unavailable with transformers %s. "
        "Upgrade transformers to a version that supports Muse Glimmer.",
        transformers.__version__,
    )
else:
    __all__.extend(
        [
            "MuseGlimmerConfig",
            "MuseGlimmerTextConfig",
            "MuseGlimmerVisionConfig",
        ]
    )
