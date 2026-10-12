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

"""GLM5 providers.

``GLM5ModelProvider`` remains an alias of ``MLAModelProvider`` for import
compatibility. ``GLM5DSANativeFp8Provider`` additionally routes the DSA Indexer
scoring path through the native SM90 FP8 cuDNN scorer when it builds a GLM-5.2
model on an SM90 device.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from megatron.bridge.models.mla_provider import MLAModelProvider


logger = logging.getLogger(__name__)

# Import-compatibility alias retained from the original module.
GLM5ModelProvider = MLAModelProvider


@dataclass
class GLM5DSANativeFp8Provider(MLAModelProvider):
    """MLA provider that auto-enables native SM90 FP8 DSA Indexer scoring.

    On a GLM-5.2 configuration running on an SM90 device, ``provide`` routes the
    DSA Indexer scoring path through the native cuDNN FP8 scorer so the Indexer
    top-k matches the vLLM GLM-5.2 reference. It is a no-op for other GLM-5
    variants, non-SM90 devices, and incompatible DSA configurations.
    """

    # HF model id/path, used to select the GLM-5.2 adapter and exclude the
    # GLM-5/5.1/5.3 siblings that share the DSA fingerprint.
    glm_dsa_native_fp8_hf_model_id: Optional[str] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build the Megatron model and enable native FP8 DSA where supported."""
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Lazy import: the adapter pulls in heavy Megatron-Core DSA modules that
        # should not be imported merely to construct a provider instance.
        from megatron.bridge.models.glm_moe_dsa.glm_dsa_native_fp8 import maybe_enable_glm52_dsa_native_fp8

        patched = maybe_enable_glm52_dsa_native_fp8(
            model,
            hf_model_id=self.glm_dsa_native_fp8_hf_model_id or "",
            model_config=self,
        )
        if patched:
            logger.info("Enabled GLM-5.2 native SM90 FP8 cuDNN DSA on %d local Indexer modules", patched)
        return model


__all__ = ["GLM5ModelProvider", "GLM5DSANativeFp8Provider"]
