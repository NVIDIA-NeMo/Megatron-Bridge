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

"""Step3.7 HF ``PretrainedConfig`` surrogate.

Mirrors ``configuration_step35.py``. The real ``Step37Config`` ships with the
upstream checkpoint at
``stepfun-ai/step3p7_flash_bf16/configuration_step3p7.py`` and is loaded via
``trust_remote_code=True`` at inference time. This file exists so the
Megatron-Bridge package can be self-describing — ``Step37Config`` /
``Step37TextConfig`` / ``Step37VisionConfig`` here surface the same fields
the bridge reads in ``Step37Bridge.provider_bridge``, without requiring the
remote-code shim to be on ``sys.path``.

When the upstream config ships on HF, ``Step37Bridge`` can be retargeted at
the upstream class; until then the Auto* classes pick the right config via
``auto_map`` in the checkpoint's ``config.json``.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.stepfun.configuration_step35 import Step35Config


class Step37VisionConfig(PretrainedConfig):
    """HF-style config for the PE-G/14 vision tower used by Step3.7."""

    model_type = "perception_encoder"

    def __init__(
        self,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        num_channels: int = 3,
        image_size: int = 728,
        mlp_ratio: float = 8960 / 1536,
        patch_size: int = 14,
        hidden_act: str = "quick_gelu",
        layer_norm_eps: float = 1e-5,
        use_cls_token: bool = False,
        use_ln_pre: bool = True,
        use_ln_post: bool = False,
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        ls_init_value: float = 0.1,
        **kwargs,
    ) -> None:
        self.width = width
        self.layers = layers
        self.heads = heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.ls_init_value = ls_init_value
        super().__init__(**kwargs)


class Step37TextConfig(Step35Config):
    """HF-style text-decoder config for Step3.7.

    Identical schema to :class:`Step35Config` — Step3.7's text backbone is
    Step-3.5. Keeping a distinct subclass makes future divergence trivial.
    """

    model_type = "step3p5"


class Step37Config(PretrainedConfig):
    """Top-level HF-style config for Step3.7 (the multimodal wrapper)."""

    model_type = "step3p7"
    architectures = ["Step3p7ForConditionalGeneration"]

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step37VisionConfig]] = None,
        text_config: Optional[Union[dict, Step37TextConfig]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        **kwargs: Any,
    ) -> None:
        if vision_config is None:
            vision_config = Step37VisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Step37VisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step37TextConfig()
        elif isinstance(text_config, dict):
            text_config = Step37TextConfig(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id
        self.hidden_size = text_config.hidden_size
        self.max_position_embeddings = text_config.max_position_embeddings
        super().__init__(**kwargs)


__all__ = ["Step37Config", "Step37TextConfig", "Step37VisionConfig"]
