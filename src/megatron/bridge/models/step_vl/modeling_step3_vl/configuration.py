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

# Adapted from stepfun-ai/Step3-VL-10B (Apache 2.0 license)

from typing import Optional, Union

from transformers import Qwen3Config
from transformers.configuration_utils import PretrainedConfig


class StepRoboticsVisionEncoderConfig(PretrainedConfig):
    """Configuration for the Step3-VL custom vision encoder."""

    def __init__(
        self,
        *,
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
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.use_cls_token = use_cls_token
        self.use_ln_pre = use_ln_pre
        self.use_ln_post = use_ln_post
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.ls_init_value = ls_init_value
        super().__init__(**kwargs)


class StepRoboticsConfig(PretrainedConfig):
    """Top-level configuration for the Step3-VL vision–language model."""

    model_type = "step_robotics"

    def __init__(
        self,
        vision_config: Optional[Union[dict, StepRoboticsVisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, Qwen3Config]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 151679,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = StepRoboticsVisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = StepRoboticsVisionEncoderConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen3Config()
        elif isinstance(text_config, dict):
            text_config = Qwen3Config(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.hidden_size = text_config.hidden_size
        self.image_token_id = image_token_id
        super().__init__(**kwargs)
