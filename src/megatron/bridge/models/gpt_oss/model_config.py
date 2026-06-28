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

"""Provider-neutral GPT-OSS model config and builder."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.models.gpt import GPTModelBuilder

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.gpt.yarn import build_gpt_with_yarn


@dataclass(kw_only=True)
class GPTOSSModelConfig(BridgeGPTModelConfig):
    """GPT-OSS build config preserving all YaRN construction parameters."""

    builder: ClassVar[str] = "megatron.bridge.models.gpt_oss.model_config.GPTOSSModelBuilder"
    yarn_rotary_scaling_factor: float = 32.0
    yarn_original_max_position_embeddings: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_mscale: float | None = None
    yarn_mscale_all_dim: float | None = None
    yarn_correction_range_round_to_int: bool = False


class GPTOSSModelBuilder(GPTModelBuilder):
    """GPT builder that binds outer YaRN config to MCore during construction."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build GPT-OSS with YaRN sourced only from the outer config."""
        return build_gpt_with_yarn(
            self._model_config,
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )


__all__ = ["GPTOSSModelBuilder", "GPTOSSModelConfig"]
