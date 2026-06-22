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

"""Shared Transformer Engine layer extensions for Post-LayerNorm architectures."""

from megatron.core.transformer import TransformerConfig

from megatron.bridge.utils.import_utils import safe_import_from


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")


class TERowParallelLinearLayerNorm(TERowParallelLinear):
    """Row-parallel linear with an additional Post-LayerNorm on the output.

    Used by models that attach a Post-LN module to row-parallel projection
    outputs, such as Gemma2, Gemma3, and EXAONE 4.0.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x):
        """Forward with additional Post-LN on output."""
        output, bias = super().forward(x)
        if bias is not None:
            raise ValueError(
                "TERowParallelLinearLayerNorm assumes add_bias_linear=False. "
                "Post-LN before deferred bias addition is incorrect when bias is present."
            )
        return self.post_layernorm(output), bias
