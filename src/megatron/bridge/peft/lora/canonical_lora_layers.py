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

from typing import Optional, Tuple

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from torch import nn

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper


class ModuleDict(nn.ModuleDict):
    """
    nn.ModuleDict with a sharded_state_dict implementation for checkpointing
    """

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
    ) -> "ShardedStateDict":
        """Retrieve the sharded state dictionary of the wrapped module and adapter.

        This method is used for distributed checkpointing, combining the sharded states
        of both the main module and the adapter.

        Args:
            prefix (str): A prefix added to parameter and buffer names. Defaults to ''.
            sharded_offsets (Tuple[Tuple[int, int, int]]): Offsets for sharded parameters.
                                                           Defaults to an empty tuple.
            metadata (Optional[dict]): Additional metadata for the sharded state.
                                       Defaults to None.

        Returns:
            ShardedStateDict: The combined sharded state dictionary.
        """
        sharded_state_dict = {}
        for key, layer in self.items():
            sharded_state_dict.update(layer.sharded_state_dict(f"{prefix}{key}.", sharded_offsets, metadata))
        return sharded_state_dict


class LoRALinearSplitQKV(AdapterWrapper):
    """An adapter wrapper for `linear_qkv` where q, k, v are three separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x)

        if getattr(self, "_merged", False):
            # Adapters already merged into base weights
            return linear_output, bias

        query = self.adapter.adapter_q(layernorm_output)
        key = self.adapter.adapter_k(layernorm_output)
        value = self.adapter.adapter_v(layernorm_output)

        query_4d = query.reshape(query.shape[0], query.shape[1], -1, self.to_wrap.config.kv_channels)
        key_4d = key.reshape(key.shape[0], key.shape[1], -1, self.to_wrap.config.kv_channels)
        value_4d = value.reshape(value.shape[0], value.shape[1], -1, self.to_wrap.config.kv_channels)

        qkv_4d = torch.cat([query_4d, key_4d, value_4d], dim=2)
        adapter_output = qkv_4d.reshape(qkv_4d.shape[0], qkv_4d.shape[1], -1)

        return linear_output + adapter_output, bias


class LoRALinearSplitFC1UpGate(AdapterWrapper):
    """An adapter wrapper for `linear_fc1` where up_proj and gate_proj are two separate adapters.
    This module that adds the output of the adapters to the output of the wrapped module while taking care of shape.

    This class is designed to be used with LoRA (Low-Rank Adaptation) and similar techniques
    where the adapter's output is added to the main module's output. It extends the AdapterWrapper
    class to provide a specific implementation of the forward method.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pylint: disable=C0115,C0116
        linear_output, bias, layernorm_output = self.base_linear_forward(x)

        if getattr(self, "_merged", False):
            # Adapters already merged into base weights
            return linear_output, bias

        adapter_output_gate = self.adapter.adapter_gate(layernorm_output)
        adapter_output_up = self.adapter.adapter_up(layernorm_output)
        adapter_output = torch.cat([adapter_output_gate, adapter_output_up], dim=2)
        return linear_output + adapter_output, bias
