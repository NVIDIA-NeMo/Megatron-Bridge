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

import torch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_tp_sharded_tensor_for_checkpoint
from torch import nn

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.utils import ParallelLinearAdapter


class ParallelLinearDoRAAdapter(ParallelLinearAdapter):
    """
    Adapter class for DoRA to handle the additional weight_magnitude parameter.

    This class extends ParallelLinearAdapter to add DoRA-specific functionality,
    including weight magnitude tracking and sharded state dict support for distributed training.
    """

    def init_weight_magnitude(self, value: torch.Tensor) -> None:
        """
        Initialize weight_magnitude with shape (d,), where d is the output dim of the linear layer.

        Args:
            value (torch.Tensor): Initial values for the weight magnitude parameter.
        """
        self.weight_magnitude = nn.Parameter(value, requires_grad=True)

    def get_weight_magnitude(self) -> torch.Tensor:
        """
        Public function to get the weight magnitude parameter.

        Returns:
            torch.Tensor: The weight magnitude parameter.
        """
        return self.weight_magnitude

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """
        Sharded state dict implementation for DoRA adapter.
        Weight magnitude is TP sharded for linear_qkv and linear_fc1 only.

        Args:
            prefix (str): Prefix for parameter names. Defaults to ''.
            sharded_offsets (tuple): Offsets for sharded parameters. Defaults to ().
            metadata (dict | None): Additional metadata. Defaults to None.

        Returns:
            ShardedStateDict: The sharded state dictionary.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        magnitude_key = f"{prefix}weight_magnitude"
        if self.input_is_parallel:
            # RPL output is gathered, so weight_magnitude is not sharded for TP
            magnitude_sharded_tensor = make_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, prepend_offsets=sharded_offsets
            )
        else:
            # CPL output is sharded, so weight_magnitude is sharded for TP
            magnitude_sharded_tensor = make_tp_sharded_tensor_for_checkpoint(
                self.weight_magnitude, magnitude_key, 0, prepend_offsets=sharded_offsets
            )
        sharded_state_dict[magnitude_key] = magnitude_sharded_tensor

        return sharded_state_dict


class DoRALinear(AdapterWrapper):
    """
    An adapter wrapper that is designed to be used with DoRA.

    DoRA (Weight-Decomposed Low-Rank Adaptation) extends LoRA by decomposing
    the pre-trained weight into magnitude and direction components. This class
    implements the DoRA forward pass that applies magnitude scaling to the
    combined base and adapter outputs.

    It extends the AdapterWrapper class to provide DoRA-specific implementation
    of the forward method.
    """

    def __init__(self, to_wrap: nn.Module, adapter: ParallelLinearDoRAAdapter):
        """
        Initialize the DoRALinear wrapper.

        Args:
            to_wrap (nn.Module): The base linear module to wrap.
            adapter (ParallelLinearDoRAAdapter): The DoRA adapter instance.
        """
        super().__init__(to_wrap, adapter)
        self.adapter: ParallelLinearDoRAAdapter
        self.scaling = adapter.alpha / adapter.dim
        self.adapter.init_weight_magnitude(self._get_weight_norm())

    def _get_weight_norm(self) -> torch.Tensor:
        """
        Calculate the norm of the combined weight matrix (W_0 + B A).

        This method handles tensor parallel communication to gather weights
        when needed and computes the L2 norm along the appropriate dimension.

        Returns:
            torch.Tensor: The L2 norm of the combined weight matrix.
        """
        if self.adapter.input_is_parallel:
            linear_out_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_out.weight.T).T
            linear_in_weight = self.adapter.linear_in.weight
        else:
            linear_out_weight = self.adapter.linear_out.weight
            linear_in_weight = gather_from_tensor_model_parallel_region(self.adapter.linear_in.weight.T).T

        weight = self.to_wrap.weight + self.scaling * linear_out_weight @ linear_in_weight
        return torch.linalg.norm(weight, dim=1).to(weight.dtype).detach()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method for DoRA.

        The DoRA forward pass implements:
          mag_norm_scale * (linear_output + adapter_output)
        = ||W_0 + B_0 A_0|| / ||W_0 + B A|| * (W_0 x + B A x)
        = ||W_0 + B_0 A_0|| ((W_0 + B A) / ||W_0 + B A||) x
        = m ((W_0 + B A) / ||W_0 + B A||) x
        = equation 5 in DoRA paper

        When dropout is used, equation becomes:
          W_0 x + (m /||W_0 + B A|| - 1) W_0 dropout(x) + m /||W_0 + B A|| B A dropout(x)
        = ...
        = m /||W_0 + B A|| (W_0 x + B A dropout(x)) + (m /||W_0 + B A|| - 1) W_0 (dropout(x) - x)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the DoRA output and bias term.
        """
        linear_output, bias, layernorm_output = self.base_linear_forward(x)
        adapter_output = self.adapter(layernorm_output.contiguous())

        # mag_norm_scale is  ||W_0 + B_0 A_0|| / ||W_0 + B A||  (scaling in front of BA not shown)
        mag_norm_scale = (self.adapter.get_weight_magnitude() / self._get_weight_norm()).view(1, 1, -1)
        if self.adapter.dropout is None or not self.training:
            dropout_correction = 0
        else:
            dropout_correction = (mag_norm_scale - 1) * self.base_linear_forward(
                self.adapter.dropout(layernorm_output) - layernorm_output
            )[0]

        return mag_norm_scale * (linear_output + adapter_output) + dropout_correction, bias
