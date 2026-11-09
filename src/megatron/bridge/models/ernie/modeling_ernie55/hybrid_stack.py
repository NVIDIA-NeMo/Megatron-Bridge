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

"""ERNIE 5.5 decoder stack: Megatron-Core's ``HybridStack`` carrying fp32 residual streams.

ERNIE 5.5 casts the embedding output to fp32 before it expands into the hyper-connection streams,
so every decoder residual stream is fp32 while the layers compute in bf16 (their pre-norms
downcast the fp32 layer input, see :mod:`~megatron.bridge.models.ernie.modeling_ernie55.norm`).
Megatron-Core cannot express this with configuration: it rejects ``fp32_residual_connection``
together with hyper-connections, ``HybridModel`` has no hook between the embedding and the
decoder, and ``HybridStack`` builds its final norm as a plain ``TENorm``.
:class:`Ernie55HybridStack` adds exactly those pieces:

* the fp32 cast of the embedding at the decoder entry (``pre_process`` stacks with
  hyper-connections; MTP stacks, whose streams stay bf16, are Megatron-Core's own class);
* a downcasting final norm, since the learned stream contraction returns fp32; under
  ``ernie_reference_rmsnorm`` it is :class:`~megatron.bridge.models.ernie.modeling_ernie55.norm.Ernie55RMSNorm`,
  the reference implementation's arithmetic, instead of Megatron-Core's ``TENorm``;
* the ERNIE 5.5 hyper-connection arithmetic on its layers
  (:func:`~megatron.bridge.models.ernie.modeling_ernie55.hyper_connection.use_ernie55_hyper_connections`).

The MTP stack, which Megatron-Core builds directly as ``HybridStack``, gets the same
hyper-connection arithmetic from
:class:`~megatron.bridge.models.ernie.modeling_ernie55.model.Ernie55HybridModel`.
"""

from __future__ import annotations

from typing import Any

import torch
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import WrappedTensor
from torch import Tensor

from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import (
    use_ernie55_hyper_connections,
    validate_ernie55_mhc_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.norm import (
    Ernie55RMSNorm,
    downcast_norm_input,
    is_downcasting_norm,
    uses_reference_rmsnorm,
)


__all__ = ["Ernie55HybridStack"]


class Ernie55HybridStack(HybridStack):
    """``HybridStack`` with the ERNIE 5.5 fp32 residual-stream contract.

    Constructor arguments are Megatron-Core's (they differ between Megatron-Core versions and pass
    through unchanged). With hyper-connections the constructor validates the config with
    :func:`~megatron.bridge.models.ernie.modeling_ernie55.hyper_connection.validate_ernie55_mhc_config`
    (which also rejects pipeline parallelism with a pipeline dtype narrower than fp32) and checks
    that every layer norm the streams reach downcasts its input, so no layer can receive an fp32
    activation. With ``ernie_reference_rmsnorm`` the final norm is replaced by
    :class:`~megatron.bridge.models.ernie.modeling_ernie55.norm.Ernie55RMSNorm` right after
    Megatron-Core builds it.

    Raises:
        TypeError: If hyper-connections are on and a layer's input or pre-MLP norm does not
            downcast (the stack was built from a spec other than
            :func:`~megatron.bridge.models.ernie.modeling_ernie55.ernie55_hybrid_stack_spec`).
    """

    def __init__(self, config: TransformerConfig, *args: Any, **kwargs: Any) -> None:
        validate_ernie55_mhc_config(config)
        super().__init__(config, *args, **kwargs)
        final_norm = getattr(self, "final_norm", None)
        if final_norm is not None:
            if uses_reference_rmsnorm(self.config):
                # Megatron-Core builds the final norm as TENorm directly; nothing refers to it yet.
                final_norm = Ernie55RMSNorm(
                    config=self.config, hidden_size=self.config.hidden_size, eps=self.config.layernorm_epsilon
                )
                self.final_norm = final_norm
            downcast_norm_input(final_norm)
        if self.config.enable_mhc_connections:
            use_ernie55_hyper_connections(self)
            for index, layer in enumerate(self.layers):
                inner = getattr(layer, "inner_layer", layer)
                for slot in ("input_layernorm", "pre_mlp_layernorm"):
                    norm = getattr(inner, slot, None)
                    if isinstance(norm, torch.nn.Module) and not isinstance(norm, IdentityOp):
                        if not is_downcasting_norm(norm):
                            raise TypeError(
                                f"ERNIE 5.5 layer {index} has a {slot} that does not downcast its input, so it "
                                "would normalize an fp32 hyper-connection stream; build the stack from "
                                "ernie55_hybrid_stack_spec()."
                            )

    @property
    def carries_fp32_streams(self) -> bool:
        """Whether this stack casts its input to fp32 before expanding it into streams."""
        return bool(self.config.enable_mhc_connections and self.pre_process and not self.is_mtp_layer)

    def forward(self, hidden_states: Tensor | WrappedTensor, *args: Any, **kwargs: Any) -> Tensor:
        """Run Megatron-Core's ``HybridStack.forward`` on the fp32 embedding when the stack carries fp32 streams.

        Args:
            hidden_states: The embedding output (``pre_process``) or ignored (later pipeline stages
                read ``input_tensor``); may be a ``WrappedTensor`` in inference.
            *args: Passed through.
            **kwargs: Passed through.

        Returns:
            Whatever ``HybridStack.forward`` returns.
        """
        if self.carries_fp32_streams:
            if isinstance(hidden_states, WrappedTensor):
                hidden_states = hidden_states.unwrap()
            hidden_states = hidden_states.to(torch.float32)
        return super().forward(hidden_states, *args, **kwargs)
