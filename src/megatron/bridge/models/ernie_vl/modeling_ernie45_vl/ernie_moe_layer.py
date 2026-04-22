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

"""
Custom MoE modules for ERNIE 4.5 VL MoE dual-pool architecture.

ERNIE 4.5 VL uses a heterogeneous dual-pool MoE where each transformer layer
(except layer 0 which is dense) contains:

- text_moe_layer: 64 experts with intermediate_size=1536 for text tokens
- vision_moe_layer: 64 experts with intermediate_size=512 for vision tokens
- shared_experts: 2 shared experts with intermediate_size=3072 for all tokens

Both pools use separate routers and expert sets. Tokens are dispatched to their
respective pool based on modality (token_type_ids: 0=text, 1=vision).

Module hierarchy (MoE layers):
    decoder.layers.{i}.mlp = ErnieMultiTypeMoE
        .text_moe_layer = MoELayer (standard Megatron)
            .router = TopKRouter
            .experts = SequentialMLP
                .local_experts.{j} = MLP (with linear_fc1, linear_fc2)
        .vision_moe_layer = MoELayer (standard Megatron)
            .router = TopKRouter
            .experts = SequentialMLP
                .local_experts.{j} = MLP (with linear_fc1, linear_fc2)
        .shared_experts = SharedExpertMLP
            .linear_fc1, .linear_fc2

Communication pattern for moe_mm_token_type_ids:
    Megatron-Core's TransformerBlock / TransformerLayer do not propagate extra
    kwargs to MLP layers.  To pass ``moe_mm_token_type_ids`` from
    ``Ernie45VLModel.forward()`` to ``ErnieMultiTypeMoE.forward()`` we use a
    module-level context variable ``_current_moe_mm_token_type_ids`` that is set
    before the language model forward and cleared afterwards.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


# Module-level context variable for passing moe_mm_token_type_ids from
# Ernie45VLModel.forward() to ErnieMultiTypeMoE.forward() without modifying
# Megatron-Core's TransformerBlock / TransformerLayer forward signatures.
# Set by Ernie45VLModel.forward() before calling language_model.forward(),
# cleared after. Read by ErnieMultiTypeMoE.forward() when token_type_ids=None.
_current_moe_mm_token_type_ids: "torch.Tensor | None" = None


def set_moe_mm_token_type_ids(token_type_ids):
    """Set the current moe_mm_token_type_ids for MoE routing.

    Called by ``Ernie45VLModel.forward()`` before ``language_model.forward()``.
    """
    global _current_moe_mm_token_type_ids
    _current_moe_mm_token_type_ids = token_type_ids


def clear_moe_mm_token_type_ids():
    """Clear the current moe_mm_token_type_ids after forward pass.

    Called by ``Ernie45VLModel.forward()`` after ``language_model.forward()``.
    """
    global _current_moe_mm_token_type_ids
    _current_moe_mm_token_type_ids = None


@dataclass
class MultiTypeMoeSubmodules:
    """Submodule specs for the dual-pool MoE layer.

    Attributes:
        text_moe_layer: Spec for the text MoE pool (larger FFN).
        vision_moe_layer: Spec for the vision MoE pool (smaller FFN).
        shared_experts: Spec for the shared expert MLP.
    """

    text_moe_layer: Union[ModuleSpec, type] = None
    vision_moe_layer: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class ErnieMultiTypeMoE(MegatronModule):
    """Dual-pool Mixture of Experts layer for ERNIE 4.5 VL.

    Routes text tokens to text_moe_layer and vision tokens to vision_moe_layer,
    then combines outputs with shared expert output.

    Each pool is a standard Megatron MoELayer with its own router and experts,
    supporting TP and EP parallelism natively.

    Args:
        config: TransformerConfig with moe_intermediate_size as a tuple/list
                of [text_ffn_size, vision_ffn_size].
        submodules: MultiTypeMoeSubmodules containing specs for both pools.
        layer_number: Layer index in the transformer stack.
        pg_collection: Process group collection for parallelism.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MultiTypeMoeSubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        self.layer_number = layer_number

        # TransformerLayer only passes pg_collection to known MLP types (MoELayer,
        # TEGroupedMLP, SequentialMLP). ErnieMultiTypeMoE is not in that list, so
        # pg_collection may be None. Fall back to default MoE process groups.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        # Create separate configs for each pool with different FFN sizes
        self.text_config = deepcopy(config)
        self.vision_config = deepcopy(config)
        self.text_config.moe_ffn_hidden_size = config.moe_intermediate_size[0]
        self.vision_config.moe_ffn_hidden_size = config.moe_intermediate_size[1]

        # Disable shared experts within each MoELayer since ErnieMultiTypeMoE
        # manages its own shared_experts externally (not inside the per-pool MoELayer).
        self.text_config.moe_shared_expert_intermediate_size = None
        self.vision_config.moe_shared_expert_intermediate_size = None

        # Build the two MoE pools and shared experts
        self.text_moe_layer = build_module(submodules.text_moe_layer, self.text_config)
        self.vision_moe_layer = build_module(submodules.vision_moe_layer, self.vision_config)
        self.shared_experts = build_module(
            submodules.shared_experts,
            config=config,
            pg_collection=pg_collection,
            gate=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for dual-pool MoE.

        Args:
            hidden_states: Input tensor [seq_len, batch, hidden_size].
                When Sequence Parallel (SP) is enabled, seq_len is the local
                partition size (full_seq_len / tp_size).
            token_type_ids: Modality indicator [batch, seq_len].
                0 = text token -> text_moe_layer
                1 or 2 = vision token -> vision_moe_layer
                When SP is enabled, this must already be sliced to match the
                local sequence partition (done by Ernie45VLModel.forward()).
            padding_mask: Optional padding mask [batch, seq_len] passed by
                Megatron's TransformerLayer. Forwarded to each MoE pool's
                router for filtering out padding tokens during routing.

        Returns:
            Tuple of (output, bias). bias is always None.
        """
        if token_type_ids is None:
            # Read from the module-level context variable set by
            # Ernie45VLModel.forward() before language_model.forward().
            token_type_ids = _current_moe_mm_token_type_ids

        if token_type_ids is None:
            # Ultimate fallback: all text tokens (no vision routing)
            token_type_ids = torch.zeros(
                hidden_states.shape[1],
                hidden_states.shape[0],
                dtype=torch.long,
                device=hidden_states.device,
            )

        # hidden_states: [seq_len, batch, hidden_size]
        # token_type_ids: [batch, seq_len]  (0=text, >=1 = vision)
        seq_len, batch_size, hidden_size = hidden_states.shape

        # Flatten batch and seq dims to create a flat token stream.
        # This supports batch_size > 1 correctly, whereas the old code
        # assumed batch=1 via squeeze(0).
        # flat_hidden: [seq_len * batch, 1, hidden_size]
        # We reshape to [N, 1, H] so that each MoELayer still sees
        # a 3D tensor with batch_dim=1 (required by MoELayer internals).
        flat_hidden = hidden_states.permute(1, 0, 2).reshape(
            batch_size * seq_len, 1, hidden_size
        )  # [batch * seq_len, 1, hidden]

        # Flatten token_type_ids: [batch, seq_len] -> [batch * seq_len]
        flat_type_ids = token_type_ids.reshape(-1)  # [batch * seq_len]

        # Build per-token modality mask. True = vision token.
        vision_mask = flat_type_ids.bool()  # >=1 means vision

        # ---------- Filter tokens by modality BEFORE routing ----------
        # This matches HF behaviour: text_moe only sees text tokens,
        # vision_moe only sees vision tokens.  The routers and expert
        # computation never touch wrong-modality tokens.
        text_indices = (~vision_mask).nonzero(as_tuple=True)[0]
        vision_indices = vision_mask.nonzero(as_tuple=True)[0]

        text_hidden = flat_hidden[text_indices]  # [N_text, 1, hidden]
        vision_hidden = flat_hidden[vision_indices]  # [N_vision, 1, hidden]

        # Route filtered tokens through their respective MoE pools.
        #
        # When EP > 1, both pools must ALWAYS be called on all EP ranks,
        # even if N_text=0 or N_vision=0 on this rank, because each
        # MoELayer's token_dispatcher performs alltoall/allgather
        # collectives that require all EP ranks to participate.
        #
        # MCore's RouterGatingLinearFunction cannot reshape [0]-element
        # tensors (shape [0, 1, -1]).  When a pool has 0 tokens but EP > 1
        # requires collective participation, we inject a single dummy token,
        # run the MoE forward (so collectives execute), then discard the
        # dummy output.
        #
        # When EP == 1 (no expert parallelism), we can safely skip the
        # empty pool since there are no inter-rank collectives.
        ep_size = parallel_state.get_expert_model_parallel_world_size()

        if text_indices.numel() > 0:
            text_output, _ = self.text_moe_layer(text_hidden, padding_mask=None)
        elif ep_size > 1:
            # Inject a dummy token to participate in EP collectives
            dummy = torch.zeros(1, 1, hidden_size, dtype=flat_hidden.dtype, device=flat_hidden.device)
            _, _ = self.text_moe_layer(dummy, padding_mask=None)
            text_output = None
        else:
            text_output = None

        if vision_indices.numel() > 0:
            vision_output, _ = self.vision_moe_layer(vision_hidden, padding_mask=None)
        elif ep_size > 1:
            # Inject a dummy token to participate in EP collectives
            dummy = torch.zeros(1, 1, hidden_size, dtype=flat_hidden.dtype, device=flat_hidden.device)
            _, _ = self.vision_moe_layer(dummy, padding_mask=None)
            vision_output = None
        else:
            vision_output = None

        # Scatter back to full flat sequence
        flat_moe_output = torch.zeros_like(flat_hidden)
        if text_output is not None and text_indices.numel() > 0:
            flat_moe_output[text_indices] = text_output
        if vision_output is not None and vision_indices.numel() > 0:
            flat_moe_output[vision_indices] = vision_output

        # Reshape back to [seq_len, batch, hidden] from [batch * seq_len, 1, hidden]
        moe_output = flat_moe_output.reshape(batch_size, seq_len, hidden_size).permute(1, 0, 2).contiguous()

        # Shared experts see ALL tokens (same as HF)
        # SharedExpertMLP.forward() returns a single tensor (not a tuple),
        # unlike MLP.forward() which returns (output, output_bias).
        shared_result = self.shared_experts(hidden_states)
        shared_output = shared_result[0] if isinstance(shared_result, tuple) else shared_result

        moe_output = moe_output + shared_output

        return moe_output, None

    def set_layer_number(self, layer_number: int):
        """Set the layer number for both MoE pools."""
        self.layer_number = layer_number
        if hasattr(self.text_moe_layer, "router"):
            self.text_moe_layer.router.set_layer_number(layer_number)
        if hasattr(self.vision_moe_layer, "router"):
            self.vision_moe_layer.router.set_layer_number(layer_number)
