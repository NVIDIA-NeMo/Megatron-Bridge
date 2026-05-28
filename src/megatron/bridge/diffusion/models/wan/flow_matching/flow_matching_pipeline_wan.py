# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.utils import get_batch_on_this_cp_rank

from megatron.bridge.diffusion.common.flow_matching.adapters.base import FlowMatchingContext, ModelAdapter
from megatron.bridge.diffusion.common.flow_matching.flow_matching_pipeline import FlowMatchingPipeline
from megatron.bridge.diffusion.models.wan.utils import thd_split_inputs_cp


class WanAdapter(ModelAdapter):
    """
    Model adapter for Wan model (Megatron version).

    Handles mapping of standard FlowMatchingContext to Wan specific inputs.
    """

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        grid_sizes = context.batch["grid_sizes"]
        noisy_latents = context.noisy_latents
        context_embeddings = context.batch["context_embeddings"]
        timesteps = context.timesteps
        packed_seq_params = context.batch["packed_seq_params"]

        # ========================================================================
        # Cast model inputs to bf16
        # ========================================================================

        noisy_latents = noisy_latents.to(torch.bfloat16)
        context_embeddings = context_embeddings.to(torch.bfloat16)

        # NOTE: investigate the affect of bf16 timesteps on embedding precision
        # CRITICAL: Keep timesteps in fp32 for embedding precision
        # timesteps = timesteps.float()  # NOT bf16!
        timesteps = timesteps.to(torch.bfloat16)

        if packed_seq_params is None:
            # ====================================================================
            # SBHD mode: inputs are [n, seq, D] (batch-first).
            # Split across CP ranks, then transpose to SBHD [seq, n, D] for model.
            # AttnMaskType.no_mask is used for both self- and cross-attention, so
            # no attention_mask or context_mask tensors are needed.
            # ====================================================================
            if parallel_state.get_context_parallel_world_size() > 1:
                cp_batch = get_batch_on_this_cp_rank({"noisy_latents": noisy_latents})
                noisy_latents = cp_batch["noisy_latents"]

            # Transpose from BSHD [n, s, D] to SBHD [s, n, D]
            noisy_latents = noisy_latents.transpose(0, 1)
            context_embeddings = context_embeddings.transpose(0, 1)

            return {
                "noisy_latents": noisy_latents,
                "grid_sizes": grid_sizes,
                "timesteps": timesteps,
                "context_embeddings": context_embeddings,
                "packed_seq_params": None,
            }
        else:
            # ====================================================================
            # THD mode: inputs are [1, seq, D] (BSHD); transpose to THD/SBHD
            # [seq, 1, D], then split across CP ranks via cu_seqlens.
            # ====================================================================

            # tranpose back to have shape "sbhd"
            # (before we reshaped to "bshd" to be compatible with flow matching pipeline)
            noisy_latents = noisy_latents.transpose(0, 1)

            if parallel_state.get_context_parallel_world_size() > 1:
                noisy_latents = thd_split_inputs_cp(
                    noisy_latents,
                    packed_seq_params["self_attention"].cu_seqlens_q_padded,
                    parallel_state.get_context_parallel_group(),
                )
                # TODO (pmannan): Disable CP for CrossAttention as KV context is small.
                # We don't need to split context embeddings across context parallelism
                # if we disable context parallelism for cross-attention
                context_embeddings = thd_split_inputs_cp(
                    context_embeddings,
                    packed_seq_params["cross_attention"].cu_seqlens_kv_padded,
                    parallel_state.get_context_parallel_group(),
                )

            return {
                "noisy_latents": noisy_latents,
                "grid_sizes": grid_sizes,
                "timesteps": timesteps,
                "context_embeddings": context_embeddings,
                "packed_seq_params": packed_seq_params,
            }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for Wan model.

        Args:
            model: Wan model
            inputs: Dictionary from prepare_inputs()

        Returns:
            Model prediction tensor
        """

        model_pred = model(
            x=inputs["noisy_latents"],
            grid_sizes=inputs["grid_sizes"],
            t=inputs["timesteps"],
            context=inputs["context_embeddings"],
            packed_seq_params=inputs["packed_seq_params"],
            attention_mask=inputs.get("attention_mask"),
            context_mask=inputs.get("context_mask"),
        )
        return self.post_process_prediction(model_pred)


class WanFlowMatchingPipeline(FlowMatchingPipeline):
    """
    Wan-specific Flow Matching pipeline handling Context Parallelism and Custom Noise.

    This pipeline extends the standard FlowMatchingPipeline to support:
    1. Wan-specific noise generation (patching + padding)
    2. Context Parallelism (CP) splitting of inputs
    3. Masked loss computation
    """

    def determine_task_type(self, data_type: str) -> str:
        """Determine task type based on data type and randomization."""
        return "t2v"

    def compute_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_mask = batch["loss_mask"]
        packed_seq_params = batch["packed_seq_params"]

        if packed_seq_params is None:
            # ====================================================================
            # SBHD mode: model_pred comes out of the model as SBHD [S/cp, n, D];
            # transpose to BSHD [n, S/cp, D] so sigma ([n]) broadcasts correctly
            # as loss_weight.view(-1, 1, 1) = [n, 1, 1].
            # target and loss_mask are already BSHD [n, seq_q, D/1] — no transpose needed.
            # ====================================================================
            model_pred = model_pred.transpose(0, 1)  # SBHD [S/cp,n,D] → BSHD [n,S/cp,D]

            if parallel_state.get_context_parallel_world_size() > 1:
                cp_batch = get_batch_on_this_cp_rank({
                    "target": target,
                    "loss_mask": loss_mask,
                })
                target = cp_batch["target"]
                split_loss_mask = cp_batch["loss_mask"]
            else:
                split_loss_mask = loss_mask
            # model_pred:      BSHD [n, S/cp, D]
            # target:          BSHD [n, S/cp, D]
            # split_loss_mask: BSHD [n, S/cp]

        else:
            # ====================================================================
            # THD mode: model_pred comes out of the model as THD [S/cp, 1, D];
            # transpose it to BSHD [1, S/cp, D].
            # target is [1, seq_q, D] (BSHD); transpose to THD [seq_q, 1, D] for
            # CP splitting via cu_seqlens, then transpose back to BSHD [1, S/cp, D].
            # ====================================================================
            model_pred = model_pred.transpose(0, 1)  # THD [S/cp,1,D] → BSHD [1,S/cp,D]

            if parallel_state.get_context_parallel_world_size() > 1:
                target = thd_split_inputs_cp(
                    target.transpose(0, 1),  # BSHD [1,S,D] → THD [S,1,D] for split
                    packed_seq_params["self_attention"].cu_seqlens_q_padded,
                    parallel_state.get_context_parallel_group(),
                ).transpose(0, 1)  # THD [S/cp,1,D] → BSHD [1,S/cp,D]
                split_loss_mask = thd_split_inputs_cp(
                    loss_mask,
                    packed_seq_params["self_attention"].cu_seqlens_q_padded,
                    parallel_state.get_context_parallel_group(),
                ).transpose(0, 1)  # THD [S/cp,1] → BSHD [1,S/cp]
            else:
                split_loss_mask = loss_mask.transpose(0, 1)  # THD [S,1] → BSHD [1,S]
            # model_pred:      BSHD [n=1, S/cp, D]
            # target:          BSHD [n=1, S/cp, D]
            # split_loss_mask: BSHD [n=1, S/cp]

        batch["loss_mask"] = split_loss_mask
        weighted_loss, average_weighted_loss, unweighted_loss, average_unweighted_loss, loss_weight, loss_mask = (
            super().compute_loss(model_pred, target, sigma, batch)
        )
        return weighted_loss, average_weighted_loss, unweighted_loss, average_unweighted_loss, loss_weight, loss_mask
