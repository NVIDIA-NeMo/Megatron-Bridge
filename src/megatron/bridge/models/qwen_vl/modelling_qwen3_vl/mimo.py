# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Carry Qwen3-VL DeepStack data in each MIMO context-parallel batch."""

from dataclasses import dataclass
from typing import Any

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.models.mimo import MimoModel

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


@dataclass(eq=False)
class CombinedEmbeddingsWithDeepStackFeatures:
    """Carry aligned token embeddings and auxiliary image rows into input preparation."""

    embeddings: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]

    @property
    def shape(self) -> torch.Size:
        """Return the aligned token shape used by MIMO forward logging."""
        return self.embeddings.shape


@dataclass(eq=False)
class DeepStackContextParallelBatch(ContextParallelBatch):
    """Keep rank-local image indices and differentiable features for one forward."""

    image_token_indices: torch.Tensor | None = None
    deepstack_features: tuple[torch.Tensor, ...] = ()

    def get_deepstack_inputs(self) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Select this CP/SP rank's image rows in local batch-major token order."""
        if self.image_token_indices is None:
            return None, None

        # Select auxiliary rows in the same batch-major order as local image tokens.
        # Negative indices exclude text and padding from every injection layer.
        mask = self.image_token_indices >= 0
        rows = self.image_token_indices[mask]
        return mask, [feature.index_select(0, rows) for feature in self.deepstack_features]


class Qwen3VLMIMOVisionModel(Qwen3VLVisionModel):
    """Pack primary and DeepStack embeddings for the tensor-only image boundary."""

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        grid_thw: torch.Tensor,
        inference_params: InferenceParams | None = None,
        extra_block_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Concatenate primary and auxiliary features along the hidden dimension."""
        primary, deepstack = super().forward(hidden_states, grid_thw, inference_params, extra_block_kwargs)
        # Keep all visual features in one [num_image_tokens, (1 + num_deepstack) * H]
        # payload so they share the image boundary's forward/backward communication.
        return torch.cat([primary, *deepstack], dim=-1)


class Qwen3VLMIMOModel(MimoModel):
    """Prepare aligned embeddings and DeepStack features for the language component."""

    def align_embeddings_by_token_positions(
        self,
        modality_embeddings: dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        special_token_ids: dict[str, int],
        modality_token_indices: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | CombinedEmbeddingsWithDeepStackFeatures:
        """Pass primary embeddings and DeepStack metadata together to preparation."""
        # Text-only batches and encoders without DeepStack need only standard alignment.
        images = modality_embeddings.get("images")
        config = self.mimo_config.language_model_spec.params["config"]
        count = len(config.vision_config.deepstack_visual_indexes)
        if images is None or not images.numel() or not count:
            return super().align_embeddings_by_token_positions(
                modality_embeddings, input_ids, special_token_ids, modality_token_indices
            )

        # Separate the H-wide primary embeddings from the auxiliary decoder features.
        expected_width = config.hidden_size * (count + 1)
        if images.shape[-1] != expected_width:
            raise ValueError(f"Qwen3-VL image payload has width {images.shape[-1]}, expected {expected_width}.")
        primary, *features = images.split(config.hidden_size, dim=-1)

        # Insert only primary features into the token stream ([S, B, H]). Carry the
        # auxiliary image rows alongside it until rank-local token positions are known.
        primary_inputs = dict(modality_embeddings, images=primary)
        combined = super().align_embeddings_by_token_positions(
            primary_inputs, input_ids, special_token_ids, modality_token_indices
        )
        return CombinedEmbeddingsWithDeepStackFeatures(combined, tuple(features))

    def _prepare_language_inputs(
        self,
        embeddings: torch.Tensor | CombinedEmbeddingsWithDeepStackFeatures | None,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_padded: torch.Tensor | None,
        max_seqlen: torch.Tensor | None,
        owns_mtp: bool,
        text_token_indices: torch.Tensor | None = None,
    ) -> DeepStackContextParallelBatch:
        """Partition image-token indices with the CP layout and SP split."""
        # Map each image-token position to its row in the unpartitioned visual features.
        indices = None
        features: tuple[torch.Tensor, ...] = ()
        if isinstance(embeddings, CombinedEmbeddingsWithDeepStackFeatures):
            features = embeddings.deepstack_features
            embeddings = embeddings.embeddings
            assert input_ids is not None
            image_positions = (input_ids == self.special_token_ids["images"]).reshape(-1).nonzero().flatten()
            # Zero denotes non-image tokens, including padding added by the CP partitioner.
            indices = torch.zeros_like(input_ids)
            indices.view(-1).index_copy_(
                0, image_positions, torch.arange(1, image_positions.numel() + 1, device=input_ids.device)
            )

        # Prepare decoder embeddings and language metadata with the configured CP/SP
        # partition. DeepStack indices below must follow exactly the same token order.
        prepared = super()._prepare_language_inputs(
            embeddings=embeddings,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            max_seqlen=max_seqlen,
            owns_mtp=owns_mtp,
            text_token_indices=text_token_indices,
        )
        # One local index map describes one shared linear/attention token layout.
        if len(prepared.batches_by_layout) != 1:
            raise NotImplementedError("Qwen3-VL DeepStack currently requires one CP layout.")
        if indices is not None:
            # Apply the same CP partition, including per-document packed boundaries.
            # Token metadata remains [B, S/cp] here; the adapter applies SP only to embeddings.
            index_batch = self.partition_adapter.partition(
                embeddings=None,
                input_ids=indices,
                position_ids=None,
                labels=None,
                loss_mask=None,
                mtp_input_mask=None,
                cu_seqlens=cu_seqlens,
                cu_seqlens_padded=cu_seqlens_padded,
                max_seqlen=max_seqlen,
            )
            # Restore zero-based image rows, with -1 for text and CP padding.
            indices = index_batch.get_batch()["tokens"] - 1
            if self.partition_adapter.cfg.seq_parallel:
                # Match the decoder's sequence-first SP split, then restore batch-major
                # indices for boolean masking inside the DeepStack decoder layers.
                group = self.partition_adapter.cfg.tp_group
                indices = (
                    tensor_parallel.scatter_to_sequence_parallel_region(
                        indices.transpose(0, 1).contiguous(), group=group
                    )
                    .transpose(0, 1)
                    .contiguous()
                )
                # The bridge sends one TP leader's input gradient. Sum disjoint SP
                # image-row gradients so every TP rank has that complete gradient.
                features = tuple(
                    tensor_parallel.copy_to_tensor_model_parallel_region(f, group=group) for f in features
                )

        # Bind local image indices and differentiable features to this forward's
        # language batch, together with its packed-sequence and CP metadata.
        return DeepStackContextParallelBatch(
            boundary_layout=prepared.boundary_layout,
            batches_by_layout=prepared.batches_by_layout,
            packed_seq_params_by_layout=prepared.packed_seq_params_by_layout,
            thd_plan=prepared.thd_plan,
            image_token_indices=indices,
            deepstack_features=features,
        )

    def _forward_prepared_language_inputs(
        self, cp_batch: DeepStackContextParallelBatch, attention_mask: torch.Tensor | None
    ) -> Any:
        """Forward language inputs and rank-local DeepStack features to the decoder."""
        batch = cp_batch.get_batch()
        # Select this microbatch's CP/SP-local image mask and auxiliary feature rows
        # for the Qwen decoder's visual inputs.
        visual_pos_masks, deepstack_visual_embeds = cp_batch.get_deepstack_inputs()
        return self.language_model(
            input_ids=batch["tokens"],
            position_ids=batch["position_ids"],
            attention_mask=attention_mask,
            decoder_input=batch["decoder_input"],
            labels=batch["labels"],
            loss_mask=batch["loss_mask"],
            mtp_input_mask=batch["mtp_input_mask"],
            packed_seq_params=cp_batch.get_packed_seq_params(),
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
