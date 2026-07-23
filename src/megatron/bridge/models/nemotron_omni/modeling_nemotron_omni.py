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

"""Nemotron Omni model for processor-expanded multimodal token sequences.

Unlike MCore ``LLaVAModel``, it does
not collapse a run of image placeholders before the model and reconstruct it
inside the model. The processor-provided sequence already contains one image
placeholder per projected RADIO feature. This model replaces those positions
in place, then owns sequence packing and context-parallel sharding.

The historical collapse/expand implementation remains available explicitly as
``NemotronOmniLlavaModel`` for compatibility with existing checkpoints, but it
is not the canonical model selected by AutoBridge.
"""

import logging
from collections import namedtuple
from typing import Optional

import torch
from megatron.core import tensor_parallel
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.nemotron_omni.sequence_packing import (
    pack_sequences_from_attention_mask,
)


def _ignore_transformer_engine_extra_state(module: torch.nn.Module, incompatible_keys: namedtuple) -> None:
    """Allow checkpoints produced before Transformer Engine added extra state."""

    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning("Ignoring Transformer Engine checkpoint key %s", key)
                keys.remove(key)


def _build_vision_packed_seq_params(imgs_sizes: torch.Tensor, patch_dim: int) -> PackedSeqParams:
    """Build RADIO's per-image THD boundaries from image sizes."""

    sizes = imgs_sizes.tolist()
    sequence_lengths = [(int(height) // patch_dim) * (int(width) // patch_dim) for height, width in sizes]
    cumulative_lengths = [0]
    for sequence_length in sequence_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)

    cu_seqlens = torch.tensor(cumulative_lengths, dtype=torch.int32, device=imgs_sizes.device)
    max_seqlen = max(sequence_lengths, default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _pixel_shuffle_dynamic_resolution(
    features: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Group each spatial 2x2 patch block into the channel dimension.

    A plain reshape groups four adjacent elements in the flattened sequence,
    which is not the same operation for a row-major non-square patch grid.
    Keep the spatial permutation identical to the historical Omni LLaVA path
    and the HF/vLLM implementation.
    """

    if features.ndim != 3:
        raise ValueError(f"Expected [batch, patches, hidden] features, got {tuple(features.shape)}")
    if height * width != features.shape[1]:
        raise ValueError(f"Patch grid {height}x{width} does not match sequence length {features.shape[1]}")
    if height % 2 or width % 2:
        raise ValueError(f"Pixel shuffle requires an even patch grid, got {height}x{width}")

    batch, _, hidden = features.shape
    shuffled = features.reshape(batch, height, width, hidden)
    shuffled = shuffled.reshape(batch, height, width // 2, hidden * 2)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch, width // 2, height // 2, hidden * 4)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    return shuffled.reshape(batch, (height * width) // 4, hidden * 4)


class NemotronOmniModel(MegatronModule):
    """Nemotron Omni model whose input sequence is already media-expanded.

    NeMo-RL detects ``model_owns_packing`` and passes this model the complete
    padded ``[batch, sequence]`` token tensor and boolean validity mask. Media
    features are inserted before the generic Qwen-style THD packing helper is
    called, so packing and CP sharding happen exactly once.

    Image and text inputs are supported. The sound modules are retained in the
    model namespace, but sound insertion remains unsupported until its
    one-feature-per-placeholder contract is implemented and tested.
    """

    model_owns_packing = True
    model_owns_mtp_loss_mask_packing = True

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        image_token_index: int,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "rope",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        hybrid_layer_pattern: Optional[str] = None,
        img_h: int = 512,
        img_w: int = 512,
        patch_dim: int = 16,
        dynamic_resolution: bool = True,
        vision_class_token_len: int = 10,
        radio_force_eval_mode: bool = False,
        radio_force_cpe_eval_mode: bool = False,
        radio_interpolate_only_cpe: bool = False,
        radio_cpe_aspect_ratio_select: bool = False,
        radio_disable_cpe: bool = False,
        temporal_patch_dim: int = 1,
        separate_video_embedder: bool = False,
        temporal_ckpt_compat: bool = False,
        sound_model: Optional[torch.nn.Module] = None,
        sound_projection: Optional[torch.nn.Module] = None,
        sound_token_index: int = 0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.image_token_index = image_token_index
        self.sound_token_index = sound_token_index
        self.patch_dim = patch_dim
        self.dynamic_resolution = dynamic_resolution
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_hidden_state = None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.language_model = None
        if add_decoder:
            self.language_model = HybridModel(
                config=language_transformer_config,
                hybrid_stack_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                pre_process=pre_process,
                hybrid_layer_pattern=hybrid_layer_pattern,
                post_process=post_process,
                scatter_embedding_sequence_parallel=False,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.language_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

        self.vision_model = None
        self.vision_projection = None
        if add_encoder:
            self.vision_model = RADIOViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                max_img_h=2048,
                max_img_w=2048,
                class_token_len=vision_class_token_len,
                patch_dim=patch_dim,
                add_class_token=True,
                embedder_bias=False,
                dynamic_resolution=dynamic_resolution,
                force_eval_mode=radio_force_eval_mode,
                force_cpe_eval_mode=radio_force_cpe_eval_mode,
                interpolate_only_cpe=radio_interpolate_only_cpe,
                cpe_aspect_ratio_select=radio_cpe_aspect_ratio_select,
                has_cpe=not radio_disable_cpe,
                temporal_patch_dim=temporal_patch_dim,
                separate_video_embedder=separate_video_embedder,
                temporal_ckpt_compat=temporal_ckpt_compat,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                "mlp",
                vision_transformer_config.hidden_size * 4,
                tp_group=pg_collection.tp,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
            self.vision_projection.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

        # Preserve the top-level sound-module namespace for checkpoint
        # conversion while expanded-sequence sound insertion is unsupported.
        self.sound_model = sound_model
        self.sound_projection = sound_projection

    def shared_embedding_or_output_weight(self):
        """Expose the language embedding for Megatron gradient finalization."""

        if self.language_model is None:
            return None
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        """Set the pipeline input on the language model."""

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor must contain exactly one tensor"
        if self.language_model is not None:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        *,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
        freeze_sound_model: bool = False,
        freeze_sound_projection: bool = False,
    ) -> None:
        """Freeze selected leaf components."""

        modules = (
            (freeze_language_model, self.language_model),
            (freeze_vision_model, self.vision_model),
            (freeze_vision_projection, self.vision_projection),
            (freeze_sound_model, self.sound_model),
            (freeze_sound_projection, self.sound_projection),
        )
        for should_freeze, module in modules:
            if should_freeze and module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _merge_projected_media(
        language_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        media_embeddings: torch.Tensor,
        media_token_id: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Replace each valid media placeholder with exactly one feature row."""

        media_mask = input_ids == media_token_id
        if attention_mask is not None:
            media_mask = media_mask & attention_mask.bool()

        expected_features = int(media_mask.sum().item())
        actual_features = media_embeddings.shape[0]
        if expected_features != actual_features:
            raise ValueError(
                "Expanded-sequence media alignment failed: "
                f"found {expected_features} valid placeholders for "
                f"{actual_features} projected features. NemotronOmniModel requires the processor "
                "to emit one placeholder for every projected media token before model-owned packing. "
                "A single placeholder for multiple features usually means this batch was prepared "
                "for the legacy LLaVAModel collapse/expand path. Use expanded processor output, or "
                "load the checkpoint through the explicit NemotronOmniLlavaModelProvider/"
                "NemotronOmniLlavaBridge compatibility path."
            )

        # With batch size one, transpose can remain contiguous and
        # ``contiguous()`` may return the original autograd view. Media
        # insertion is in-place, so force independent storage for the merge.
        merged = language_embeddings.transpose(0, 1).clone()
        if actual_features > 0:
            merged[media_mask] = media_embeddings.to(dtype=merged.dtype)
        return merged.transpose(0, 1).contiguous()

    def _encode_images(
        self,
        images: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor],
        vision_packed_seq_params: Optional[PackedSeqParams],
        num_frames: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode dynamic-resolution images and return one row per image token."""

        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError("Image data was provided on a stage without the vision encoder")

        parameter = next(self.vision_model.parameters())
        images = images.to(dtype=parameter.dtype)

        if imgs_sizes is not None and imgs_sizes.numel() > 0:
            images = self._patchify_dynamic_images(images, imgs_sizes)
            if vision_packed_seq_params is None:
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim)
            use_temporal = getattr(self.vision_model, "temporal_patch_dim", 1) > 1
            if use_temporal and num_frames is None:
                raise ValueError(
                    "num_frames is required by the configured RADIO encoder; "
                    "use one entry with value 1 for each image."
                )
            if num_frames is not None and torch.any(num_frames != 1):
                raise NotImplementedError("Video insertion is not implemented; num_frames must be 1 for image inputs.")

            vision_output = self.vision_model(
                images,
                imgs_sizes=imgs_sizes,
                packed_seq_params=vision_packed_seq_params,
                num_frames=num_frames,
            )
            if use_temporal:
                encoded, imgs_sizes, _ = vision_output
            else:
                encoded = vision_output
            sizes = [(int(height), int(width)) for height, width in imgs_sizes.tolist()]
            class_tokens = (
                self.vision_model.class_token_len if getattr(self.vision_model, "add_class_token", False) else 0
            )
            patch_counts = [(height // self.patch_dim) * (width // self.patch_dim) for height, width in sizes]
            chunks = torch.split(
                encoded.squeeze(0),
                [patch_count + class_tokens for patch_count in patch_counts],
                dim=0,
            )
            chunks = [chunk[class_tokens:] for chunk in chunks]
            shuffled = [
                _pixel_shuffle_dynamic_resolution(
                    chunk.unsqueeze(0),
                    height=height // self.patch_dim,
                    width=width // self.patch_dim,
                ).squeeze(0)
                for chunk, (height, width) in zip(chunks, sizes)
            ]
            encoded = torch.cat(shuffled, dim=0)
        else:
            encoded = self.vision_model(images)
            class_tokens = self.vision_model.class_token_len
            encoded = encoded[:, class_tokens:, :]
            encoded = pixel_shuffle(encoded).reshape(-1, encoded.shape[-1] * 4)

        projected = self.vision_projection(encoded.unsqueeze(1))
        return projected.squeeze(1).contiguous()

    def _patchify_dynamic_images(self, images: torch.Tensor, imgs_sizes: torch.Tensor) -> torch.Tensor:
        """Convert padded processor pixels to RADIO's packed patch representation.

        The processor emits ``[num_images, channels, padded_height, padded_width]``.
        RADIO's dynamic-resolution path consumes
        ``[1, total_patches, channels * patch_dim**2]``. Keeping this conversion
        here makes raw media tensors part of the model contract and avoids an
        Omni-only NeMo-RL pre-forward adapter. Already-patchified inputs remain
        accepted for Bridge/SFT callers.
        """

        patch_features = 3 * self.patch_dim * self.patch_dim
        if images.ndim == 3 and images.shape[0] == 1:
            if images.shape[-1] != patch_features:
                raise ValueError(
                    "Patchified RADIO input has the wrong feature width: "
                    f"expected {patch_features}, got {images.shape[-1]}."
                )
            return images
        if images.ndim != 4:
            raise ValueError(
                "Dynamic-resolution RADIO input must be padded pixels [N,C,H,W] "
                "or packed patches [1,total_patches,C*P*P]; "
                f"got shape {tuple(images.shape)}."
            )
        if images.shape[0] != imgs_sizes.shape[0]:
            raise ValueError(f"Received {images.shape[0]} images but {imgs_sizes.shape[0]} image sizes.")

        patches = []
        for image, size in zip(images, imgs_sizes):
            height, width = (int(value) for value in size.tolist())
            if height % self.patch_dim or width % self.patch_dim:
                raise ValueError(f"Image size {(height, width)} is not divisible by patch_dim={self.patch_dim}.")
            image = image[:, :height, :width]
            channels = image.shape[0]
            rows = height // self.patch_dim
            columns = width // self.patch_dim
            image_patches = (
                image.reshape(
                    channels,
                    rows,
                    self.patch_dim,
                    columns,
                    self.patch_dim,
                )
                .permute(1, 3, 0, 2, 4)
                .reshape(rows * columns, channels * self.patch_dim * self.patch_dim)
            )
            patches.append(image_patches)
        return torch.cat(patches, dim=0).unsqueeze(0).contiguous()

    def _pack_after_media_insertion(
        self,
        *,
        input_ids: torch.Tensor,
        combined_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        PackedSeqParams,
    ]:
        """Pack and CP-shard every token-aligned tensor from one full mask."""

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        packed_input_ids, packed_seq_params = pack_sequences_from_attention_mask(
            input_ids,
            attention_mask,
            pre_process=True,
            pg_collection=self.pg_collection,
        )
        packed_embeddings, _ = pack_sequences_from_attention_mask(
            combined_embeddings.transpose(0, 1).contiguous(),
            attention_mask,
            pre_process=True,
            pg_collection=self.pg_collection,
        )
        packed_embeddings = packed_embeddings.transpose(0, 1).contiguous()

        def pack_optional(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            packed_tensor, _ = pack_sequences_from_attention_mask(
                tensor,
                attention_mask,
                pre_process=True,
                pg_collection=self.pg_collection,
            )
            return packed_tensor

        return (
            packed_input_ids,
            packed_embeddings,
            pack_optional(labels),
            pack_optional(loss_mask),
            packed_seq_params,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        num_frames: Optional[torch.Tensor] = None,
        sound_clips: Optional[torch.Tensor] = None,
        sound_length: Optional[torch.Tensor] = None,
        *,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor:
        """Insert media into the expanded sequence, then call NemotronH."""

        del kwargs, sound_length
        if images is None:
            images = pixel_values

        has_sound = sound_clips is not None and sound_clips.numel() > 0
        if has_sound:
            raise NotImplementedError("Sound insertion is not implemented; use an image or text input.")

        lm_input_ids = input_ids
        combined_embeddings = None
        if self.pre_process:
            if images is not None and images.numel() > 0:
                image_embeddings = self._encode_images(
                    images,
                    imgs_sizes,
                    vision_packed_seq_params,
                    num_frames,
                )
            else:
                image_embeddings = None

            # Match LLaVAModel's execution order. Besides keeping the two
            # implementations directly comparable, this ensures that RADIO's
            # first distributed forward sees the same runtime/collective state.
            input_ids_text = input_ids.masked_fill(input_ids == self.image_token_index, 0)
            combined_embeddings = self.language_model.embedding(input_ids=input_ids_text, position_ids=position_ids)

            if image_embeddings is None:
                image_embeddings = combined_embeddings.new_empty((0, combined_embeddings.shape[-1]))

            combined_embeddings = self._merge_projected_media(
                combined_embeddings,
                input_ids,
                image_embeddings,
                self.image_token_index,
                attention_mask,
            )

        if packed_seq_params is not None:
            if not self.pre_process:
                raise NotImplementedError("Model-owned packing on non-first pipeline stages is not implemented.")
            (
                lm_input_ids,
                combined_embeddings,
                labels,
                loss_mask,
                packed_seq_params,
            ) = self._pack_after_media_insertion(
                input_ids=input_ids,
                combined_embeddings=combined_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            attention_mask = None
            position_ids = None
        elif self.context_parallel_lm > 1:
            raise ValueError("Context parallelism requires model-owned packing for expanded Omni sequences.")

        language_packed_seq_params = packed_seq_params
        if language_packed_seq_params is not None and self.context_parallel_lm == 1 and input_ids.shape[0] == 1:
            # A one-sample CP=1 "pack" neither concatenates sequences nor
            # shards them. Keeping PackedSeqParams here would nevertheless
            # switch Mamba to its packed-sequence kernel. Use the ordinary
            # dense path, which has the same token order and causal semantics
            # but matches vLLM prefill numerics. Multi-sample and CP runs keep
            # packed metadata so their sequence boundaries are preserved.
            language_packed_seq_params = None

        if combined_embeddings is not None and self.sequence_parallel_lm:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings).contiguous()

        # Match LLaVAModel's external-embedding contract. Once media has been
        # merged into decoder embeddings, the language model must not receive
        # the pre-merge token IDs as a second input. MTP is the exception: its
        # training loss derives targets from input_ids, so retain them only
        # when MTP layers are actually enabled.
        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        mtp_enabled = mtp_num_layers is not None and mtp_num_layers > 0
        if combined_embeddings is not None and not mtp_enabled:
            lm_input_ids = None

        return self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_context=inference_context,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=language_packed_seq_params,
        )
