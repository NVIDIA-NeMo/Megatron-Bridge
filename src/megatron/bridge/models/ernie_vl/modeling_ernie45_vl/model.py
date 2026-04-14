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
Megatron-Core compatible ERNIE 4.5 VL MoE model.

This module wraps the HuggingFace ERNIE 4.5 VL MoE vision encoder and resampler
with a Megatron-Core GPT language model to create a distributable VLM.

Architecture:
    - Vision Tower: Ernie4_5_VLMoeVisionTransformerPretrainedModel (HF, replicated across TP)
    - Resampler: Ernie4_5_VLMoeVariableResolutionResamplerModel (HF, replicated across TP)
    - Language Model: MCoreGPTModel (Megatron-Core, distributed across TP/PP/EP)
      with custom ErnieMultiTypeMoE layers supporting dual-pool MoE:
        * text_moe_layer: 64 experts (FFN=1536) for text tokens
        * vision_moe_layer: 64 experts (FFN=512) for vision tokens
        * shared_experts: shared MLP for all tokens
"""

import types
from typing import Optional

import torch
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from torch import Tensor
from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeModel,
    Ernie4_5_VLMoeVariableResolutionResamplerModel,
    Ernie4_5_VLMoeVisionTransformerPretrainedModel,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync

from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_moe_layer import (
    set_moe_mm_token_type_ids,
    clear_moe_mm_token_type_ids,
)
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.vision_model import ErnieVLVisionModel
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.vision_layer_spec import get_ernie_vit_layer_spec
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.vision_transformer_config import get_ernie_vision_config
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core import parallel_state


def _normalize_hf_config(hf_config):
    """Ensure the HF config has a ``text_config`` attribute.

    The transformers-builtin ``Ernie4_5_VLMoeVariableResolutionResamplerModel`` accesses
    ``config.text_config.hidden_size`` and ``config.text_config.rms_norm_eps``.  The
    nested config (Instruct model) has ``text_config`` as a sub-object, but the flat
    config (Thinking model) stores all LLM fields directly on the top-level config.

    For flat configs, we set ``text_config`` to point to the config itself so that
    ``config.text_config.hidden_size`` resolves to ``config.hidden_size``.
    """
    if hf_config is not None and not hasattr(hf_config, "text_config"):
        hf_config.text_config = hf_config
    return hf_config


def _normalize_vision_config(vision_config, hf_config=None):
    """Ensure the vision config has all attributes required by the transformers-builtin
    vision model classes (Ernie4_5_VLMoeVisionBlock, Ernie4_5_VLMoeVisionTransformerPretrainedModel,
    Ernie4_5_VLMoeVariableResolutionResamplerModel).

    The Thinking model's custom ``DFNRopeVisionTransformerConfig`` (auto_map) uses
    ``mlp_ratio`` + ``embed_dim`` instead of ``intermediate_size``, omits ``rms_norm_eps``,
    and omits ``temporal_merge_size``.  This function adds the missing attributes so
    the same config object works with the transformers-builtin vision model code.
    """
    # rms_norm_eps: used by LayerNorm in vision blocks and final LN
    if not hasattr(vision_config, "rms_norm_eps"):
        # Prefer the top-level config's rms_norm_eps (LLM side), default 1e-6
        rms_norm_eps = 1e-6
        if hf_config is not None:
            rms_norm_eps = getattr(hf_config, "rms_norm_eps", 1e-6)
        vision_config.rms_norm_eps = rms_norm_eps

    # intermediate_size: used by vision MLP (= mlp_ratio * embed_dim)
    if not hasattr(vision_config, "intermediate_size"):
        mlp_ratio = getattr(vision_config, "mlp_ratio", 4)
        embed_dim = getattr(vision_config, "embed_dim", getattr(vision_config, "hidden_size", 1280))
        vision_config.intermediate_size = int(mlp_ratio * embed_dim)

    # temporal_merge_size: used by resampler for temporal pooling
    if not hasattr(vision_config, "temporal_merge_size"):
        vision_config.temporal_merge_size = getattr(vision_config, "spatial_merge_size", 2)

    return vision_config


class _MgVitTowerAdapter(torch.nn.Module):
    """Thin adapter that makes the MG-native ErnieVLVisionModel compatible
    with the HF-bound ``get_image_features`` / ``get_video_features`` methods.

    The HF methods call ``self.vision_tower(pixel_values, grid_thw, return_dict=True)``
    and expect a ``BaseModelOutputWithPooling`` with ``.last_hidden_state``.  They also
    access ``self.vision_tower.spatial_merge_size``.

    This adapter wraps ``ErnieVLVisionModel`` to match that interface exactly.
    """

    def __init__(self, mg_vision_model: "ErnieVLVisionModel"):
        super().__init__()
        self.mg_vision_model = mg_vision_model
        self.spatial_merge_size = mg_vision_model.spatial_merge_size

    def forward(self, pixel_values, grid_thw, return_dict=True, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        hidden_states = self.mg_vision_model(pixel_values, grid_thw)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class ErnieMultimodalRotaryEmbedding(MultimodalRotaryEmbedding):
    """ERNIE-specific 3D M-RoPE with interleaved H/W frequency allocation.

    ERNIE 4.5 VL uses a custom RoPE layout that differs from the standard
    Qwen2VL-style contiguous block layout used by ``MultimodalRotaryEmbedding``.

    Standard (Qwen2VL) layout with mrope_section=[22, 22, 20]:
        head dims [0:44]   -> T (temporal) axis, freq bands 0-21
        head dims [44:88]  -> H (height) axis,   freq bands 0-21
        head dims [88:128] -> W (width) axis,     freq bands 0-19

    ERNIE layout with freq_allocation=20:
        head dims [0:44]   -> H,W interleaved: even freq bands -> H, odd -> W
        head dims [44:88]  -> (same interleaving continues)
        head dims [88:128] -> T (temporal) axis, freq bands 44-63

    More precisely, for freq band index f (0..63):
        f in {0,2,4,...,42}  (even, f<44) -> H position
        f in {1,3,5,...,43}  (odd,  f<44) -> W position
        f in {44,45,...,63}  (last 20)    -> T position

    For text tokens (T=H=W=p), both layouts produce identical results since
    all axes have the same position value. The difference only manifests for
    image/video tokens where T, H, W have distinct values.

    This subclass overrides ``forward()`` to implement ERNIE's interleaved
    layout while reusing the parent's ``inv_freq`` and infrastructure.
    """

    def __init__(self, freq_allocation: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.freq_allocation = freq_allocation

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section,
        cp_group=None,
    ) -> Tensor:
        """Compute ERNIE-style interleaved M-RoPE embeddings.

        Args:
            position_ids: [3, batch, seq_len] where axis 0=T, 1=H, 2=W
            mrope_section: Ignored (kept for API compatibility). ERNIE uses
                freq_allocation instead.
            cp_group: Context parallel group.

        Returns:
            Tensor: RoPE embedding of shape [seq_len, batch, 1, head_dim].
        """
        device = self.inv_freq.device
        seq = position_ids.to(device=device, dtype=self.inv_freq.dtype)
        # seq: [3, bs, seq_len]

        if self.seq_len_interpolation_factor is not None:
            seq = seq * (1.0 / self.seq_len_interpolation_factor)

        bs = seq.shape[1]
        seq_len = seq.shape[2]
        num_freqs = self.inv_freq.shape[0]  # head_dim // 2 = 64

        # Compute freqs for each axis: freqs[axis, bs, seq_len, num_freqs]
        # Each axis independently: theta_f = inv_freq[f] * pos[axis]
        inv_freq_exp = self.inv_freq[None, None, :, None].expand(3, bs, -1, 1)
        seq_exp = seq[:, :, None, :].float()
        freqs = (inv_freq_exp @ seq_exp).transpose(2, 3)
        # freqs: [3, bs, seq_len, num_freqs=64]

        # ERNIE interleaved layout:
        # For freq band f (0-indexed into inv_freq):
        #   f < (num_freqs - freq_allocation) AND f is even -> H (axis 1)
        #   f < (num_freqs - freq_allocation) AND f is odd  -> W (axis 2)
        #   f >= (num_freqs - freq_allocation)              -> T (axis 0)
        #
        # Build the combined freq tensor by selecting the right axis per band.
        hw_bands = num_freqs - self.freq_allocation  # 44
        # H bands: even indices 0,2,4,...,42 -> 22 bands
        h_freq_indices = torch.arange(0, hw_bands, 2, device=device)
        # W bands: odd indices 1,3,5,...,43 -> 22 bands
        w_freq_indices = torch.arange(1, hw_bands, 2, device=device)
        # T bands: last freq_allocation indices 44,...,63 -> 20 bands
        t_freq_indices = torch.arange(hw_bands, num_freqs, device=device)

        # Gather per-axis freqs for their assigned bands
        # freqs[axis]: [bs, seq_len, 64]
        h_freqs = freqs[1, :, :, h_freq_indices]  # [bs, seq_len, 22]
        w_freqs = freqs[2, :, :, w_freq_indices]  # [bs, seq_len, 22]
        t_freqs = freqs[0, :, :, t_freq_indices]  # [bs, seq_len, 20]

        # Interleave H and W: [H0, W0, H1, W1, ..., H21, W21] -> 44 values
        hw_interleaved = torch.stack([h_freqs, w_freqs], dim=-1).reshape(
            bs, seq_len, hw_bands
        )  # [bs, seq_len, 44]

        # Concatenate HW + T
        combined_freqs = torch.cat([hw_interleaved, t_freqs], dim=-1)
        # combined_freqs: [bs, seq_len, 64]

        # Apply interleaved doubling (matching rotary_interleaved=True):
        # Each freq band f expands to two consecutive head dims: [f, f]
        combined_flat = combined_freqs.reshape(bs, -1, 1)
        emb = torch.stack((combined_flat, combined_flat), dim=-1).reshape(
            bs, seq_len, -1
        )
        # emb: [bs, seq_len, 128]

        # Reshape to match MCore expected output: [seq_len, bs, 1, head_dim]
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        # emb: [seq_len, bs, 1, 128]

        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb


class Ernie45VLModel(MegatronModule):
    """
    ERNIE 4.5 VL MoE Model (Vision-Language with Mixture of Experts).

    This model combines:
    - A HuggingFace ERNIE 4.5 vision encoder (32-layer ViT with 2D RoPE)
    - A variable-resolution resampler (spatial + temporal merging)
    - A Megatron-Core GPT language model with heterogeneous dual-pool MoE

    The vision tower and resampler are borrowed directly from HuggingFace
    and replicated across TP ranks. The language model uses standard
    Megatron-Core distributed infrastructure.

    Args:
        config (GPTModelProvider): Language model provider configuration.
        pre_process (bool): Include embedding layer (used with pipeline parallelism).
        post_process (bool): Include output layer (used with pipeline parallelism).
        vp_stage (int, optional): Virtual pipeline stage index.
    """

    def __init__(
        self,
        config: GPTModelProvider,
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        # HF bound methods (get_image_features, get_video_features, etc.) access
        # self.config.return_dict via the @can_return_tuple decorator.
        # Ensure the attribute exists on the provider config.
        if not hasattr(config, "return_dict"):
            config.return_dict = True

        self.use_mg_vit = getattr(config, "use_mg_vit", False)

        if pre_process:
            # Normalize configs for compatibility with transformers-builtin
            # vision model and resampler classes.
            # 1. Vision config: DFNRopeVisionTransformerConfig may lack rms_norm_eps,
            #    intermediate_size, temporal_merge_size.
            _normalize_vision_config(config.vision_config, hf_config=config.hf_config)
            # 2. HF config: flat config (Thinking) lacks text_config sub-object
            #    that the resampler accesses as config.text_config.hidden_size, etc.
            _normalize_hf_config(config.hf_config)

            if self.use_mg_vit:
                # Megatron-Core native ViT: TP-native attention and MLP via
                # TransformerBlock with TE modules.
                vision_transformer_config = get_ernie_vision_config(
                    config.vision_config, megatron_config=config,
                )
                vision_layer_spec = get_ernie_vit_layer_spec()
                self.vision_model = ErnieVLVisionModel(
                    transformer_config=vision_transformer_config,
                    transformer_layer_spec=vision_layer_spec,
                )
                # Wrap MG ViT with an adapter that matches the HF
                # vision_tower interface (forward signature, spatial_merge_size
                # attribute) so the bound HF get_image/video_features methods
                # work transparently.
                self.vision_tower = _MgVitTowerAdapter(self.vision_model)
            else:
                # HF-wrapped ViT: replicated across TP ranks.
                self.vision_tower = Ernie4_5_VLMoeVisionTransformerPretrainedModel._from_config(
                    config.vision_config
                )
                # Ensure HF vision tower params are tracked for TP gradient sync
                hook_hf_module_setattr_for_tp_grad_sync(self.vision_tower)

            # Instantiate the HF resampler (spatial + temporal merging + projection).
            # The resampler is kept as an HF module regardless of use_mg_vit because
            # it is small, replicated, and already has bridge weight mappings.
            self.resampler_model = Ernie4_5_VLMoeVariableResolutionResamplerModel(config.hf_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.resampler_model)

        # Build the Megatron-Core GPT language model
        self.language_model = self.config.provide_language_model(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

        # Replace the default MultimodalRotaryEmbedding with ERNIE's custom
        # interleaved variant.  GPTModel.__init__ creates rotary_pos_emb as a
        # standard MultimodalRotaryEmbedding, but ERNIE 4.5 VL uses a non-standard
        # interleaved H/W frequency layout (see ErnieMultimodalRotaryEmbedding).
        if hasattr(self.language_model, "rotary_pos_emb") and isinstance(
            self.language_model.rotary_pos_emb, MultimodalRotaryEmbedding
        ):
            old_rope = self.language_model.rotary_pos_emb
            freq_allocation = getattr(config.hf_config, "freq_allocation", None)
            if freq_allocation is None:
                # Derive from mrope_section: last element is T (temporal) allocation.
                # Real model: mrope_section=[22,22,20] -> freq_allocation=20
                # Toy model:  mrope_section=[2,2,2]    -> freq_allocation=2
                mrope_section = getattr(config, "mrope_section", None)
                freq_allocation = mrope_section[-1] if mrope_section else 20
            self.language_model.rotary_pos_emb = ErnieMultimodalRotaryEmbedding(
                freq_allocation=freq_allocation,
                kv_channels=config.kv_channels,
                rotary_percent=1.0,
                rotary_interleaved=config.rotary_interleaved,
                seq_len_interpolation_factor=old_rope.seq_len_interpolation_factor,
                rotary_base=config.rotary_base,
            )

        # Required for finalize_model_grads and tied weights
        self.share_embeddings_and_output_weights = config.share_embeddings_and_output_weights
        self.shared_embedding_or_output_weight = self.language_model.shared_embedding_or_output_weight

        # Bind utility methods from HF Ernie4_5_VLMoeModel to this instance
        self.get_placeholder_mask = types.MethodType(Ernie4_5_VLMoeModel.get_placeholder_mask, self)
        self.get_image_features = types.MethodType(Ernie4_5_VLMoeModel.get_image_features, self)
        self.get_video_features = types.MethodType(Ernie4_5_VLMoeModel.get_video_features, self)
        self.get_rope_index = types.MethodType(Ernie4_5_VLMoeModel.get_rope_index, self)
        self.get_vision_position_ids = types.MethodType(Ernie4_5_VLMoeModel.get_vision_position_ids, self)

        if pre_process:
            # Register pixel normalization buffers for the vision encoder.
            # The ERNIE 4.5 VL processor outputs raw (unnormalized) pixel patches
            # with do_rescale=False, do_normalize=False.  Normalization is expected
            # to happen on-device before the ViT, matching the HF custom model's
            # vision_forward() + add_image_preprocess() logic.
            #
            # OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
            # OPENAI_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
            patch_size = getattr(config.vision_config, "patch_size", 14)
            pixels_per_patch = patch_size * patch_size  # 196 for patch_size=14

            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
            clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

            # Expand to match flattened patch layout: [C * patch_size^2] = [588]
            # Each channel's mean/std is repeated patch_size^2 times
            pixel_mean = clip_mean.repeat_interleave(pixels_per_patch)  # [588]
            pixel_std = clip_std.repeat_interleave(pixels_per_patch)    # [588]

            self.register_buffer("pixel_mean", pixel_mean, persistent=False)
            self.register_buffer("pixel_std", pixel_std, persistent=False)

    @property
    def decoder(self):
        """Expose language model decoder for mcore inference compatibility."""
        return getattr(self.language_model, "decoder", None)

    def set_input_tensor(self, input_tensor) -> None:
        """Set model chunk input tensor."""
        self.language_model.set_input_tensor(input_tensor)

    def _normalize_pixel_values(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Normalize raw pixel patches for the vision encoder.

        The ERNIE 4.5 VL processor outputs raw pixel patches (0-255 range,
        ``do_rescale=False, do_normalize=False``).  This method applies CLIP
        normalization on-device, matching the HF custom model's
        ``vision_forward()`` + ``add_image_preprocess()`` logic:

            pixel_values = pixel_values / 255.0
            pixel_values = (pixel_values - CLIP_MEAN) / CLIP_STD

        Args:
            pixel_values: Raw pixel patches [total_patches, C*patch_size^2].
                          Values in 0-255 range (any dtype).

        Returns:
            Normalized pixel patches in bfloat16, values in ~(-2, 2.5) range.
        """
        # Rescale: divide by 255 (in float32 for precision)
        pixel_values = pixel_values.to(torch.float32) * (1.0 / 255.0)
        # Normalize: (x - mean) / std using CLIP mean/std
        pixel_values = (pixel_values - self.pixel_mean.to(pixel_values.device)) / self.pixel_std.to(pixel_values.device)
        # Cast to bfloat16 for the ViT
        return pixel_values.to(torch.bfloat16)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        moe_mm_token_type_ids: Optional[torch.IntTensor] = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Forward pass for ERNIE 4.5 VL MoE.

        Args:
            input_ids: Token IDs [batch_size, seq_len].
            pixel_values: Image pixel values for the vision encoder.
            pixel_values_videos: Video pixel values for the vision encoder.
            image_grid_thw: Grid dimensions (T, H, W) per image [num_images, 3].
            video_grid_thw: Grid dimensions (T, H, W) per video [num_videos, 3].
            mm_token_type_ids: Token type IDs for M-RoPE computation (0=text, 1=image, 2=video).
            moe_mm_token_type_ids: Token type IDs for MoE routing (0=text, 1/2=vision).
            labels: Labels for language modeling loss.
            loss_mask: Mask for loss computation.
        """

        if self.pre_process:
            if inputs_embeds is None:
                # Get text embeddings from language model embedding layer
                inputs_embeds = self.language_model.embedding(
                    input_ids=input_ids, position_ids=None
                )  # [seq_len, batch, hidden]

                inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()  # [batch, seq_len, hidden]

            # Process images through vision tower + resampler
            if pixel_values is not None:
                # Normalize raw pixel patches from the processor (0-255 uint8-valued).
                # The custom ERNIE processor outputs do_rescale=False, do_normalize=False,
                # so normalization must happen on-device before the ViT.
                pixel_values = self._normalize_pixel_values(pixel_values)
                image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # Process videos through vision tower + resampler
            if pixel_values_videos is not None:
                pixel_values_videos = self._normalize_pixel_values(pixel_values_videos)
                video_embeds = self.get_video_features(
                    pixel_values_videos, video_grid_thw, return_dict=True
                ).pooler_output
                video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Transpose back to [seq_len, batch, hidden] for Megatron-Core
            inputs_embeds = inputs_embeds.transpose(1, 0)

            if self.config.sequence_parallel:
                inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # Compute 3D MRoPE position IDs on ALL pipeline stages
        # Each stage has input_ids and visual grid info from the data iterator
        #
        # The custom ERNIE processor marks IMAGE_START/IMAGE_END/VIDEO_START/VIDEO_END
        # boundary tokens as image/video type in token_type_ids (for MoE routing).
        # But get_rope_index expects these boundary tokens to be text type (0),
        # because it generates exactly T*H*W/merge^2 vision positions per image
        # group from grid_thw — boundary tokens should get sequential text positions.
        rope_mm_token_type_ids = mm_token_type_ids
        if mm_token_type_ids is not None and input_ids is not None:
            boundary_token_ids = [
                getattr(self.config, "image_start_token_id", 101304),
                getattr(self.config, "image_end_token_id", 101305),
                getattr(self.config, "video_start_token_id", 101306),
                getattr(self.config, "video_end_token_id", 101307),
            ]
            boundary_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in boundary_token_ids:
                boundary_mask |= (input_ids == tid)
            if boundary_mask.any():
                rope_mm_token_type_ids = mm_token_type_ids.clone()
                rope_mm_token_type_ids[boundary_mask] = 0

        position_ids, _rope_deltas = self.get_rope_index(
            input_ids,
            rope_mm_token_type_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask=None,
        )

        # Set moe_mm_token_type_ids in the module-level context so that
        # ErnieMultiTypeMoE layers can read it during forward.  This avoids
        # modifying Megatron-Core's TransformerBlock/TransformerLayer signatures.
        #
        # When Sequence Parallel (SP) is enabled, hidden_states entering each
        # TransformerLayer's MLP are scattered across TP ranks:
        #   hidden_states shape = [seq_len / tp_size, batch, hidden]
        # The moe_mm_token_type_ids must be sliced to match the local sequence
        # partition so that ErnieMultiTypeMoE sees the correct modality labels
        # for the tokens on this TP rank.
        sp_moe_mm_token_type_ids = moe_mm_token_type_ids
        if (
            moe_mm_token_type_ids is not None
            and self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        ):
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            full_seq_len = moe_mm_token_type_ids.shape[-1]
            local_seq_len = full_seq_len // tp_size
            start = tp_rank * local_seq_len
            end = start + local_seq_len
            sp_moe_mm_token_type_ids = moe_mm_token_type_ids[..., start:end].contiguous()

        set_moe_mm_token_type_ids(sp_moe_mm_token_type_ids)

        try:
            outputs = self.language_model.forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                runtime_gather_output=runtime_gather_output,
                packed_seq_params=packed_seq_params,
            )
        finally:
            # Always clear after forward to avoid holding stale references
            clear_moe_mm_token_type_ids()

        return outputs

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Args:
            freeze_language_model: Freeze the language model module.
            freeze_vision_model: Freeze the vision encoder (patch_embed + blocks).
            freeze_vision_projection: Freeze the resampler / projector.
        """
        modules = []

        if freeze_language_model and hasattr(self, "language_model") and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model:
            if hasattr(self, "vision_model") and self.vision_model is not None:
                # MG-native ViT: freeze the entire vision model
                modules.append(self.vision_model)
            elif hasattr(self, "vision_tower") and self.vision_tower is not None:
                # HF ViT: freeze patch_embed and blocks
                if hasattr(self.vision_tower, "patch_embed"):
                    modules.append(self.vision_tower.patch_embed)
                if hasattr(self.vision_tower, "blocks"):
                    modules.append(self.vision_tower.blocks)

        if freeze_vision_projection and hasattr(self, "resampler_model") and self.resampler_model is not None:
            modules.append(self.resampler_model)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
