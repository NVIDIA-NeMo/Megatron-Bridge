# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""FLUX diffusion model implementation with Megatron Core."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import nn

from megatron.bridge.diffusion.models.flux.flux_layer_spec import (
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from megatron.bridge.diffusion.models.flux.layers import EmbedND, MLPEmbedder, TimeStepEmbedder


if TYPE_CHECKING:
    pass


class Flux(VisionModule):
    """
    FLUX diffusion model implementation with Megatron Core.

    FLUX is a state-of-the-art text-to-image diffusion model that uses
    a combination of double (MMDiT-style) and single transformer blocks.

    Args:
        config: FluxProvider containing model hyperparameters.

    Attributes:
        out_channels: Number of output channels.
        hidden_size: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        patch_size: Patch size for image embedding.
        in_channels: Number of input channels.
        guidance_embed: Whether guidance embedding is used.
        pos_embed: N-dimensional position embedding module.
        img_embed: Image embedding linear layer.
        txt_embed: Text embedding linear layer.
        timestep_embedding: Timestep embedding module.
        vector_embedding: Vector (CLIP pooled) embedding module.
        guidance_embedding: Guidance embedding module (if guidance_embed=True).
        double_blocks: List of MMDiT layers for double blocks.
        single_blocks: List of single transformer blocks.
        norm_out: Output normalization layer.
        proj_out: Output projection layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        **kwargs,
    ):
        super(Flux, self).__init__(config=config)

        self.config: TransformerConfig = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        self.out_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.guidance_embed = config.guidance_embed

        # Position embedding for rotary embeddings
        self.pos_embed = EmbedND(dim=self.hidden_size, theta=10000, axes_dim=config.axes_dims_rope)

        # Input embeddings
        self.img_embed = nn.Linear(config.in_channels, self.hidden_size)
        self.txt_embed = nn.Linear(config.context_dim, self.hidden_size)

        # Timestep and conditioning embeddings
        self.timestep_embedding = TimeStepEmbedder(config.model_channels, self.hidden_size)
        self.vector_embedding = MLPEmbedder(in_dim=config.vec_in_dim, hidden_dim=self.hidden_size)

        # Optional guidance embedding (for FLUX-dev)
        if config.guidance_embed:
            self.guidance_embedding = MLPEmbedder(in_dim=config.model_channels, hidden_dim=self.hidden_size)

        # Double blocks (MMDiT-style joint attention)
        self.double_blocks = nn.ModuleList(
            [
                MMDiTLayer(
                    config=config,
                    submodules=get_flux_double_transformer_engine_spec().submodules,
                    layer_number=i,
                    context_pre_only=False,
                )
                for i in range(config.num_joint_layers)
            ]
        )

        # Single blocks
        self.single_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    config=config,
                    submodules=get_flux_single_transformer_engine_spec().submodules,
                    layer_number=i,
                )
                for i in range(config.num_single_layers)
            ]
        )

        # Output layers
        self.norm_out = AdaLNContinuous(config=config, conditioning_embedding_dim=self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_size * self.patch_size * self.out_channels, bias=True)

    def get_fp8_context(self):
        """Get FP8 autocast context if FP8 is enabled."""
        if not self.config.fp8:
            fp8_context = nullcontext()
        else:
            # Import TE dependencies only when training in fp8
            from transformer_engine.common.recipe import (
                DelayedScaling,
                Float8BlockScaling,
                Float8CurrentScaling,
                Format,
                MXFP8BlockScaling,
            )
            from transformer_engine.pytorch import fp8_autocast

            if self.config.fp8 == "e4m3":
                fp8_format = Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            # Defaults to delayed scaling for backward compatibility
            if not self.config.fp8_recipe:
                self.config.fp8_recipe = "delayed"

            if self.config.fp8_recipe == "delayed":
                fp8_recipe = DelayedScaling(
                    margin=self.config.fp8_margin,
                    interval=self.config.fp8_interval,
                    fp8_format=fp8_format,
                    amax_compute_algo=self.config.fp8_amax_compute_algo,
                    amax_history_len=self.config.fp8_amax_history_len,
                    override_linear_precision=(False, False, not self.config.fp8_wgrad),
                )
            elif self.config.fp8_recipe == "current":
                fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "block":
                fp8_recipe = Float8BlockScaling(fp8_format=fp8_format)
            elif self.config.fp8_recipe == "mxfp8":
                fp8_recipe = MXFP8BlockScaling(fp8_format=fp8_format)
            else:
                raise ValueError(f"Unsupported FP8 recipe: {self.config.fp8_recipe}")

            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(with_context_parallel=True)
            fp8_context = fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group)
        return fp8_context

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor = None,
        y: torch.Tensor = None,
        timesteps: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        controlnet_double_block_samples: torch.Tensor = None,
        controlnet_single_block_samples: torch.Tensor = None,
    ):
        """
        Forward pass through the FLUX model.

        Args:
            img: Image input tensor (latents) [B, S, C].
            txt: Text input tensor (text embeddings) [B, S, D].
            y: Vector input for embedding (CLIP pooled output) [B, D].
            timesteps: Timestep input tensor [B].
            img_ids: Image position IDs for rotary embedding [B, S, 3].
            txt_ids: Text position IDs for rotary embedding [B, S, 3].
            guidance: Guidance input for conditioning (FLUX-dev) [B].
            controlnet_double_block_samples: Optional controlnet samples for double blocks.
            controlnet_single_block_samples: Optional controlnet samples for single blocks.

        Returns:
            Output tensor of shape [B, S, out_channels].
        """
        # Embed image and text
        hidden_states = self.img_embed(img)
        encoder_hidden_states = self.txt_embed(txt)

        # Timestep embedding
        timesteps = timesteps.to(img.dtype) * 1000
        vec_emb = self.timestep_embedding(timesteps)

        # Optional guidance embedding
        if guidance is not None:
            vec_emb = vec_emb + self.guidance_embedding(self.timestep_embedding.time_proj(guidance * 1000))

        # Add vector (CLIP pooled) embedding
        vec_emb = vec_emb + self.vector_embedding(y)

        # Compute rotary position embeddings
        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary_pos_emb = self.pos_embed(ids)

        # Process through double blocks (MMDiT)
        for id_block, block in enumerate(self.double_blocks):
            with self.get_fp8_context():
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )

                # Apply controlnet residuals if provided
                if controlnet_double_block_samples is not None:
                    interval_control = len(self.double_blocks) / len(controlnet_double_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_double_block_samples[id_block // interval_control]

        # Concatenate encoder and image hidden states for single blocks
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=0)

        # Process through single blocks
        for id_block, block in enumerate(self.single_blocks):
            with self.get_fp8_context():
                hidden_states, _ = block(
                    hidden_states=hidden_states,
                    rotary_pos_emb=rotary_pos_emb,
                    emb=vec_emb,
                )

                # Apply controlnet residuals if provided
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = torch.cat(
                        [
                            hidden_states[: encoder_hidden_states.shape[0]],
                            hidden_states[encoder_hidden_states.shape[0] :]
                            + controlnet_single_block_samples[id_block // interval_control],
                        ]
                    )

        # Extract image hidden states (remove text portion)
        hidden_states = hidden_states[encoder_hidden_states.shape[0] :, ...]

        # Output normalization and projection
        hidden_states = self.norm_out(hidden_states, vec_emb)
        output = self.proj_out(hidden_states)

        return output

    def sharded_state_dict(self, prefix="", sharded_offsets: tuple = (), metadata: dict = None) -> ShardedStateDict:
        """
        Get sharded state dict for distributed checkpointing.

        Args:
            prefix: Prefix for state dict keys.
            sharded_offsets: Sharded offsets tuple.
            metadata: Additional metadata.

        Returns:
            ShardedStateDict for the model.
        """
        sharded_state_dict = {}

        # Handle double blocks
        layer_prefix = f"{prefix}double_blocks."
        for layer in self.double_blocks:
            offset = layer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."
            sharded_prefix = f"{layer_prefix}{global_layer_offset}."
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Handle single blocks
        layer_prefix = f"{prefix}single_blocks."
        for layer in self.single_blocks:
            offset = layer._get_layer_offset(self.config)

            global_layer_offset = layer.layer_number
            state_dict_prefix = f"{layer_prefix}{global_layer_offset - offset}."
            sharded_prefix = f"{layer_prefix}{global_layer_offset}."
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(state_dict_prefix, sharded_pp_offset, metadata)
            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Handle other modules
        for name, module in self.named_children():
            if not (module is self.single_blocks or module is self.double_blocks):
                sharded_state_dict.update(
                    sharded_state_dict_default(module, f"{prefix}{name}.", sharded_offsets, metadata)
                )

        # Set replica IDs for embedding and output layers
        # These layers are replicated across tensor parallel ranks and need proper replica IDs
        replica_modules = ["img_embed", "txt_embed", "timestep_embedding", "vector_embedding", "proj_out"]
        if self.guidance_embed:
            replica_modules.append("guidance_embedding")

        for module_name in replica_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                for param_name, param in module.named_parameters():
                    weight_key = f"{prefix}{module_name}.{param_name}"
                    if weight_key in sharded_state_dict:
                        self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)

        return sharded_state_dict

    def _set_embedder_weights_replica_id(
        self, tensor: torch.Tensor, sharded_state_dict: ShardedStateDict, embedder_weight_key: str
    ) -> None:
        """Set replica IDs of the weights in embedding layers for sharded state dict.

        Args:
            tensor: The parameter tensor to set replica ID for.
            sharded_state_dict: State dict with the weight to tie.
            embedder_weight_key: Key of the weight in the state dict.

        Returns:
            None, acts in-place.
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vpp_rank = vpp_rank if vpp_rank else 0
        vpp_world = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        vpp_world = vpp_world if vpp_world else 1
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

        # Remove the existing entry and replace with properly configured sharded tensor
        del sharded_state_dict[embedder_weight_key]

        replica_id = (
            tp_rank,
            (vpp_rank + pp_rank * vpp_world),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[embedder_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=embedder_weight_key,
            replica_id=replica_id,
            allow_shape_mismatch=False,
        )

    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism."""
        pass
