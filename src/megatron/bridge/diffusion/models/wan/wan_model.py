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

# pylint: disable=C0115,C0116,C0301

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps
from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from torch import Tensor

from megatron.bridge.diffusion.models.common.dit_embeddings import ParallelTimestepEmbedding
from megatron.bridge.diffusion.models.wan.wan_layer_spec import (
    get_wan_block_with_transformer_engine_spec as WanLayerWithAdaLNspec,
)

from .rope_utils import Wan3DRopeEmbeddings


def sinusoidal_embedding_1d(dim, position):  # noqa: D103
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class Head(nn.Module):  # noqa: D101
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanModel(VisionModule):
    """
    WanModel is a VisionModule that implements a Wan model.
    Attributes:
        config (TransformerConfig): Configuration for the transformer.
        pre_process (bool): Whether to apply pre-processing steps.
        post_process (bool): Whether to apply post-processing steps.
        fp16_lm_cross_entropy (bool): Whether to use fp16 for cross-entropy loss.
        parallel_output (bool): Whether to use parallel output.
        transformer_decoder_layer_spec (WanLayerWithAdaLNspec): Specification for the transformer decoder layer.
        model_type (ModelType): Type of the model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        transformer_decoder_layer_spec=WanLayerWithAdaLNspec,
        **kwargs,
    ):
        super(WanModel, self).__init__(config=config)

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = transformer_decoder_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        self.num_heads = self.config.num_attention_heads
        self.freq_dim = self.config.freq_dim
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.patch_spatial = self.config.patch_spatial
        self.patch_temporal = self.config.patch_temporal
        self.patch_size = (self.patch_temporal, self.patch_spatial, self.patch_spatial)

        # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
        self.share_embeddings_and_output_weights = False

        ######################################
        ########## Wan architecture ##########

        # embeddings
        if self.pre_process:
            self.patch_embedding = nn.Conv3d(
                self.in_channels, self.config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size
            )

        self.text_embedding = nn.Sequential(
            nn.Linear(self.config.text_dim, self.config.crossattn_emb_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.config.crossattn_emb_size, self.config.crossattn_emb_size),
        )

        # As in diffuser's Wan implementation
        self.timesteps_proj = Timesteps(num_channels=self.freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = ParallelTimestepEmbedding(
            in_channels=self.freq_dim, time_embed_dim=self.config.hidden_size
        )
        self.time_proj_act_fn = nn.SiLU()
        self.time_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size * 6)

        self.rope_embeddings = Wan3DRopeEmbeddings(
            dim_head=self.config.hidden_size // self.num_heads, max_position_len=1024
        )

        # decoder blocks
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

        # output head
        if self.post_process:
            self.head = Head(self.config.hidden_size, self.out_channels, self.patch_size, eps=1e-6)

        # set attributes "average_gradients_across_tp_domain" for nn.Parameter objects
        # this is used for gradient averaging across TP domain with sequence parallelism
        self._mark_trainable_params_for_tp_grad_avg(
            [
                self.patch_embedding,
                self.text_embedding,
                self.time_embedder,
                self.time_proj,
                self.head,
            ]
        )

    def forward(
        self,
        x: Tensor,
        grid_sizes: list[Tuple[int, int, int]],
        t: Tensor,
        context: Tensor,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x List[Tensor]: list of vae encoded data (in_channel, f, h, w)
            grid_sizes List[Tuple[int, int, int]]: list of grid sizes (f, h, w)
            t Tensor: timesteps
            context List[Tensor]: list of context (text_len, hidden_size)
            packed_seq_params PackedSeqParams: packed sequence parameters

        Returns:
            Tensor: output tensor (still patchified) of shape [seq_len, batch_size, hidden_size]
        """
        #################################
        ########## Wan forward ##########

        # ============= embedders =============

        # run input embedding
        if self.pre_process:
            # x.shape [s, b, c * pF * pH * pW]
            seq_len, batch_size, _ = x.shape
            c = self.out_channels
            pF, pH, pW = self.patch_size
            x = x.reshape(seq_len * batch_size, pF, pH, pW, c)  # output: x.shape [s * b, pF, pH, pW, c]
            x = x.permute(0, 4, 1, 2, 3)  # output: x.shape [s * b, c, pF, pH, pW]
            x = self.patch_embedding(x)  # output: x.shape [s * b, hidden_size, 1, 1, 1]
            x = x.flatten(1)  # output: x.shape [s * b, hidden_size]
            x = x.reshape(seq_len, batch_size, -1)  # output: x.shape [s, b, hidden_size]

            # split sequence for sequence_parallel
            # TODO: for PP, do we move scatter_to_sequence_parallel_region here or after "x = self.decoder.input_tensor" ???
            if self.config.sequence_parallel:
                x = tensor_parallel.scatter_to_sequence_parallel_region(
                    x
                )  # output: x.shape [s * b // tp_size, hidden_size]

        else:
            # intermediate stage of pipeline
            x = self.decoder.input_tensor

        # time embeddings
        e = self.time_embedder(self.timesteps_proj(t).to(x.dtype))
        e0 = self.time_proj(self.time_proj_act_fn(e)).unflatten(1, (6, self.config.hidden_size))

        # context embeddings
        context = self.text_embedding(context)  # shape [text_len, b, hidden_size]

        # ============= decoder =============
        # calculate rotary pos emb
        n_head, dim_head = self.num_heads, self.config.hidden_size // self.num_heads
        cu_seqlens_q_padded = packed_seq_params["self_attention"].cu_seqlens_q_padded
        rotary_pos_emb = self.rope_embeddings(
            n_head, dim_head, cu_seqlens_q_padded, grid_sizes, t.device
        )  # output: rotary_pos_emb.shape [s, b, 1, dim_head]

        # run decoder
        x = self.decoder(
            hidden_states=x,
            attention_mask=e0,
            context=context,
            context_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=packed_seq_params,
        )

        # return if not post_process
        if not self.post_process:
            return x

        # head
        x = x.transpose(0, 1)  # head expects shape [b, s, hidden_size]
        x = self.head(x, e)  # output: x.shape [b, s, c * pF * pH * pW]
        x = x.transpose(0, 1)  # reshape back to shape [s, b, c * pF * pH * pW]

        # gather outputs for sequence_parallel
        # Note: in GPT models, because the vocab projection matrix is ColumnParallelLinear, the sequence is
        #   automatically gathered in ColumnParallelLinear forward pass.
        #   However, in Wan models, we need to gather the outputs manually.
        if self.config.sequence_parallel:
            x = tensor_parallel.gather_from_sequence_parallel_region(x)
        return x  # output: x.shape [s, b, c * pF * pH * pW]

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, "input_tensor should only be length 1 for gpt/bert"
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = "module.", sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        # Ensure replica ids for non-transformer embedder weights include pipeline dimension
        for module in ["text_embedding", "time_embedding", "time_projection"]:
            if hasattr(self, module):
                for param_name, param in getattr(self, module).named_parameters():
                    weight_key = f"{prefix}{module}.{param_name}"
                    if weight_key in sharded_state_dict:
                        self._set_embedder_weights_replica_id(param, sharded_state_dict, weight_key)

        return sharded_state_dict

    def _mark_trainable_params_for_tp_grad_avg(self, modules: Optional[list] = None) -> None:
        """Mark selected modules' trainable parameters to average gradients across TP domain."""
        target_modules = modules if modules is not None else [self]
        for module in target_modules:
            for _name, param in module.named_parameters(recurse=True):
                if isinstance(param, nn.Parameter) and param.requires_grad:
                    setattr(param, "average_gradients_across_tp_domain", True)

    def _set_embedder_weights_replica_id(
        self, tensor: Tensor, sharded_state_dict: ShardedStateDict, embedder_weight_key: str
    ) -> None:
        """set replica ids of the weights in t_embedder for sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            weight_key (str): key of the weight in the state dict.
                This entry will be replaced with a tied version

        Returns: None, acts in-place
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
        vpp_rank = vpp_rank if vpp_rank else 0
        vpp_world = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        vpp_world = vpp_world if vpp_world else 1
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
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
