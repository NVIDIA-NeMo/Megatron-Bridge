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

from typing import Optional

import torch
from megatron.core import InferenceParams
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from torch.nn import functional as F

from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_block import Exaone45VisionTransformerBlock
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import Exaone45TransformerConfig
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.utils import (
    Exaone45VisionPatchEmbed,
    Exaone45VisionPatchMerger,
    Exaone45VisionRotaryEmbedding,
)


class Exaone45VisionModel(VisionModule):
    """EXAONE 4.5 ViT vision model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        patch_merger_spec (ModuleSpec): Specifies module to use for transformer layers.
    """

    def __init__(
        self,
        transformer_config: Exaone45TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_merger_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        pg_collection: Optional[ProcessGroupCollection] = None,
        class_token_len: int = 1,
        patch_dim: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        spatial_patch_size: int = 14,
        img_h: int = 336,
        img_w: int = 336,
        window_size: int = 112,
    ) -> None:
        super().__init__(config=transformer_config)

        self.transformer_config = transformer_config
        self.class_token_len = class_token_len
        self.visual_hidden_size = transformer_config.hidden_size
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.window_size = window_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_patch_size = spatial_patch_size
        self.merge_hidden_size = self.visual_hidden_size * (spatial_merge_size**2)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = 3

        self.patch_size = transformer_config.patch_size
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = self.pg_collection.cp

        self.patch_embed = Exaone45VisionPatchEmbed(transformer_config)
        head_dim = transformer_config.hidden_size // transformer_config.num_attention_heads
        self.rotary_pos_emb = Exaone45VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        self.decoder = Exaone45VisionTransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
            pg_collection=self.pg_collection,
        )

        self.merger = None
        if self.post_process:
            self.merger = Exaone45VisionPatchMerger(
                transformer_config,
                patch_merger_spec,
                use_postshuffle_norm=False,
            )

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw):
        # pylint: disable=C0115,C0116
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the EXAONE 4.5 vision model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        assert grid_thw is not None
        assert self.input_tensor is None
        assert inference_params is None

        hidden_states = self.patch_embed(hidden_states)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        hidden_states = hidden_states.unsqueeze(1)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :]

        packed_seq_params_full = self.get_packed_seq_params(grid_thw)
        packed_seq_params = self.get_packed_seq_params(None, cu_window_seqlens)

        hidden_states = self.decoder(
            hidden_states=hidden_states,
            attention_mask=None,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            packed_seq_params_full=packed_seq_params_full,
            **(extra_block_kwargs or {}),
        )
        hidden_states = self.merger(hidden_states)

        # Restore original token order after window-based shuffling.
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices]

        # Encodes images into continuous embeddings that can be forwarded to the language model.
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        hidden_states = torch.split(hidden_states, split_sizes)
        hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states

    def get_packed_seq_params(
        self,
        grid_thw: Optional[torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # pylint: disable=C0115,C0116
        from megatron.core.packed_seq_params import PackedSeqParams

        if grid_thw is not None:
            seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
            cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            cu_seqlens = cu_seqlens.squeeze()
        else:
            cu_seqlens = cu_seqlens.squeeze()
            seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        max_seqlen = seqlens.max().item()

        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

    def get_window_index(self, grid_thw):
        # pylint: disable=C0115,C0116
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0

        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_dim

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens
