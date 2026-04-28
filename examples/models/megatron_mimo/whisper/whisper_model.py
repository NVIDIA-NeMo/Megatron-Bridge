# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from typing import Optional, Union

import torch
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


try:
    from megatron.core.extensions.transformer_engine import TENorm

    NORM_IMPL = TENorm
except ImportError:
    NORM_IMPL = torch.nn.LayerNorm


def _sinusoidal_position_embedding(max_len: int, d_model: int) -> torch.Tensor:
    """Compute sinusoidal positional embeddings matching HF Whisper's sinusoids().

    Layout: [sin_all..., cos_all...] (NOT interleaved), with
    log_timescale_increment = log(10000) / (d_model // 2 - 1).
    """
    half = d_model // 2
    log_timescale_increment = math.log(10000.0) / (half - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(half).float())
    scaled_time = torch.arange(max_len).float().unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class WhisperEncoder(MegatronModule):
    """Megatron-native Whisper audio encoder.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_post_impl (ModuleSpec or type): Specifies the layer norm type for the final layer norm.
        num_mel_bins (int): Number of mel-frequency bins in the input spectrogram.
        max_source_positions (int): Maximum number of encoder positions (before conv downsampling).
        pg_collection (ProcessGroupCollection): Model communication process groups.
        vp_stage (int): Virtual pipeline stage.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        ln_post_impl: Union[ModuleSpec, type] = NORM_IMPL,
        num_mel_bins: int = 80,
        max_source_positions: int = 1500,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:  # pragma: no cover
        super().__init__(config=transformer_config)

        if has_config_logger_enabled(transformer_config):
            log_config_to_disk(transformer_config, locals(), prefix=type(self).__name__)

        self.visual_hidden_size = transformer_config.hidden_size
        self.num_mel_bins = num_mel_bins
        self.max_source_positions = max_source_positions
        self.pg_collection = pg_collection
        self.vp_stage = vp_stage

        # Conv feature extractor: two 1-D convolutions (not TP-sharded).
        # conv1: (num_mel_bins -> d_model, kernel=3, stride=1, padding=1)
        # conv2: (d_model -> d_model, kernel=3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv1d(
            in_channels=num_mel_bins,
            out_channels=self.visual_hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=self.visual_hidden_size,
            out_channels=self.visual_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Sinusoidal positional embedding (frozen nn.Embedding for ckpt compat w/ CLIP pattern).
        self.position_embeddings = torch.nn.Embedding(max_source_positions, self.visual_hidden_size)
        self.position_embeddings.weight.data.copy_(
            _sinusoidal_position_embedding(max_source_positions, self.visual_hidden_size)
        )
        self.position_embeddings.weight.requires_grad = False

        # Final layer norm (applied after transformer blocks).
        self.ln_post = build_module(
            ln_post_impl,
            config=transformer_config,
            hidden_size=self.visual_hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )

        self.model_type = ModelType.encoder_or_decoder

        # Transformer layers.
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function of the Whisper Encoder.

        Args:
            input_features (torch.Tensor): Mel spectrogram of shape [batch, num_mel_bins, mel_frames].
            attention_mask (torch.Tensor with dtype=bool): Attention mask to use.
            seq_lengths (torch.Tensor, optional): Per-sample number of valid encoder
                output tokens (shape [batch]).  When provided, only the first
                ``seq_lengths[i]`` tokens of each sample are kept and the rest
                (padding-derived) are dropped.  The return shape becomes
                ``[total_valid_tokens, hidden]`` instead of ``[batch, seq, hidden]``.

        Returns:
            x (torch.Tensor): Encoder output.  Shape is ``[b, s, h]`` when
                *seq_lengths* is ``None``, or ``[total_valid_tokens, h]`` when
                *seq_lengths* is given.
        """
        # Conv feature extraction.
        x = torch.nn.functional.gelu(self.conv1(input_features))  # [B, H, mel_frames]
        x = torch.nn.functional.gelu(self.conv2(x))  # [B, H, mel_frames // 2]

        x = x.permute(0, 2, 1)  # [B, S, H]  where S = mel_frames // 2 (after stride-2 conv)

        # Add sinusoidal positional embedding (truncated to actual sequence length).
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embeddings(position_ids)

        x = x.permute(1, 0, 2)  # [B, S, H] -> [S, B, H]
        # `permute` can make the tensor non-contiguous, breaking pipelining.
        x = x.contiguous()

        x = self.decoder(x, attention_mask)
        x = x.permute(1, 0, 2)  # [S, B, H] -> [B, S, H]
        x = x.contiguous()
        x = self.ln_post(x)

        if seq_lengths is not None:
            # Drop encoder outputs that correspond to padding frames.
            max_len = x.shape[1]
            valid_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
            x = x[valid_mask]  # [total_valid_tokens, H]

        return x
