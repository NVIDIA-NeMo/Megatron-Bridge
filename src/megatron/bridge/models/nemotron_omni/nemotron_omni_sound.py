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

import torch
from megatron.core.transformer.module import MegatronModule
from transformers import ParakeetEncoder, ParakeetEncoderConfig


class BridgeSoundEncoder(MegatronModule):
    """Sound encoder wrapper for Bridge that wraps HF transformers' ParakeetEncoder.

    Uses the public ``ParakeetEncoder`` from ``transformers`` so that Megatron-side
    parameter names line up 1:1 with the Nemotron-Omni HF checkpoint's
    ``sound_encoder.encoder.*`` state dict.

    The outer config carries fields required by LLaVAModel's sound interface
    (sound_model_type, sound_pad_to_clip_duration, sound_batch_split) plus the
    ParakeetEncoderConfig fields needed to build the inner encoder.

    Does NOT include a feature extractor -- input is pre-processed mel spectrograms
    of shape (batch, frames, mel_bins), not raw audio waveforms.
    """

    def __init__(self, config):
        super().__init__(config=config)
        parakeet_config = ParakeetEncoderConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_mel_bins=config.num_mel_bins,
            subsampling_factor=config.subsampling_factor,
            conv_kernel_size=getattr(config, "conv_kernel_size", 9),
            attention_bias=getattr(config, "use_bias", False),
            convolution_bias=getattr(config, "use_bias", False),
            scale_input=False,
        )
        self.encoder = ParakeetEncoder(parakeet_config)

    def __setattr__(self, name, value):
        # Flag replicated params so finalize_model_grads all-reduces their grads
        # across tensor-parallel ranks, keeping the HF-replica weights in sync.
        super().__setattr__(name, value)
        if isinstance(value, torch.nn.Module):
            for param in value.parameters(recurse=True):
                setattr(param, "average_gradients_across_tp_domain", True)

    def set_input_tensor(self, input_tensor):
        """Dummy for pipeline parallel set_input_tensor hook."""
        self.input_tensor = input_tensor

    def forward(self, sound_clips, sound_length):
        max_frames = sound_clips.size(1)
        attention_mask = (
            torch.arange(max_frames, device=sound_clips.device)[None, :] < sound_length[:, None]
        )
        output = self.encoder(
            input_features=sound_clips,
            attention_mask=attention_mask,
        )
        embedding_lengths = self.encoder._get_subsampling_output_length(sound_length)
        return output.last_hidden_state, embedding_lengths
