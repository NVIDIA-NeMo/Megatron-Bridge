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

import math

import torch

from megatron.core.models.huggingface.fastconformer.configuration_fastconformer import (
    FastConformerConfig,
)
from megatron.core.models.huggingface.fastconformer.modeling_fastconformer import (
    FastConformerModel,
)
from megatron.core.models.huggingface.module import HuggingFaceModule


class BridgeSoundEncoder(HuggingFaceModule):
    """Sound encoder wrapper for Bridge that wraps vlm2's FastConformerModel.

    Uses NeMo-compatible parameter naming (via vlm2's bundled FastConformerModel)
    to ensure Megatron-format checkpoints are compatible with vlm2 training scripts.

    The outer config carries fields required by LLaVAModel's sound interface
    (sound_model_type, sound_pad_to_clip_duration, sound_batch_split) plus the
    FastConformerConfig fields needed to build the inner model.

    Does NOT include a feature extractor -- input is pre-processed mel spectrograms
    of shape (batch, frames, mel_bins), not raw audio waveforms.
    """

    def __init__(self, config):
        super().__init__(config)
        fc_config = FastConformerConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_mel_bins=config.num_mel_bins,
            subsampling_factor=config.subsampling_factor,
            conv_kernel_size=getattr(config, "conv_kernel_size", 9),
            use_bias=getattr(config, "use_bias", False),
        )
        self.model = FastConformerModel(fc_config)

        # Cache subsampling parameters for output length computation.
        # These mirror FastConformerSubsamplingConv2D's constants.
        self._sub_num_layers = int(math.log2(config.subsampling_factor))
        self._sub_kernel_size = 3
        self._sub_stride = 2
        self._sub_padding = (self._sub_kernel_size - 1) // 2

    def forward(self, sound_clips, sound_length):
        output = self.model(
            input_features=sound_clips,
            input_lengths=sound_length,
        )
        embedding_lengths = self._compute_output_lengths(sound_length)
        return output.last_hidden_state, embedding_lengths

    def _compute_output_lengths(self, input_lengths):
        """Compute post-subsampling sequence lengths.

        Matches the calc_length logic in FastConformerSubsamplingConv2D:
        Conv2D subsampling with kernel_size=3, stride=2, padding=1, ceil_mode=False,
        repeated log2(subsampling_factor) times.
        """
        lengths = input_lengths.to(dtype=torch.float)
        all_paddings = self._sub_padding * 2
        for _ in range(self._sub_num_layers):
            lengths = torch.div(lengths + all_paddings - self._sub_kernel_size, self._sub_stride) + 1.0
            lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)
