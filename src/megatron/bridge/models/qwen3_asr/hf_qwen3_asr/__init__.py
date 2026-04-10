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

# Why we vendor the HF model code here instead of `pip install qwen3-asr`:
#
# 1. Dependency conflict – the official qwen3-asr package pins its own
#    transformers / torch versions in pyproject.toml
#    (https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml#L23),
#    which conflicts with the versions required by Megatron-LM.
#
# 2. Missing source on HF Hub – the Qwen3-ASR HuggingFace model repo only
#    ships weights and config; it does not include the modeling / processing
#    source code, so `trust_remote_code=True` alone is not enough to load
#    the model without installing the package.
#
# To work around both issues we keep a local copy of the HF model code and
# register the Auto classes ourselves below.

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig, Qwen3ASRThinkerConfig
from .modeling_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
