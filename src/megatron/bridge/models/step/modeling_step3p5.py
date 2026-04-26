# Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved.
#
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
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from .configuration_step3p5 import Step3p5Config

__all__ = ["MockStep3p5Model", "MockStep3p5ForCausalLM"]


class MockStep3p5PreTrainedModel(PreTrainedModel):
    # Link this model family to its configuration class so PreTrainedModel.from_pretrained
    # can load the config instead of failing with a NoneType error.
    config_class = Step3p5Config


class MockStep3p5Model(MockStep3p5PreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]
    config: Step3p5Config
    def __init__(self, config: Step3p5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size


class MockStep3p5ForCausalLM(MockStep3p5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config: Step3p5Config

    def __init__(self, config: Step3p5Config):
        super().__init__(config)
        self.model = Step3p5Model(config)
        self.lm_head = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.post_init()
