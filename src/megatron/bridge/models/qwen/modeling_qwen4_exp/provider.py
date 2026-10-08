# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compatibility import for the declarative Qwen4-Exp configuration.

Construction now uses ``Qwen4ExpModelBuilder(config).build_model(pg_collection)``.
"""

from megatron.bridge.models.qwen.modeling_qwen4_exp.model_config import Qwen4ExpModelConfig


Qwen4ExpModelProvider = Qwen4ExpModelConfig
