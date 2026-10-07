# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Native DiffusionGemma model and provider (no checkpoint conversion)."""

from megatron.bridge.diffusion.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaModel
from megatron.bridge.diffusion.models.diffusion_gemma.provider import DiffusionGemmaModelProvider


__all__ = ["DiffusionGemmaModel", "DiffusionGemmaModelProvider"]
