# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Opt-in native DiffusionGemma recipes; no global recipe registration."""

from megatron.bridge.diffusion.recipes.diffusion_gemma.sft import diffusion_gemma_sft_config


__all__ = ["diffusion_gemma_sft_config"]
