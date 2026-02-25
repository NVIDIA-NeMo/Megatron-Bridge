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

from typing import List, Optional, Union

import torch
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from PIL.Image import Image


class VLMEngine(MCoreEngine):
    """VLM inference engine extending MCoreEngine with image support."""

    # pylint: disable=C0115,C0116
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Union[Image, List[Image]]]] = None,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[InferenceRequest]:
        # pylint: disable=C0115,C0116
        request_ids: List[str] = []

        if self.random_seed:
            torch.random.manual_seed(self.random_seed)

        if images is not None and len(images) != len(prompts):
            raise ValueError(f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})")

        for i in range(len(prompts)):
            prompt = prompts[i]
            image = images[i] if images is not None else None
            prompt_tokens, image_dict = self.controller.tokenize_prompt(prompt, image)

            # Reuse encoder_prompt from scheduler to pass image
            request_id = self.scheduler.add_request(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                encoder_prompt=image_dict,
                sampling_params=sampling_params,
            )
            request_ids.append(request_id)

        self.run_engine()

        result: List[InferenceRequest] = [
            self.scheduler.completed_request_pool[request_id] for request_id in request_ids
        ]
        return result
