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

from unittest.mock import MagicMock

import torch
from megatron.core.inference.contexts import StaticInferenceContext

from megatron.bridge.inference.vlm._mcore_compat import InferenceMode
from megatron.bridge.inference.vlm.vlm_engine import VLMEngine


class TestVLMEngine:
    def test_generate(self):
        InferenceMode.unset_active()
        mock_controller = MagicMock()
        mock_controller.tokenize_prompt.return_value = ([1, 2, 3], "image_dict")
        # MCoreEngine/StaticInferenceEngine expects inference_context to be a StaticInferenceContext
        # (and uses inference_wrapper_config.inference_max_requests for scheduler batch size).
        mock_controller.inference_wrapped_model.inference_context = StaticInferenceContext(
            max_batch_size=128, max_sequence_length=8192
        )
        mock_controller.inference_wrapped_model.inference_wrapper_config = MagicMock(inference_max_requests=128)

        try:
            engine = VLMEngine(mock_controller, max_batch_size=4)
            engine.scheduler = MagicMock()
            engine.scheduler.add_request.return_value = "req_id"
            engine.scheduler.completed_request_pool = {"req_id": "result"}
            engine.run_engine = MagicMock()

            results = engine.generate(["prompt"], ["image"])

            assert results == ["result"]
            engine.scheduler.add_request.assert_called()
            engine.run_engine.assert_called()
            assert not InferenceMode.is_active()
        finally:
            InferenceMode.unset_active()

    def test_generate_uses_requested_seed_for_sampling(self):
        def sample_tokens(random_seed):
            InferenceMode.unset_active()
            mock_controller = MagicMock()
            mock_controller.tokenize_prompt.return_value = ([1, 2, 3], "image_dict")
            mock_controller.inference_wrapped_model.inference_context = StaticInferenceContext(
                max_batch_size=128, max_sequence_length=8192
            )
            mock_controller.inference_wrapped_model.inference_wrapper_config = MagicMock(inference_max_requests=128)
            mock_controller.sampling_rng = torch.Generator(device="cuda")
            mock_controller.sampling_rng.manual_seed(42)

            try:
                engine = VLMEngine(mock_controller, max_batch_size=4, random_seed=random_seed)
                engine.scheduler = MagicMock()
                engine.scheduler.add_request.return_value = "req_id"

                def run_engine():
                    probabilities = torch.full((2,), 0.5, device="cuda")
                    sampled_tokens = torch.multinomial(
                        probabilities,
                        num_samples=16,
                        replacement=True,
                        generator=mock_controller.sampling_rng,
                    )
                    engine.scheduler.completed_request_pool = {"req_id": sampled_tokens.cpu().tolist()}

                engine.run_engine = run_engine
                return engine.generate(["prompt"], ["image"])
            finally:
                InferenceMode.unset_active()

        first_sample = sample_tokens(random_seed=17)
        replayed_sample = sample_tokens(random_seed=17)
        different_sample = sample_tokens(random_seed=23)
        default_sample = sample_tokens(random_seed=None)
        config_seed_sample = sample_tokens(random_seed=42)

        assert first_sample == replayed_sample
        assert first_sample != different_sample
        assert default_sample == config_seed_sample
