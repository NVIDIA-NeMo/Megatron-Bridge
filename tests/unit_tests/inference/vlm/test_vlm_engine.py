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

from megatron.bridge.inference.vlm.vlm_engine import VLMEngine


class TestVLMEngine:
    def test_generate(self):
        mock_controller = MagicMock()
        mock_controller.tokenize_prompt.return_value = ([1, 2, 3], "image_dict")
        # Fix for TypeError: '>' not supported between instances of 'int' and 'MagicMock'
        mock_controller.inference_wrapped_model.context.max_batch_size = 128

        engine = VLMEngine(mock_controller, max_batch_size=4)
        engine.scheduler = MagicMock()
        engine.scheduler.add_request.return_value = "req_id"
        engine.scheduler.completed_request_pool = {"req_id": "result"}
        engine.run_engine = MagicMock()

        results = engine.generate(["prompt"], ["image"])

        assert results == ["result"]
        engine.scheduler.add_request.assert_called()
        engine.run_engine.assert_called()
