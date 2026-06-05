# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the NemotronLabsDiffusion recipe inference script."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from examples.diffusion.recipes.nemotron_labs_diffusion import inference_nemotron_labs_diffusion as recipe


pytestmark = [pytest.mark.unit]


def test_load_model_uses_checkpoint_config_loader():
    args = SimpleNamespace(megatron_path="/checkpoints/model", tp=2)
    loaded_model = MagicMock()
    loaded_model.cuda.return_value = loaded_model
    loaded_model.eval.return_value = "ready"

    with patch("megatron.bridge.training.model_load_save.load_megatron_model", return_value=[loaded_model]) as load:
        assert recipe.load_model(args) == "ready"

    load.assert_called_once_with(
        "/checkpoints/model",
        skip_temp_dist_context=True,
        mp_overrides={
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "pipeline_dtype": recipe.torch.bfloat16,
        },
    )
