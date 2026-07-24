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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model import Exaone45VisionModel


pytestmark = pytest.mark.unit


def test_default_process_groups_are_stored_on_model():
    config = SimpleNamespace(hidden_size=64, num_attention_heads=4, patch_size=14)
    process_groups = SimpleNamespace(tp=Mock(), cp=Mock())

    with (
        patch(
            "megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model.VisionModule.__init__",
            return_value=None,
        ),
        patch(
            "megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model."
            "ProcessGroupCollection.use_mpu_process_groups",
            return_value=process_groups,
        ),
        patch("megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model.Exaone45VisionPatchEmbed"),
        patch("megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model.Exaone45VisionRotaryEmbedding"),
        patch("megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model.Exaone45VisionTransformerBlock"),
        patch("megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model.Exaone45VisionPatchMerger"),
    ):
        model = Exaone45VisionModel(config, Mock(), Mock(), pg_collection=None)

    assert model.pg_collection is process_groups
    assert model.tp_group is process_groups.tp
    assert model.cp_group is process_groups.cp


def _make_freeze_test_model() -> Exaone45Model:
    model = Exaone45Model.__new__(Exaone45Model)
    torch.nn.Module.__init__(model)
    model.language_model = torch.nn.Module()
    model.language_model.dense = torch.nn.Linear(2, 2)
    model.language_model.mtp = torch.nn.Linear(2, 2)
    model.vision_model = None
    return model


def test_freeze_mtp_model_without_freezing_language_model():
    model = _make_freeze_test_model()

    model.freeze(
        freeze_language_model=False,
        freeze_vision_model=False,
        freeze_vision_projection=False,
        freeze_mtp_model=True,
    )

    assert all(param.requires_grad for param in model.language_model.dense.parameters())
    assert not any(param.requires_grad for param in model.language_model.mtp.parameters())


def test_language_freeze_can_leave_mtp_trainable():
    model = _make_freeze_test_model()

    model.freeze(
        freeze_language_model=True,
        freeze_vision_model=False,
        freeze_vision_projection=False,
        freeze_mtp_model=False,
    )

    assert not any(param.requires_grad for param in model.language_model.dense.parameters())
    assert all(param.requires_grad for param in model.language_model.mtp.parameters())
