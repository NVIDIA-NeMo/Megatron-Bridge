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

"""Regression tests for attention state left by Nemotron checkpoint fixtures."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from megatron.core import parallel_state, rerun_state_machine
from megatron.core.transformer import utils as transformer_utils
from megatron.core.transformer.enums import AttnBackend

from tests.functional_tests import utils
from tests.functional_tests.test_groups.recipes import test_nemotronh_recipes_finetune as fixtures


@pytest.mark.unit
@pytest.mark.skipif(
    not hasattr(transformer_utils, "set_attention_backend"),
    reason="Requires MCore's standalone attention-backend helper",
)
@pytest.mark.parametrize(
    "fixture_class,fixture_name",
    [
        (fixtures.TestNemotron3NanoFinetuneRecipes, "nemotron_3_nano_megatron_checkpoint"),
        (fixtures.TestNemotron3SuperFinetuneRecipes, "nemotron_3_super_megatron_checkpoint"),
    ],
)
def test_checkpoint_conversion_does_not_constrain_finetune_attention(
    monkeypatch, tmp_path, tmp_path_factory, fixture_class, fixture_name
):
    """A completed conversion must allow the next model to select fused attention."""
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(utils, "initialize_distributed", Mock())
    monkeypatch.setattr(utils, "broadcast_path", lambda path: path)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(rerun_state_machine, "destroy_rerun_state_machine", Mock())

    def import_checkpoint(**kwargs):
        transformer_utils.set_attention_backend(
            SimpleNamespace(
                attention_backend=AttnBackend.auto,
                batch_invariant_mode=False,
                flash_attention_version=None,
            )
        )

    monkeypatch.setattr(fixtures.AutoBridge, "import_ckpt", import_checkpoint)
    fixture = getattr(fixture_class, fixture_name).__wrapped__
    fixture(fixture_class(), "toy-checkpoint", tmp_path_factory, tmp_path)

    transformer_utils.set_attention_backend(
        SimpleNamespace(
            attention_backend=AttnBackend.fused,
            batch_invariant_mode=False,
            flash_attention_version=None,
        )
    )
