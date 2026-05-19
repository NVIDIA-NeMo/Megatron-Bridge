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

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.training.finetune import finetune
from tests.unit_tests.training.test_config import (
    create_test_checkpoint_config,
    create_test_config_container,
    create_test_gpt_config,
    restore_get_world_size_safe,
)


class TestFinetune:
    """Tests for the finetune function."""

    def test_finetune_fails_without_pretrained_checkpoint(self):
        """Test that finetune fails when config doesn't have pretrained_checkpoint or load set."""
        # Create a config with both pretrained_checkpoint and load as None
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(
            pretrained_checkpoint=None,
            load=None,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        # Create a mock forward step function
        mock_forward_step_func = Mock()

        try:
            # Assert that finetune raises AssertionError with the expected message
            with pytest.raises(
                AssertionError,
                match="Finetuning requires a loading from a pretrained checkpoint or resuming from a checkpoint",
            ):
                finetune(container, mock_forward_step_func)
        finally:
            restore_get_world_size_safe(og_ws, cfg_mod)

    def test_finetune_succeeds_with_pretrained_checkpoint(self):
        """Test that finetune succeeds when config has pretrained_checkpoint set."""
        # Create a config with pretrained_checkpoint set
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(
            pretrained_checkpoint="/path/to/pretrained/checkpoint",
            load=None,
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        # Create a mock forward step function
        mock_forward_step_func = Mock()

        # Mock the pretrain function to avoid actual training
        with patch("megatron.bridge.training.finetune.pretrain") as mock_pretrain:
            try:
                # This should not raise an AssertionError
                finetune(container, mock_forward_step_func)
                # Verify that pretrain was called with the correct arguments
                mock_pretrain.assert_called_once_with(container, mock_forward_step_func, callbacks=None)
            finally:
                restore_get_world_size_safe(og_ws, cfg_mod)

    def test_finetune_succeeds_with_load_checkpoint(self):
        """Test that finetune succeeds when config has load set."""
        # Create a config with load set
        gpt_model_cfg = create_test_gpt_config()
        checkpoint_cfg = create_test_checkpoint_config(
            pretrained_checkpoint=None,
            load="/path/to/load/checkpoint",
        )

        container, og_ws, cfg_mod = create_test_config_container(
            world_size_override=1,
            model_config=gpt_model_cfg,
            checkpoint_config=checkpoint_cfg,
        )

        # Create a mock forward step function
        mock_forward_step_func = Mock()

        # Mock the pretrain function to avoid actual training
        with patch("megatron.bridge.training.finetune.pretrain") as mock_pretrain:
            try:
                # This should not raise an AssertionError
                finetune(container, mock_forward_step_func)
                # Verify that pretrain was called with the correct arguments
                mock_pretrain.assert_called_once_with(container, mock_forward_step_func, callbacks=None)
            finally:
                restore_get_world_size_safe(og_ws, cfg_mod)
