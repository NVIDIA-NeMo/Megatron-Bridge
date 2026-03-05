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

"""Tests for setup_optimizer in optim.py."""

from unittest.mock import MagicMock, patch

from megatron.core.optimizer import OptimizerConfig, ParamGroupOverride, ParamKey

from megatron.bridge.training.config import SchedulerConfig


class TestSetupOptimizerMuP:
    """Tests for μP optimizer scaling in setup_optimizer."""

    def _make_optimizer_config(self, lr=1e-3, min_lr=1e-5, optimizer="adam"):
        return OptimizerConfig(optimizer=optimizer, lr=lr, min_lr=min_lr, bf16=True)

    def _make_scheduler_config(self):
        cfg = SchedulerConfig(lr_decay_iters=1000, lr_decay_style="cosine")
        cfg.lr_warmup_steps = 0
        cfg.lr_decay_steps = 1000
        cfg.wsd_decay_steps = None
        return cfg

    def _make_model_mock(self, use_mup=False, mup_width_mult=1.0):
        model = MagicMock()
        model_config = MagicMock()
        model_config.use_mup = use_mup
        model_config.mup_width_mult = mup_width_mult
        return model, model_config

    def _make_param_key(self):
        """Create a simple ParamKey instance for use in fake overrides."""
        return ParamKey(name="*.weight")

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_disabled_skips_overrides(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When use_mup=False, get_mup_config_overrides is not called."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=False)
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        with patch("megatron.bridge.training.optim.get_mup_config_overrides") as mock_mup:
            setup_optimizer(
                optimizer_config=self._make_optimizer_config(),
                scheduler_config=self._make_scheduler_config(),
                model=model,
            )
            mock_mup.assert_not_called()

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_enabled_calls_overrides(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When use_mup=True, get_mup_config_overrides is called with correct args."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=True, mup_width_mult=2.0)
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        fake_overrides = {self._make_param_key(): ParamGroupOverride(lr_mult=0.5)}

        with patch("megatron.bridge.training.optim.get_mup_config_overrides", return_value=fake_overrides) as mock_mup:
            optimizer_config = self._make_optimizer_config(lr=1e-3, optimizer="adam")
            setup_optimizer(
                optimizer_config=optimizer_config,
                scheduler_config=self._make_scheduler_config(),
                model=model,
            )
            mock_mup.assert_called_once_with(
                config=optimizer_config,
                mup_width_mult=2.0,
                optimizer_type="adam",
            )

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_overrides_merged_with_existing(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """μP overrides are merged with existing config_overrides."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=True, mup_width_mult=4.0)
        mock_get_model_config.return_value = model_config

        mup_key = ParamKey(name="*.weight")
        existing_key = ParamKey(name="*.bias")
        mup_overrides = {mup_key: ParamGroupOverride(lr_mult=0.25)}
        existing_overrides = {existing_key: ParamGroupOverride(wd_mult=0.0)}

        captured_overrides = {}

        def capture_optimizer_call(**kwargs):
            captured_overrides.update(kwargs.get("config_overrides") or {})
            return MagicMock()

        mock_get_optimizer.side_effect = capture_optimizer_call

        with patch("megatron.bridge.training.optim.get_mup_config_overrides", return_value=mup_overrides):
            with patch(
                "megatron.bridge.training.optim.OptimizerConfigOverrideProvider.build_config_overrides",
                return_value=existing_overrides,
            ):
                setup_optimizer(
                    optimizer_config=self._make_optimizer_config(),
                    scheduler_config=self._make_scheduler_config(),
                    model=model,
                )

        assert mup_key in captured_overrides
        assert existing_key in captured_overrides

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_model_list_uses_first_chunk(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When model is a list, get_model_config is called on the first chunk."""
        from megatron.bridge.training.optim import setup_optimizer

        model1, model_config = self._make_model_mock(use_mup=False)
        model2 = MagicMock()
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        setup_optimizer(
            optimizer_config=self._make_optimizer_config(),
            scheduler_config=self._make_scheduler_config(),
            model=[model1, model2],
        )

        mock_get_model_config.assert_called_once_with(model1)
