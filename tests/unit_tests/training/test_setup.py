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


from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from megatron.bridge.data.builders import GPTSFTDatasetConfig
from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.bridge.training.checkpointing import load_checkpoint
from megatron.bridge.training.setup import (
    _bind_dataset_provider_context,
    _build_distributed_model,
    _register_pre_wrap_hook,
    _should_load_checkpoint,
    _update_model_config_funcs,
    _validate_and_set_vocab_size,
    maybe_log_and_save_config,
)
from megatron.bridge.training.state import GlobalState


def _make_transformer(**kwargs):
    defaults = dict(num_layers=2, hidden_size=128, num_attention_heads=1)
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_gpt_model_config(**kwargs):
    tc_kwargs = kwargs.pop("transformer_kwargs", {})
    defaults = dict(transformer=_make_transformer(**tc_kwargs), vocab_size=32000)
    defaults.update(kwargs)
    return GPTModelConfig(**defaults)


def _make_checkpoint_source_config(
    *,
    load,
    pretrained_checkpoint=None,
    required=True,
    non_persistent_ckpt_type=None,
    non_persistent_global_ckpt_dir=None,
    peft=None,
):
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            load=load,
            pretrained_checkpoint=pretrained_checkpoint,
            non_persistent_ckpt_type=non_persistent_ckpt_type,
            non_persistent_global_ckpt_dir=non_persistent_global_ckpt_dir,
            finetune=False,
        ),
        peft=peft,
        _checkpoint_load_required=required,
    )


def test_gpt_sft_config_receives_tokenizer_through_builder_binding_without_mutation():
    config = GPTSFTDatasetConfig(seq_length=128, dataset_root="/tmp/data")
    tokenizer = object()
    received = []

    def provider(samples, dataset_config, tokenizer=None):
        received.append((samples, dataset_config, tokenizer))

    bound_provider = _bind_dataset_provider_context(
        provider,
        tokenizer=tokenizer,
        pg_collection=object(),
    )
    bound_provider([1, 0, 0], config)

    assert received == [([1, 0, 0], config, tokenizer)]
    assert not hasattr(config, "tokenizer")


class TestShouldLoadCheckpoint:
    """Tests for checkpoint source detection at setup time."""

    @patch("megatron.bridge.training.setup.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.setup.checkpoint_exists", return_value=False)
    def test_required_checkpoint_must_exist(self, _mock_exists, _mock_is_hf):
        cfg = _make_checkpoint_source_config(load="/missing/checkpoint")
        checkpoint_manager = SimpleNamespace(checkpointing_context={})

        with pytest.raises(FileNotFoundError, match="Finetuning requires loading from an available"):
            _should_load_checkpoint(cfg, checkpoint_manager)

    @patch("megatron.bridge.training.setup.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.setup.checkpoint_exists", return_value=False)
    def test_local_checkpoint_satisfies_required_source(self, _mock_exists, _mock_is_hf):
        cfg = _make_checkpoint_source_config(load="/missing/global/checkpoint")
        local_manager = Mock()
        local_manager.find_latest.return_value = 12
        checkpoint_manager = SimpleNamespace(checkpointing_context={"local_checkpoint_manager": local_manager})

        assert _should_load_checkpoint(cfg, checkpoint_manager) is True

    def test_global_non_persistent_checkpoint_reaches_checkpoint_loader(self, tmp_path):
        load_dir = tmp_path / "checkpoints"
        non_persistent_dir = load_dir / "non_persistent"
        non_persistent_dir.mkdir(parents=True)
        (non_persistent_dir / "latest_train_state.pt").touch()
        cfg = _make_checkpoint_source_config(
            load=str(load_dir),
            required=False,
            non_persistent_ckpt_type="global",
        )
        checkpoint_manager = SimpleNamespace(checkpointing_context={})

        assert _should_load_checkpoint(cfg, checkpoint_manager) is True

    @patch("megatron.bridge.training.checkpointing._load_checkpoint_from_path", return_value=(12, 0))
    def test_peft_resume_prefers_global_non_persistent_checkpoint(self, mock_load, tmp_path):
        load_dir = tmp_path / "checkpoints"
        non_persistent_dir = load_dir / "non_persistent"
        non_persistent_dir.mkdir(parents=True)
        (non_persistent_dir / "latest_train_state.pt").touch()
        pretrained_dir = tmp_path / "pretrained"
        pretrained_dir.mkdir()
        (pretrained_dir / "latest_train_state.pt").touch()
        cfg = _make_checkpoint_source_config(
            load=str(load_dir),
            pretrained_checkpoint=str(pretrained_dir),
            non_persistent_ckpt_type="global",
            peft=object(),
        )

        load_checkpoint(SimpleNamespace(cfg=cfg), [], None, None)

        assert mock_load.call_args.args[0] == str(load_dir)
        assert cfg.checkpoint.finetune is False

    @patch("megatron.bridge.training.setup.checkpoint_exists", return_value=False)
    def test_hf_load_reaches_checkpoint_loader_for_targeted_error(self, _mock_exists):
        cfg = _make_checkpoint_source_config(load="/hf/full-model")
        checkpoint_manager = SimpleNamespace(checkpointing_context={})

        with patch(
            "megatron.bridge.training.setup.is_hf_checkpoint_dir",
            side_effect=lambda path: path == "/hf/full-model",
        ):
            assert _should_load_checkpoint(cfg, checkpoint_manager) is True

    @patch("megatron.bridge.training.setup.is_hf_checkpoint_dir", return_value=False)
    @patch("megatron.bridge.training.setup.checkpoint_exists", return_value=False)
    def test_missing_checkpoint_remains_optional_for_pretraining(self, _mock_exists, _mock_is_hf):
        cfg = _make_checkpoint_source_config(load="/missing/checkpoint", required=False)
        checkpoint_manager = SimpleNamespace(checkpointing_context={})

        assert _should_load_checkpoint(cfg, checkpoint_manager) is False


class TestValidateAndSetVocabSize:
    """Test cases for the _validate_and_set_vocab_size function."""

    def test_vocab_size_none_uses_tokenizer_vocab_size(self):
        """Test that None vocab_size uses tokenizer's vocab size and enables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=None,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 32004
        assert should_pad_vocab is True

    def test_vocab_size_smaller_than_tokenizer_raises_error(self):
        """Test that vocab_size smaller than tokenizer raises ValueError."""
        with pytest.raises(ValueError, match="cannot be smaller than tokenizer's vocab_size"):
            _validate_and_set_vocab_size(
                model_vocab_size=30000,
                tokenizer_vocab_size=32004,
            )

    def test_vocab_size_larger_than_tokenizer_returns_same_value(self):
        """Test that vocab_size larger than tokenizer returns the same value and disables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=40960,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 40960
        assert should_pad_vocab is False

    def test_vocab_size_equal_to_tokenizer_returns_same_value(self):
        """Test that vocab_size equal to tokenizer returns the same value and disables padding."""
        vocab_size, should_pad_vocab = _validate_and_set_vocab_size(
            model_vocab_size=32004,
            tokenizer_vocab_size=32004,
        )
        assert vocab_size == 32004
        assert should_pad_vocab is False


class TestMaybeLogAndSaveConfig:
    """Tests for maybe_log_and_save_config."""

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=0)
    def test_rank_zero_saves_and_logs(self, mock_get_rank, tmp_path):
        filepath = tmp_path / "config.yaml"
        cfg = Mock()
        cfg.logger.save_config_filepath = str(filepath)
        cfg.to_yaml = Mock()
        cfg.log_non_default_values = Mock()

        maybe_log_and_save_config(cfg)

        cfg.to_yaml.assert_called_once_with(str(filepath))
        cfg.log_non_default_values.assert_called_once()

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=1)
    def test_non_zero_rank_noop(self, mock_get_rank):
        cfg = Mock()
        cfg.logger.save_config_filepath = "unused"
        cfg.to_yaml = Mock()
        cfg.log_non_default_values = Mock()

        maybe_log_and_save_config(cfg)

        cfg.to_yaml.assert_not_called()
        cfg.log_non_default_values.assert_not_called()

    @patch("megatron.bridge.training.setup.get_rank_safe", return_value=0)
    @patch("megatron.bridge.training.setup.print_rank_0")
    def test_save_failure_is_logged(self, mock_print, mock_get_rank):
        cfg = Mock()
        cfg.logger.save_config_filepath = "path"

        def raise_io_error(_):
            raise IOError("boom")

        cfg.to_yaml.side_effect = raise_io_error
        cfg.log_non_default_values = Mock()

        maybe_log_and_save_config(cfg)

        # Check that error was logged via print_rank_0
        mock_print.assert_called_once()
        assert "Error saving config" in mock_print.call_args[0][0]
        cfg.log_non_default_values.assert_called_once()


class TestRegisterPreWrapHook:
    """Test cases for the _register_pre_wrap_hook function."""

    def test_register_hook_on_model_config(self):
        """Test that a hook is appended to pre_wrap_hooks on a ModelConfig instance."""
        cfg = _make_gpt_model_config()
        hook = lambda models: models  # noqa: E731
        _register_pre_wrap_hook(cfg, hook)
        assert hook in cfg.pre_wrap_hooks

    def test_register_hook_on_provider(self):
        """Test that register_pre_wrap_hook is called on a provider-style object."""
        mock_provider = MagicMock(spec=GPTModelProvider)
        hook = lambda models: models  # noqa: E731
        _register_pre_wrap_hook(mock_provider, hook)
        mock_provider.register_pre_wrap_hook.assert_called_once_with(hook)


class TestBuildDistributedModel:
    """Test cases for the _build_distributed_model function."""

    def _make_cfg_with_model_config(self):
        """Create a mock cfg whose .model is a real GPTModelConfig."""
        model_cfg = _make_gpt_model_config()
        cfg = MagicMock()
        cfg.model = model_cfg
        cfg.ddp = MagicMock()
        cfg.optimizer.overlap_param_gather_with_optimizer_step = False
        cfg.dist.use_megatron_fsdp = False
        cfg.dist.use_torch_fsdp2 = False
        cfg.rng.data_parallel_random_init = False
        return cfg, model_cfg

    @patch("megatron.bridge.training.setup.GPTModelConfig")
    def test_build_with_model_config(self, _mock_gpt_cls):
        """Test that builder.build_distributed_models is called for ModelConfig."""
        cfg, model_cfg = self._make_cfg_with_model_config()

        mock_builder_cls = MagicMock()
        mock_builder = MagicMock()
        mock_builder_cls.return_value = mock_builder

        mock_dist_model = [MagicMock()]
        mock_builder.build_distributed_models.return_value = mock_dist_model

        with patch.object(type(model_cfg), "get_builder_cls", return_value=mock_builder_cls):
            result = _build_distributed_model(cfg, pg_collection=MagicMock())

        mock_builder.build_distributed_models.assert_called_once()
        assert result == mock_dist_model

    def test_build_with_provider(self):
        """Test that provide_distributed_model is called for providers."""
        mock_provider = MagicMock()
        # Ensure isinstance(mock_provider, ModelConfig) is False
        mock_provider.__class__ = type("FakeProvider", (), {})

        mock_dist_model = [MagicMock()]
        mock_provider.provide_distributed_model.return_value = mock_dist_model

        cfg = MagicMock()
        cfg.model = mock_provider
        cfg.ddp = MagicMock()
        cfg.optimizer.overlap_param_gather_with_optimizer_step = False
        cfg.dist.use_megatron_fsdp = False
        cfg.dist.use_torch_fsdp2 = False
        cfg.rng.data_parallel_random_init = False

        result = _build_distributed_model(cfg, pg_collection=MagicMock())

        mock_provider.provide_distributed_model.assert_called_once()
        assert result == mock_dist_model


def test_restart_rebinds_overlap_callbacks_to_rebuilt_model():
    """A restart must replace callbacks bound to the discarded model."""

    class FakeDDP:
        def no_sync(self):
            pass

        def start_grad_sync(self):
            pass

    model_config = _make_gpt_model_config()
    transformer_config = model_config.transformer
    ddp_config = SimpleNamespace(
        overlap_grad_reduce=True,
        overlap_param_gather=False,
        align_param_gather=False,
    )
    first_model = FakeDDP()
    rebuilt_model = FakeDDP()
    state = GlobalState()
    state._cfg = SimpleNamespace(model=model_config)

    with patch("megatron.bridge.training.setup.DistributedDataParallel", FakeDDP):
        _update_model_config_funcs([first_model], transformer_config, ddp_config, optimizer=None)
        assert transformer_config.no_sync_func.__self__ is first_model

        state.reset_for_restart()
        _update_model_config_funcs([rebuilt_model], transformer_config, ddp_config, optimizer=None)

    assert transformer_config.no_sync_func.__self__ is rebuilt_model
