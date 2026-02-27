# Copyright (c) 2025, NVIDIA CORPORATION.
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

from unittest.mock import Mock, call, patch

import pytest
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.model_provider import ModelProviderMixin


class MockMegatronModule(MegatronModule):
    """Mock MegatronModule for testing."""

    def __init__(self, config=None):
        if config is None:
            config = TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=1)
        super().__init__(config)


class TestProvider(ModelProviderMixin):
    """A concrete implementation of ModelProviderMixin for testing."""

    def provide(self, pre_process=None, post_process=None) -> MockMegatronModule:
        return MockMegatronModule()


@pytest.fixture
def provider():
    """Fixture to create a TestProvider instance."""
    return TestProvider()


@pytest.fixture
def ddp_config():
    """Fixture to create a DistributedDataParallelConfig instance."""
    return DistributedDataParallelConfig()


@patch("megatron.bridge.models.model_provider.ProcessGroupCollection.use_mpu_process_groups")
@patch("megatron.bridge.models.model_provider.get_model")
@patch("megatron.bridge.models.model_provider.torch.distributed")
@patch("megatron.bridge.models.model_provider.parallel_state.is_initialized", return_value=True)
def test_provide_distributed_model_with_hooks_as_args(
    mock_ps_init, mock_dist, mock_get_model, mock_use_pg, provider, ddp_config
):
    """Test that provide_distributed_model calls hooks passed as arguments."""
    mock_dist.is_initialized.return_value = True
    mock_model = [MockMegatronModule()]
    mock_get_model.return_value = mock_model

    pre_hook = Mock(return_value=mock_model)
    post_hook = Mock(return_value=mock_model)

    # Attach minimal pg_collection required by provider.provide
    provider._pg_collection = type(
        "PG",
        (),
        {"pp": object(), "tp": object(), "cp": object(), "dp": object(), "dp_cp": object(), "expt_dp": object()},
    )()

    # Return a stub pg_collection to avoid requiring real initialization
    class _PG:
        def __init__(self):
            self.pp = object()
            self.tp = object()
            self.cp = object()
            self.dp = object()
            self.dp_cp = object()
            self.expt_dp = object()

    mock_use_pg.return_value = _PG()

    provider.provide_distributed_model(
        ddp_config=ddp_config, pre_wrap_hook=pre_hook, post_wrap_hook=post_hook, wrap_with_ddp=False
    )

    mock_get_model.assert_called_once()
    # Check that the argument hook is passed directly to get_model
    assert mock_get_model.call_args.kwargs["pre_wrap_hook"] is pre_hook
    # Check that the argument hook is called after get_model
    post_hook.assert_called_once_with(mock_model)


@patch("megatron.bridge.models.model_provider.ProcessGroupCollection.use_mpu_process_groups")
@patch("megatron.bridge.models.model_provider.get_model")
@patch("megatron.bridge.models.model_provider.torch.distributed")
@patch("megatron.bridge.models.model_provider.parallel_state.is_initialized", return_value=True)
def test_provide_distributed_model_with_registered_hooks(
    mock_ps_init, mock_dist, mock_get_model, mock_use_pg, provider, ddp_config
):
    """Test that provide_distributed_model uses hooks registered on the instance."""
    mock_dist.is_initialized.return_value = True
    mock_model = [MockMegatronModule()]
    mock_get_model.return_value = mock_model

    pre_hook = Mock(return_value=mock_model)
    post_hook = Mock(return_value=mock_model)

    provider.register_pre_wrap_hook(pre_hook)
    provider.register_post_wrap_hook(post_hook)

    # stub pg
    class _PG:
        def __init__(self):
            self.pp = object()
            self.tp = object()
            self.cp = object()
            self.dp = object()
            self.dp_cp = object()
            self.expt_dp = object()

    mock_use_pg.return_value = _PG()
    provider.provide_distributed_model(ddp_config=ddp_config, wrap_with_ddp=False)

    mock_get_model.assert_called_once()

    # Check that the composed hook from the property was passed to get_model
    passed_pre_hook = mock_get_model.call_args.kwargs["pre_wrap_hook"]
    assert callable(passed_pre_hook)

    # Execute the passed hook and verify our mock was called
    passed_pre_hook(mock_model)
    pre_hook.assert_called_once_with(mock_model)

    # Check that the registered post-hook was called
    post_hook.assert_called_once_with(mock_model)


@patch("megatron.bridge.models.model_provider.ProcessGroupCollection.use_mpu_process_groups")
@patch("megatron.bridge.models.model_provider.get_model")
@patch("megatron.bridge.models.model_provider.torch.distributed")
@patch("megatron.bridge.models.model_provider.parallel_state.is_initialized", return_value=True)
def test_arg_hook_overrides_registered_hook(
    mock_ps_init, mock_dist, mock_get_model, mock_use_pg, provider, ddp_config
):
    """Test that argument hooks override registered instance hooks."""
    mock_dist.is_initialized.return_value = True
    mock_model = [MockMegatronModule()]
    mock_get_model.return_value = mock_model

    # Register hooks on the instance
    registered_pre_hook = Mock()
    registered_post_hook = Mock()
    provider.register_pre_wrap_hook(registered_pre_hook)
    provider.register_post_wrap_hook(registered_post_hook)

    # Pass different hooks as arguments
    arg_pre_hook = Mock(return_value=mock_model)
    arg_post_hook = Mock(return_value=mock_model)

    class _PG:
        def __init__(self):
            self.pp = object()
            self.tp = object()
            self.cp = object()
            self.dp = object()
            self.dp_cp = object()
            self.expt_dp = object()

    mock_use_pg.return_value = _PG()
    provider.provide_distributed_model(
        ddp_config=ddp_config, pre_wrap_hook=arg_pre_hook, post_wrap_hook=arg_post_hook, wrap_with_ddp=False
    )

    mock_get_model.assert_called_once()

    # The argument hook should be passed, not the composed hook
    assert mock_get_model.call_args.kwargs["pre_wrap_hook"] is arg_pre_hook

    # The argument hook should be called
    arg_post_hook.assert_called_once_with(mock_model)
    # The registered hook should NOT be called
    registered_post_hook.assert_not_called()


@patch("megatron.bridge.models.model_provider.ProcessGroupCollection.use_mpu_process_groups")
@patch("megatron.bridge.models.model_provider.get_model")
@patch("megatron.bridge.models.model_provider.torch.distributed")
@patch("megatron.bridge.models.model_provider.parallel_state.is_initialized", return_value=True)
def test_pg_collection_attached_to_provider(
    mock_ps_init, mock_dist, mock_get_model, mock_use_pg, provider, ddp_config
):
    """Process group collection should be available on the provider."""
    mock_dist.is_initialized.return_value = True
    mock_model = [MockMegatronModule()]
    mock_get_model.return_value = mock_model

    class _PG:
        def __init__(self):
            self.pp = object()
            self.tp = object()
            self.cp = object()
            self.dp = object()
            self.dp_cp = object()
            self.expt_dp = object()

    pg_instance = _PG()
    mock_use_pg.return_value = pg_instance

    provider.provide_distributed_model(ddp_config=ddp_config, wrap_with_ddp=False)

    assert provider._pg_collection is pg_instance


def test_hook_registration_and_composition(provider):
    """Test hook registration order and composition."""
    # Initially, no hooks are registered
    assert provider.pre_wrap_hook is None
    assert provider.post_wrap_hook is None

    # Register hooks
    pre_hook_1 = Mock(side_effect=lambda x: x)
    pre_hook_2 = Mock(side_effect=lambda x: x)
    pre_hook_3_prepended = Mock(side_effect=lambda x: x)

    provider.register_pre_wrap_hook(pre_hook_1)
    provider.register_pre_wrap_hook(pre_hook_2)
    provider.register_pre_wrap_hook(pre_hook_3_prepended, prepend=True)

    post_hook_1 = Mock(side_effect=lambda x: x)
    provider.register_post_wrap_hook(post_hook_1)

    # Check that properties return a callable
    assert callable(provider.pre_wrap_hook)
    assert callable(provider.post_wrap_hook)

    # Create a mock model to test the composed hook
    mock_model = [MockMegatronModule()]

    # Set up a manager to track hook calls before executing
    manager = Mock()
    manager.attach_mock(pre_hook_3_prepended, "pre_3")
    manager.attach_mock(pre_hook_1, "pre_1")
    manager.attach_mock(pre_hook_2, "pre_2")

    # Execute the composed pre-wrap hook
    provider.pre_wrap_hook(mock_model)

    # Check that the hooks were called in the correct order
    expected_calls = [
        call.pre_3(mock_model),
        call.pre_1(mock_model),
        call.pre_2(mock_model),
    ]
    assert manager.mock_calls == expected_calls

    # Execute the composed post-wrap hook
    provider.post_wrap_hook(mock_model)
    post_hook_1.assert_called_once_with(mock_model)
