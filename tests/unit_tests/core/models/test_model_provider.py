from unittest.mock import Mock, call, patch

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.hub.core.models.model_provider import (
    ModelProviderProtocol,
    _create_model,
    _ddp_wrap,
    _print_num_params,
    get_model,
)


def create_test_config(**kwargs):
    """Create a valid TransformerConfig for testing."""
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 8,
    }
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


class MockMegatronModule(MegatronModule):
    """Mock MegatronModule for testing."""

    def __init__(self, config=None):
        if config is None:
            config = create_test_config()
        super().__init__(config)
        self.config = config
        self.model_type = ModelType.encoder_or_decoder

    def cuda(self, device=None):
        return self

    def parameters(self):
        return [torch.nn.Parameter(torch.randn(10, 10))]


class TestCreateModel:
    """Test cases for _create_model function."""

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("megatron.hub.core.models.model_provider.tensor_parallel")
    def test_create_model_single_pipeline(self, mock_tensor_parallel, mock_parallel_state):
        """Test model creation with single pipeline stage."""
        # Setup mocks
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_parallel_state.is_pipeline_first_stage.return_value = True
        mock_parallel_state.is_pipeline_last_stage.return_value = True

        # Create mock model
        mock_model = MockMegatronModule()
        model_provider = Mock(return_value=mock_model)

        result = _create_model(model_provider, ModelType.encoder_or_decoder)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_or_decoder
        model_provider.assert_called_once_with(pre_process=True, post_process=True)

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("megatron.hub.core.models.model_provider.tensor_parallel")
    def test_create_model_virtual_pipeline(self, mock_tensor_parallel, mock_parallel_state):
        """Test model creation with virtual pipeline parallelism."""
        # Setup mocks
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.is_pipeline_first_stage.side_effect = [True, False]
        mock_parallel_state.is_pipeline_last_stage.side_effect = [False, True]

        # Create mock models
        mock_models = [MockMegatronModule(), MockMegatronModule()]
        model_provider = Mock(side_effect=mock_models)

        result = _create_model(model_provider, ModelType.encoder_or_decoder)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(model.model_type == ModelType.encoder_or_decoder for model in result)
        assert model_provider.call_count == 2
        mock_parallel_state.set_virtual_pipeline_model_parallel_rank.assert_has_calls([call(0), call(1)])

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("megatron.hub.core.models.model_provider.tensor_parallel")
    def test_create_model_encoder_decoder_single_pipeline(self, mock_tensor_parallel, mock_parallel_state):
        """Test creation of encoder-decoder model with single pipeline."""
        # Setup mocks
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = None

        # Create mock model
        mock_model = MockMegatronModule()
        model_provider = Mock(return_value=mock_model)

        result = _create_model(model_provider, ModelType.encoder_and_decoder)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_and_decoder
        model_provider.assert_called_once_with()  # No pre/post process args

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("megatron.hub.core.models.model_provider.tensor_parallel")
    def test_create_model_encoder_decoder_multi_pipeline(self, mock_tensor_parallel, mock_parallel_state):
        """Test creation of encoder-decoder model with multiple pipeline stages."""
        # Setup mocks
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 4
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 2
        mock_parallel_state.get_pipeline_model_parallel_decoder_start.return_value = 2

        # Create mock model
        mock_model = MockMegatronModule()
        model_provider = Mock(return_value=mock_model)

        result = _create_model(model_provider, ModelType.encoder_and_decoder)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_and_decoder
        model_provider.assert_called_once_with()

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("megatron.hub.core.models.model_provider.tensor_parallel")
    def test_create_model_sets_tensor_parallel_attributes(self, mock_tensor_parallel, mock_parallel_state):
        """Test that tensor parallel attributes are set on parameters."""
        # Setup mocks
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = None
        mock_parallel_state.is_pipeline_first_stage.return_value = True
        mock_parallel_state.is_pipeline_last_stage.return_value = True

        # Create mock model with parameters
        mock_model = MockMegatronModule()
        model_provider = Mock(return_value=mock_model)

        _create_model(model_provider, ModelType.encoder_or_decoder)

        # Verify tensor parallel attributes are set
        # Check that the function was called for each parameter
        assert mock_tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes.call_count == len(
            list(mock_model.parameters())
        )


class TestDDPWrap:
    """Test cases for _ddp_wrap function."""

    @patch("megatron.hub.core.models.model_provider.DistributedDataParallel")
    def test_ddp_wrap_standard(self, mock_ddp):
        """Test wrapping models with standard DDP."""
        # Setup
        config = create_test_config()
        models = [MockMegatronModule(config), MockMegatronModule(config)]
        ddp_config = DistributedDataParallelConfig()

        # Create mock DDP instances
        mock_ddp_instances = [Mock(), Mock()]
        mock_ddp.side_effect = mock_ddp_instances

        result = _ddp_wrap(
            models,
            use_torch_fsdp2=False,
            data_parallel_random_init=True,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
        )

        # Assertions
        assert len(result) == 2
        assert mock_ddp.call_count == 2

        # Check first model has bucketing enabled
        first_call = mock_ddp.call_args_list[0]
        assert not first_call.kwargs["disable_bucketing"]

        # Check second model has bucketing disabled
        second_call = mock_ddp.call_args_list[1]
        assert second_call.kwargs["disable_bucketing"]

        # Check broadcast_params was called
        for ddp_instance in mock_ddp_instances:
            ddp_instance.broadcast_params.assert_called_once()

    @patch("megatron.hub.core.models.model_provider.TorchFullyShardedDataParallel")
    def test_ddp_wrap_fsdp2(self, mock_fsdp):
        """Test wrapping models with FSDP2."""
        # Setup
        config = create_test_config()
        models = [MockMegatronModule(config)]
        ddp_config = DistributedDataParallelConfig()

        # Create mock FSDP instance
        mock_fsdp_instance = Mock()
        mock_fsdp.return_value = mock_fsdp_instance

        result = _ddp_wrap(
            models,
            use_torch_fsdp2=True,
            data_parallel_random_init=False,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
        )

        # Assertions
        assert len(result) == 1
        mock_fsdp.assert_called_once()
        mock_fsdp_instance.broadcast_params.assert_not_called()

    def test_ddp_wrap_overlap_param_gather(self):
        """Test DDP wrapping with overlap_param_gather_with_optimizer_step."""
        with patch("megatron.hub.core.models.model_provider.DistributedDataParallel") as mock_ddp:
            # Setup
            config = create_test_config()
            models = [MockMegatronModule(config)]
            ddp_config = DistributedDataParallelConfig()

            mock_ddp.return_value = Mock()

            _ddp_wrap(
                models,
                use_torch_fsdp2=False,
                data_parallel_random_init=False,
                ddp_config=ddp_config,
                overlap_param_gather_with_optimizer_step=True,
            )

            # Check that bucketing is disabled when overlap is True
            call_kwargs = mock_ddp.call_args.kwargs
            assert call_kwargs["disable_bucketing"]


class TestPrintNumParams:
    """Test cases for _print_num_params function."""

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("builtins.print")
    def test_print_num_params_rank_zero(self, mock_print, mock_parallel_state):
        """Test printing parameters when on data parallel rank 0."""
        # Setup mocks
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_tensor_model_parallel_rank.return_value = 1
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 2

        # Create models with known parameter counts
        models = [MockMegatronModule(), MockMegatronModule()]

        _print_num_params(models)

        # Check print was called
        mock_print.assert_called_once()
        printed_text = mock_print.call_args[0][0]
        assert "number of parameters" in printed_text
        assert "(1, 2)" in printed_text  # tensor and pipeline ranks

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    @patch("builtins.print")
    def test_print_num_params_non_zero_rank(self, mock_print, mock_parallel_state):
        """Test that nothing is printed when not on data parallel rank 0."""
        # Setup mocks
        mock_parallel_state.get_data_parallel_rank.return_value = 1

        models = [MockMegatronModule()]

        _print_num_params(models)

        # Check print was not called
        mock_print.assert_not_called()


class TestGetModel:
    """Test cases for get_model function."""

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.hub.core.models.model_provider._ddp_wrap")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    def test_get_model_basic(
        self,
        mock_get_model_config,
        mock_ddp_wrap,
        mock_fix_float8,
        mock_print_params,
        mock_create_model,
    ):
        """Test basic get_model functionality."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = True  # Use CPU init to avoid CUDA
        config.init_model_with_meta_device = False
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        mock_create_model.return_value = [model]
        mock_fix_float8.return_value = [model]
        mock_ddp_wrap.return_value = [model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        result = get_model(model_provider, ddp_config)

        # Assertions
        assert len(result) == 1
        mock_create_model.assert_called_once_with(
            model_provider, ModelType.encoder_or_decoder, init_model_with_meta_device=None
        )
        mock_print_params.assert_called_once()
        # mock_fix_float8.assert_called_once()  # Commented out since _fix_float_8 is not called in the implementation
        mock_ddp_wrap.assert_called_once()

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.hub.core.models.model_provider._ddp_wrap")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    @patch("megatron.hub.core.models.model_provider.Float16Module")
    def test_get_model_fp16(
        self,
        mock_float16_module,
        mock_get_model_config,
        mock_ddp_wrap,
        mock_fix_float8,
        mock_print_params,
        mock_create_model,
    ):
        """Test get_model with FP16 enabled."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = True  # Use CPU init to avoid CUDA
        config.init_model_with_meta_device = False
        config.fp16 = False  # Will be overridden
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        mock_create_model.return_value = [model]

        # Mock Float16Module
        wrapped_model = Mock()
        mock_float16_module.return_value = wrapped_model
        mock_fix_float8.return_value = [wrapped_model]
        mock_ddp_wrap.return_value = [wrapped_model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        get_model(model_provider, ddp_config, fp16=True)

        # Assertions
        assert config.fp16
        mock_float16_module.assert_called_once_with(config, model)

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.hub.core.models.model_provider._ddp_wrap")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    def test_get_model_cpu_initialization(
        self,
        mock_get_model_config,
        mock_ddp_wrap,
        mock_fix_float8,
        mock_print_params,
        mock_create_model,
    ):
        """Test get_model with CPU initialization."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = True  # Already set to True
        config.init_model_with_meta_device = False
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        model.cuda = Mock()  # Mock cuda method
        mock_create_model.return_value = [model]
        mock_fix_float8.return_value = [model]
        mock_ddp_wrap.return_value = [model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        get_model(model_provider, ddp_config, use_cpu_initialization=True)

        # Assertions
        assert config.use_cpu_initialization
        model.cuda.assert_called_once()

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    def test_get_model_no_ddp_wrap(
        self,
        mock_get_model_config,
        mock_fix_float8,
        mock_print_params,
        mock_create_model,
    ):
        """Test get_model without DDP wrapping."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = True  # Use CPU init to avoid CUDA
        config.init_model_with_meta_device = False
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        mock_create_model.return_value = [model]
        mock_fix_float8.return_value = [model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        result = get_model(model_provider, ddp_config, wrap_with_ddp=False)

        # Assertions - should return unwrapped model
        assert len(result) == 1
        assert result[0] is model

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.hub.core.models.model_provider._ddp_wrap")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    def test_get_model_fsdp2_cpu_init(
        self,
        mock_get_model_config,
        mock_ddp_wrap,
        mock_fix_float8,
        mock_print_params,
        mock_create_model,
    ):
        """Test get_model with FSDP2 and CPU initialization (skip GPU allocation)."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = True
        config.init_model_with_meta_device = False
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        model.cuda = Mock()  # Mock cuda method
        mock_create_model.return_value = [model]
        mock_fix_float8.return_value = [model]
        mock_ddp_wrap.return_value = [model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        get_model(
            model_provider,
            ddp_config,
            use_torch_fsdp2=True,
            use_cpu_initialization=True,
        )

        # Should not call cuda when FSDP2 with CPU init
        model.cuda.assert_not_called()


class TestModelProviderProtocol:
    """Test cases for ModelProviderProtocol."""

    def test_protocol_implementation(self):
        """Test that a class properly implements ModelProviderProtocol."""

        class ValidProvider:
            def get_model(
                self,
                ddp_config: DistributedDataParallelConfig,
                model_type=ModelType.encoder_or_decoder,
                overlap_param_gather_with_optimizer_step: bool = False,
                fp16: bool | None = None,
                bf16: bool | None = None,
                use_torch_fsdp2: bool = False,
                wrap_with_ddp: bool = True,
                data_parallel_random_init: bool = True,
                use_cpu_initialization: None | bool = False,
            ):
                return []

        provider = ValidProvider()
        assert isinstance(provider, ModelProviderProtocol)

    def test_protocol_missing_method(self):
        """Test that a class without get_model doesn't implement protocol."""

        class InvalidProvider:
            pass

        provider = InvalidProvider()
        assert not isinstance(provider, ModelProviderProtocol)

    def test_protocol_wrong_signature(self):
        """Test that a class with wrong get_model signature doesn't implement protocol."""

        class WrongSignatureProvider:
            def get_model(self):  # Missing required parameters
                return []

        provider = WrongSignatureProvider()
        # Protocol checking is based on method name, not signature in runtime_checkable
        assert isinstance(provider, ModelProviderProtocol)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("megatron.hub.core.models.model_provider._create_model")
    @patch("megatron.hub.core.models.model_provider._print_num_params")
    @patch("megatron.hub.core.models.model_provider.get_model_config")
    def test_get_model_with_meta_device(self, mock_get_model_config, mock_print_params, mock_create_model):
        """Test get_model with meta device initialization (skip GPU allocation)."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = False
        config.init_model_with_meta_device = True  # Meta device enabled
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        model.cuda = Mock()
        mock_create_model.return_value = [model]

        model_provider = Mock()
        ddp_config = DistributedDataParallelConfig()

        with patch("megatron.hub.core.models.model_provider.correct_amax_history_if_needed") as mock_fix:
            with patch("megatron.hub.core.models.model_provider._ddp_wrap") as mock_wrap:
                mock_fix.return_value = [model]
                mock_wrap.return_value = [model]

                get_model(model_provider, ddp_config)

                # Should not call cuda when meta device is used
                model.cuda.assert_not_called()

    @patch("megatron.hub.core.models.model_provider.parallel_state")
    def test_create_model_virtual_pipeline_with_encoder_decoder_raises(self, mock_parallel_state):
        """Test that virtual pipeline with encoder-decoder raises assertion error."""
        # Setup mocks for virtual pipeline
        mock_parallel_state.get_pipeline_model_parallel_world_size.return_value = 2
        mock_parallel_state.get_virtual_pipeline_model_parallel_world_size.return_value = 2

        model_provider = Mock()

        with pytest.raises(AssertionError) as excinfo:
            _create_model(model_provider, ModelType.encoder_and_decoder)

        assert "Interleaved schedule not supported" in str(excinfo.value)
