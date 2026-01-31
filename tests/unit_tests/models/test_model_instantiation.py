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

from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.model_provider import (
    ModelProviderMixin,
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


class _Rank0Group:
    def size(self):
        return 1

    def rank(self):
        return 0


class _Rank1Group:
    def size(self):
        return 1

    def rank(self):
        return 1


class _PG:
    def __init__(self):
        self.pp = _Rank0Group()
        self.tp = _Rank0Group()
        self.cp = _Rank0Group()
        self.dp = _Rank0Group()
        self.dp_cp = _Rank0Group()
        self.expt_dp = _Rank0Group()


class MockModelProvider(ModelProviderMixin):
    """Mock ModelProviderMixin for testing."""

    def __init__(self, model_instance=None):
        self.model_instance = model_instance or MockMegatronModule()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide a mock model instance."""
        return self.model_instance


class TestCreateModel:
    """Test cases for _create_model function."""

    @patch("megatron.bridge.models.model_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.model_provider.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.model_provider.tensor_parallel")
    def test_create_model_single_pipeline(self, mock_tensor_parallel, mock_is_last, mock_is_first):
        """Test model creation with single pipeline stage."""
        # Create mock model and provider
        mock_model = MockMegatronModule()
        model_provider = MockModelProvider(mock_model)

        pg = _PG()
        result = _create_model(model_provider, ModelType.encoder_or_decoder, pg_collection=pg)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_or_decoder

    @patch("megatron.bridge.models.model_provider.is_pp_first_stage")
    @patch("megatron.bridge.models.model_provider.is_pp_last_stage")
    @patch("megatron.bridge.models.model_provider.tensor_parallel")
    def test_create_model_virtual_pipeline(self, mock_tensor_parallel, mock_is_last, mock_is_first):
        """Test model creation with virtual pipeline parallelism."""
        # Setup mocks
        mock_is_first.side_effect = [True, False]
        mock_is_last.side_effect = [False, True]

        # Create mock models and provider
        mock_models = [MockMegatronModule(), MockMegatronModule()]
        model_provider = Mock()
        model_provider.provide = Mock(side_effect=mock_models)

        # pg with pp size > 1 to trigger VPP logic
        class _PP:
            def size(self):
                return 2

            def rank(self):
                return 0

        pg = _PG()
        pg.pp = _PP()
        # Also set vp size on provider
        model_provider.virtual_pipeline_model_parallel_size = 2

        result = _create_model(model_provider, ModelType.encoder_or_decoder, pg_collection=pg)

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(model.model_type == ModelType.encoder_or_decoder for model in result)
        assert model_provider.provide.call_count == 2

    @patch("megatron.bridge.models.model_provider.tensor_parallel")
    def test_create_model_encoder_decoder_single_pipeline(self, mock_tensor_parallel):
        """Test creation of encoder-decoder model with single pipeline."""

        # Create mock model and provider
        mock_model = MockMegatronModule()
        model_provider = Mock()
        model_provider.provide = Mock(return_value=mock_model)

        result = _create_model(model_provider, ModelType.encoder_and_decoder, pg_collection=_PG())

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_and_decoder
        model_provider.provide.assert_called_once_with()  # No pre/post process args

    @patch("megatron.bridge.models.model_provider.tensor_parallel")
    def test_create_model_encoder_decoder_multi_pipeline(self, mock_tensor_parallel):
        """Test creation of encoder-decoder model with multiple pipeline stages."""

        # Create mock model and provider
        mock_model = MockMegatronModule()
        model_provider = Mock()
        model_provider.provide = Mock(return_value=mock_model)

        result = _create_model(model_provider, ModelType.encoder_and_decoder, pg_collection=_PG())

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is mock_model
        assert mock_model.model_type == ModelType.encoder_and_decoder
        model_provider.provide.assert_called_once_with()

    @patch("megatron.bridge.models.model_provider.tensor_parallel")
    def test_create_model_sets_tensor_parallel_attributes(self, mock_tensor_parallel):
        """Test that tensor parallel attributes are set on parameters."""
        # Create mock model with parameters
        mock_model = MockMegatronModule()
        model_provider = MockModelProvider(mock_model)

        _create_model(model_provider, ModelType.encoder_or_decoder, pg_collection=_PG())

        # Verify tensor parallel attributes are set
        # Check that the function was called for each parameter
        assert mock_tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes.call_count == len(
            list(mock_model.parameters())
        )


class TestDDPWrap:
    """Test cases for _ddp_wrap function."""

    @patch("megatron.bridge.models.model_provider.DistributedDataParallel")
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
            pg_collection=_PG(),
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

    @patch("megatron.bridge.models.model_provider.TorchFullyShardedDataParallel")
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
            pg_collection=_PG(),
        )

        # Assertions
        assert len(result) == 1
        mock_fsdp.assert_called_once()
        mock_fsdp_instance.broadcast_params.assert_not_called()

    def test_ddp_wrap_overlap_param_gather(self):
        """Test DDP wrapping with overlap_param_gather_with_optimizer_step."""
        with patch("megatron.bridge.models.model_provider.DistributedDataParallel") as mock_ddp:
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
                pg_collection=_PG(),
            )

            # Check that bucketing is disabled when overlap is True
            call_kwargs = mock_ddp.call_args.kwargs
            assert call_kwargs["disable_bucketing"]


class TestPrintNumParams:
    """Test cases for _print_num_params function."""

    @patch("builtins.print")
    def test_print_num_params_rank_zero(self, mock_print):
        """Test printing parameters when on data parallel rank 0."""
        # Create models with known parameter counts
        models = [MockMegatronModule(), MockMegatronModule()]

        # pg where dp and cp ranks are zero; customize tp/pp ranks to expected print
        class _TP:
            def rank(self):
                return 1

            def size(self):
                return 1

        class _PP:
            def rank(self):
                return 2

            def size(self):
                return 1

        pg = _PG()
        pg.tp = _TP()
        pg.pp = _PP()

        _print_num_params(models, pg_collection=pg)

        # Check print was called
        mock_print.assert_called_once()
        printed_text = mock_print.call_args[0][0]
        assert "number of parameters" in printed_text
        assert "(1, 2)" in printed_text  # tensor and pipeline ranks

    @patch("builtins.print")
    def test_print_num_params_non_zero_rank(self, mock_print):
        """Test that nothing is printed when not on data parallel rank 0."""
        models = [MockMegatronModule()]

        pg = _PG()
        pg.dp = _Rank1Group()
        _print_num_params(models, pg_collection=pg)

        # Check print was not called
        mock_print.assert_not_called()

    @patch("builtins.print")
    def test_print_num_params_non_zero_context_rank(self, mock_print):
        """Test that nothing is printed when not on context parallel rank 0."""
        models = [MockMegatronModule()]

        pg = _PG()
        pg.cp = _Rank1Group()
        _print_num_params(models, pg_collection=pg)

        # Check print was not called
        mock_print.assert_not_called()


class TestGetModel:
    """Test cases for get_model function."""

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        pg = _PG()
        result = get_model(model_provider, ddp_config, pg_collection=pg)

        # Assertions
        assert len(result) == 1
        mock_create_model.assert_called_once()
        # Ensure pg_collection was passed through
        assert "pg_collection" in mock_create_model.call_args.kwargs
        mock_print_params.assert_called_once()
        mock_ddp_wrap.assert_called_once()

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
    @patch("megatron.bridge.models.model_provider.Float16Module")
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        get_model(model_provider, ddp_config, fp16=True, pg_collection=_PG())

        # Assertions
        assert model_provider.fp16

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        get_model(model_provider, ddp_config, use_cpu_initialization=True, pg_collection=_PG())

        assert config.use_cpu_initialization

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
    def test_get_model_no_ddp_wrap(
        self,
        mock_get_model_config,
        mock_ddp_wrap,
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        result = get_model(model_provider, ddp_config, wrap_with_ddp=False, pg_collection=_PG())

        # Assertions - should return unwrapped model
        assert len(result) == 1
        assert result[0] is model

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        get_model(
            model_provider,
            ddp_config,
            use_torch_fsdp2=True,
            use_cpu_initialization=True,
            pg_collection=_PG(),
        )

        # Should not call cuda when FSDP2 with CPU init
        model.cuda.assert_not_called()

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider._ddp_wrap")
    @patch("megatron.bridge.models.model_provider.get_model_config")
    def test_get_model_pre_wrap_hook(
        self,
        mock_get_model_config,
        mock_ddp_wrap,
        mock_correct_amax,
        mock_print_params,
        mock_create_model,
    ):
        """Test get_model with a pre_wrap_hook."""
        # Setup mocks
        config = create_test_config()
        config.use_cpu_initialization = False
        config.init_model_with_meta_device = False
        config.fp16 = False
        config.bf16 = False

        mock_get_model_config.return_value = config
        model = MockMegatronModule(config)
        mock_create_model.return_value = [model]

        # Hook now operates on individual model modules, not the entire list
        pre_wrap_hook = Mock(return_value=None)

        mock_correct_amax.return_value = [model]
        mock_ddp_wrap.return_value = [model]

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        result = get_model(
            model_provider,
            ddp_config,
            pre_wrap_hook=pre_wrap_hook,
            use_cpu_initialization=True,
            pg_collection=_PG(),
        )

        # Assertions
        assert result == [model]
        mock_create_model.assert_called_once()
        # Hook should be called once with the model list
        pre_wrap_hook.assert_called_once_with([model])
        mock_ddp_wrap.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("megatron.bridge.models.model_provider._create_model")
    @patch("megatron.bridge.models.model_provider._print_num_params")
    @patch("megatron.bridge.models.model_provider.correct_amax_history_if_needed")
    @patch("megatron.bridge.models.model_provider.get_model_config")
    def test_get_model_with_meta_device(
        self, mock_get_model_config, mock_correct_amax, mock_print_params, mock_create_model
    ):
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

        model_provider = MockModelProvider(model)
        ddp_config = DistributedDataParallelConfig()

        with patch("megatron.bridge.models.model_provider._ddp_wrap") as mock_wrap:
            mock_correct_amax.return_value = [model]
            mock_wrap.return_value = [model]

            get_model(model_provider, ddp_config, init_model_with_meta_device=True, pg_collection=_PG())

            # Should not call cuda when meta device is used
            model.cuda.assert_not_called()

    def test_create_model_virtual_pipeline_with_encoder_decoder_raises(self):
        """Test that virtual pipeline with encoder-decoder raises assertion error."""
        model_provider = MockModelProvider()
        model_provider.virtual_pipeline_model_parallel_size = 2

        with pytest.raises(AssertionError) as excinfo:
            # Craft pg with pp size > 1 to trigger VPP branch for the assertion
            class _PP:
                def size(self):
                    return 2

                def rank(self):
                    return 0

            pg = _PG()
            pg.pp = _PP()
            _create_model(model_provider, ModelType.encoder_and_decoder, pg_collection=pg)

        assert "Interleaved schedule not supported" in str(excinfo.value)
