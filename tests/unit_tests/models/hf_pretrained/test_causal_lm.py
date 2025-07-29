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

"""
Comprehensive tests for PreTrainedCausalLM class.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import PreTrainedTokenizer

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


class TestPreTrainedCausalLMInitialization:
    """Test initialization and configuration of PreTrainedCausalLM."""

    @patch("torch.cuda.is_available")
    def test_init_minimal(self, mock_cuda):
        """Test minimal initialization."""
        mock_cuda.return_value = False  # Mock CUDA as not available
        lm = PreTrainedCausalLM()

        assert lm._model_name_or_path is None
        assert lm.device == "cpu"  # Default when CUDA not available
        assert lm.torch_dtype is None
        assert lm.trust_remote_code is False
        assert lm.init_kwargs == {}

        # Lazy loaded components should not exist yet
        assert not hasattr(lm, "_config")
        assert not hasattr(lm, "_tokenizer")
        assert not hasattr(lm, "_model")
        assert not hasattr(lm, "_generation_config")

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        model_path = "gpt2"
        lm = PreTrainedCausalLM(model_name_or_path=model_path)

        assert lm._model_name_or_path == model_path
        assert lm.model_name_or_path == model_path

    def test_init_with_device(self):
        """Test initialization with specific device."""
        lm = PreTrainedCausalLM(device="cuda:0")
        assert lm.device == "cuda:0"

        lm2 = PreTrainedCausalLM(device=torch.device("cpu"))
        assert lm2.device == torch.device("cpu")

    def test_init_with_dtype(self):
        """Test initialization with specific dtype."""
        lm = PreTrainedCausalLM(torch_dtype=torch.float16)
        assert lm.torch_dtype == torch.float16

    def test_init_with_trust_remote_code(self):
        """Test initialization with trust_remote_code."""
        lm = PreTrainedCausalLM(trust_remote_code=True)
        assert lm.trust_remote_code is True

    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs."""
        kwargs = {"use_fast": True, "padding_side": "left"}
        lm = PreTrainedCausalLM(**kwargs)
        assert lm.init_kwargs == kwargs

    def test_from_pretrained_classmethod(self):
        """Test from_pretrained class method."""
        model_path = "gpt2"
        lm = PreTrainedCausalLM.from_pretrained(
            model_path,
            device="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_fast=True,
        )

        assert lm._model_name_or_path == model_path
        assert lm.device == "cuda"
        assert lm.torch_dtype == torch.float16
        assert lm.trust_remote_code is True
        assert lm.init_kwargs == {"use_fast": True}

    @patch("torch.cuda.is_available")
    def test_cuda_device_selection(self, mock_cuda):
        """Test automatic CUDA device selection."""
        mock_cuda.return_value = True
        lm = PreTrainedCausalLM()
        assert lm.device == "cuda"

        mock_cuda.return_value = False
        lm2 = PreTrainedCausalLM()
        assert lm2.device == "cpu"


class TestPreTrainedCausalLMConfigProperty:
    """Test config property and lazy loading."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoConfig.from_pretrained")
    def test_config_lazy_load(self, mock_from_pretrained, mock_config):
        """Test config is lazy loaded on first access."""
        mock_from_pretrained.return_value = mock_config

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        # Config should not be loaded yet
        assert not hasattr(lm, "_config")
        mock_from_pretrained.assert_not_called()

        # Access config
        config = lm.config

        # Now it should be loaded
        assert config is mock_config
        assert lm._config is mock_config
        mock_from_pretrained.assert_called_once_with("gpt2", trust_remote_code=False)

    def test_config_without_model_path(self):
        """Test accessing config without model_name_or_path raises error."""
        lm = PreTrainedCausalLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.config

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoConfig.from_pretrained")
    def test_config_with_kwargs(self, mock_from_pretrained, mock_config):
        """Test config loading with additional kwargs."""
        mock_from_pretrained.return_value = mock_config

        lm = PreTrainedCausalLM(model_name_or_path="gpt2", trust_remote_code=True, revision="main")

        _ = lm.config

        mock_from_pretrained.assert_called_once_with("gpt2", trust_remote_code=True, revision="main")

    def test_config_setter(self, mock_config):
        """Test setting config manually."""
        lm = PreTrainedCausalLM()

        assert not hasattr(lm, "_config")
        lm.config = mock_config
        assert lm._config is mock_config
        assert lm.config is mock_config

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoConfig.from_pretrained")
    def test_config_cached(self, mock_from_pretrained, mock_config):
        """Test config is cached after first load."""
        mock_from_pretrained.return_value = mock_config

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        # Access config multiple times
        config1 = lm.config
        config2 = lm.config

        # Should only load once
        assert config1 is config2
        mock_from_pretrained.assert_called_once()


class TestPreTrainedCausalLMTokenizerProperty:
    """Test tokenizer property and lazy loading."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_tokenizer_lazy_load(self, mock_from_pretrained, mock_tokenizer):
        """Test tokenizer is lazy loaded on first access."""
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        # Tokenizer should not be loaded yet
        assert not hasattr(lm, "_tokenizer")
        mock_from_pretrained.assert_not_called()

        # Access tokenizer
        tokenizer = lm.tokenizer

        # Now it should be loaded
        assert tokenizer is mock_tokenizer
        assert lm._tokenizer is mock_tokenizer
        mock_from_pretrained.assert_called_once_with("gpt2", trust_remote_code=False)

    def test_tokenizer_without_model_path(self):
        """Test accessing tokenizer without model_name_or_path raises error."""
        lm = PreTrainedCausalLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.tokenizer

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_tokenizer_pad_token_set(self, mock_from_pretrained):
        """Test pad token is set to eos token if None."""
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")
        _ = lm.tokenizer

        assert mock_tokenizer.pad_token == "</s>"

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_tokenizer_pad_token_not_overwritten(self, mock_from_pretrained):
        """Test existing pad token is not overwritten."""
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")
        _ = lm.tokenizer

        assert mock_tokenizer.pad_token == "<pad>"

    def test_tokenizer_setter(self, mock_tokenizer):
        """Test setting tokenizer manually."""
        lm = PreTrainedCausalLM()

        assert not hasattr(lm, "_tokenizer")
        lm.tokenizer = mock_tokenizer
        assert lm._tokenizer is mock_tokenizer
        assert lm.tokenizer is mock_tokenizer


class TestPreTrainedCausalLMModelProperty:
    """Test model property and lazy loading."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_lazy_load(self, mock_from_pretrained, mock_model):
        """Test model is lazy loaded on first access."""
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        # Model should not be loaded yet
        assert not hasattr(lm, "_model")
        mock_from_pretrained.assert_not_called()

        # Access model
        model = lm.model

        # Now it should be loaded
        assert model is mock_model
        assert lm._model is mock_model
        mock_from_pretrained.assert_called_once()

    def test_model_without_model_path(self):
        """Test accessing model without model_name_or_path raises error."""
        lm = PreTrainedCausalLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.model

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_with_dtype(self, mock_from_pretrained, mock_model):
        """Test model loading with specific dtype."""
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedCausalLM(model_name_or_path="gpt2", torch_dtype=torch.float16)

        _ = lm.model

        # Check torch_dtype was passed
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_with_preloaded_config(self, mock_from_pretrained, mock_model, mock_config):
        """Test model loading uses preloaded config."""
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")
        lm._config = mock_config  # Manually set config

        _ = lm.model

        # Check config was passed
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["config"] is mock_config

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_moved_to_device(self, mock_from_pretrained, mock_model):
        """Test model is moved to specified device."""
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedCausalLM(model_name_or_path="gpt2", device="cuda:1")

        _ = lm.model

        mock_model.to.assert_called_once_with("cuda:1")

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_generation_config_set(self, mock_from_pretrained, mock_model, mock_generation_config):
        """Test generation config is set on model if available."""
        mock_from_pretrained.return_value = mock_model
        mock_model.generation_config = None

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")
        lm._generation_config = mock_generation_config

        _ = lm.model

        assert mock_model.generation_config is mock_generation_config

    def test_model_setter(self, mock_model):
        """Test setting model manually."""
        lm = PreTrainedCausalLM(device="cuda")

        assert not hasattr(lm, "_model")
        lm.model = mock_model
        assert lm._model is mock_model
        assert lm.model is mock_model
        mock_model.to.assert_called_once_with("cuda")


class TestPreTrainedCausalLMGenerationConfig:
    """Test generation config property."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.GenerationConfig.from_pretrained")
    def test_generation_config_lazy_load(self, mock_from_pretrained, mock_generation_config):
        """Test generation config is lazy loaded."""
        mock_from_pretrained.return_value = mock_generation_config

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        # Should not be loaded yet
        assert not hasattr(lm, "_generation_config")

        # Access generation config
        config = lm.generation_config

        assert config is mock_generation_config
        mock_from_pretrained.assert_called_once()

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.GenerationConfig.from_pretrained")
    def test_generation_config_not_found(self, mock_from_pretrained):
        """Test generation config returns None if not found."""
        mock_from_pretrained.side_effect = Exception("Not found")

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        config = lm.generation_config
        assert config is None

    def test_generation_config_without_model_path(self):
        """Test generation config returns None without model path."""
        lm = PreTrainedCausalLM()

        config = lm.generation_config
        assert config is None

    def test_generation_config_setter(self, mock_generation_config, mock_model):
        """Test setting generation config manually."""
        lm = PreTrainedCausalLM()

        # Set generation config
        lm.generation_config = mock_generation_config
        assert lm._generation_config is mock_generation_config

        # If model is loaded, it should update model's generation config
        mock_model.generation_config = None  # Initialize the attribute
        lm._model = mock_model
        lm.generation_config = mock_generation_config
        assert mock_model.generation_config is mock_generation_config


class TestPreTrainedCausalLMMethods:
    """Test methods of PreTrainedCausalLM."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_generate_method(self, mock_from_pretrained, mock_model):
        """Test generate method forwards to model."""
        mock_from_pretrained.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        input_ids = torch.tensor([[1, 2, 3]])
        result = lm.generate(input_ids, max_length=10, temperature=0.8)

        mock_model.generate.assert_called_once_with(input_ids, max_length=10, temperature=0.8)
        assert torch.equal(result, torch.tensor([[1, 2, 3, 4, 5]]))

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_call_method(self, mock_from_pretrained):
        """Test __call__ method forwards to model."""
        mock_model = Mock()
        mock_output = Mock()
        mock_model.return_value = mock_output
        # Make to() return the model itself
        mock_model.to = Mock(return_value=mock_model)
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)
        result = lm(input_ids, attention_mask=attention_mask)

        mock_model.assert_called_once_with(input_ids, attention_mask=attention_mask)
        assert result is mock_output

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_encode_method(self, mock_from_pretrained):
        """Test encode method uses tokenizer."""
        # Create tokenizer mock
        mock_tokenizer = Mock()
        mock_encoded = Mock(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )
        mock_encoded.to = Mock(return_value=mock_encoded)
        mock_tokenizer.return_value = mock_encoded
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2", device="cuda")

        result = lm.encode("Hello world", padding=True)

        mock_tokenizer.assert_called_once_with("Hello world", return_tensors="pt", padding=True)
        mock_encoded.to.assert_called_once_with("cuda")
        assert result is mock_encoded

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_encode_method_batch(self, mock_from_pretrained):
        """Test encode method with batch input."""
        mock_tokenizer = Mock()
        mock_encoded = Mock()
        mock_encoded.to = Mock(return_value=mock_encoded)
        mock_tokenizer.return_value = mock_encoded
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        texts = ["Hello", "World"]
        lm.encode(texts)

        mock_tokenizer.assert_called_once_with(texts, return_tensors="pt")

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_decode_method(self, mock_from_pretrained, mock_tokenizer):
        """Test decode method uses tokenizer."""
        mock_from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.decode.return_value = "Hello world"

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        token_ids = [1, 2, 3, 4]
        result = lm.decode(token_ids, skip_special_tokens=True)

        mock_tokenizer.decode.assert_called_once_with(token_ids, skip_special_tokens=True)
        assert result == "Hello world"


class TestPreTrainedCausalLMSaveLoad:
    """Test save and load functionality."""

    def test_save_pretrained_all_components(
        self, tmp_path, mock_model, mock_tokenizer, mock_config, mock_generation_config
    ):
        """Test saving all components."""
        lm = PreTrainedCausalLM()
        lm._model = mock_model
        lm._tokenizer = mock_tokenizer
        lm._config = mock_config
        lm._generation_config = mock_generation_config

        save_dir = tmp_path / "saved_model"
        lm.save_pretrained(save_dir)

        # Check directory was created
        assert save_dir.exists()

        # Check all components were saved
        mock_model.save_pretrained.assert_called_once_with(save_dir)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_dir)
        mock_config.save_pretrained.assert_called_once_with(save_dir)
        mock_generation_config.save_pretrained.assert_called_once_with(save_dir)


class TestPreTrainedCausalLMDeviceManagement:
    """Test device management methods."""

    def test_to_method(self, mock_model):
        """Test to method moves model to device."""
        lm = PreTrainedCausalLM(device="cpu")
        lm._model = mock_model

        result = lm.to("cuda:1")

        assert lm.device == "cuda:1"
        mock_model.to.assert_called_once_with("cuda:1")
        assert result is lm  # Should return self

    def test_to_method_no_model(self):
        """Test to method without loaded model."""
        lm = PreTrainedCausalLM(device="cpu")

        result = lm.to("cuda")

        assert lm.device == "cuda"
        assert result is lm

    def test_to_method_torch_device(self, mock_model):
        """Test to method with torch.device."""
        lm = PreTrainedCausalLM()
        lm._model = mock_model

        device = torch.device("cuda", 0)
        lm.to(device)

        assert lm.device == device
        mock_model.to.assert_called_once_with(device)


class TestPreTrainedCausalLMPrecision:
    """Test precision conversion methods."""

    def test_half_method(self, mock_model):
        """Test half method converts to float16."""
        lm = PreTrainedCausalLM()
        lm._model = mock_model

        result = lm.half()

        mock_model.half.assert_called_once()
        assert result is lm

    def test_half_method_no_model(self):
        """Test half method without loaded model."""
        lm = PreTrainedCausalLM()

        result = lm.half()

        assert result is lm

    def test_float_method(self, mock_model):
        """Test float method converts to float32."""
        lm = PreTrainedCausalLM()
        lm._model = mock_model

        result = lm.float()

        mock_model.float.assert_called_once()
        assert result is lm

    def test_float_method_no_model(self):
        """Test float method without loaded model."""
        lm = PreTrainedCausalLM()

        result = lm.float()

        assert result is lm


class TestPreTrainedCausalLMProperties:
    """Test various properties."""

    def test_dtype_property(self):
        """Test dtype property."""
        lm = PreTrainedCausalLM()

        # No model loaded
        assert lm.dtype is None

        # With model
        mock_model = Mock()
        param = Mock()
        param.dtype = torch.float16
        mock_model.parameters.return_value = iter([param])
        lm._model = mock_model

        assert lm.dtype == torch.float16

    def test_num_parameters_property(self):
        """Test num_parameters property."""
        lm = PreTrainedCausalLM()

        # No model loaded
        assert lm.num_parameters is None

        # With model
        mock_model = Mock()
        params = [
            torch.randn(10, 20),  # 200 params
            torch.randn(5, 5),  # 25 params
            torch.randn(100),  # 100 params
        ]
        mock_model.parameters.return_value = params
        lm._model = mock_model

        assert lm.num_parameters == 325


class TestPreTrainedCausalLMStateDict:
    """Test state dict accessor integration."""

    def test_state_property(self):
        """Test state property returns StateDict accessor."""
        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        state = lm.state

        assert state is not None
        assert hasattr(state, "__getitem__")
        assert hasattr(state, "keys")
        assert hasattr(state, "glob")
        assert hasattr(state, "regex")

    def test_state_property_cached(self):
        """Test state property is cached."""
        lm = PreTrainedCausalLM(model_name_or_path="gpt2")

        state1 = lm.state
        state2 = lm.state

        assert state1 is state2


class TestPreTrainedCausalLMIntegration:
    """Integration tests with mocked transformers."""

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoConfig.from_pretrained")
    def test_full_pipeline(
        self,
        mock_config_load,
        mock_tokenizer_load,
        mock_model_load,
        mock_config,
        mock_tokenizer,
        mock_model,
    ):
        """Test full pipeline from loading to generation."""
        mock_config_load.return_value = mock_config
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_model_load.return_value = mock_model

        # Initialize model
        lm = PreTrainedCausalLM.from_pretrained("gpt2", device="cuda", torch_dtype=torch.float16)

        # Encode text
        encoded = lm.encode("Hello world")

        # Generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        output_ids = lm.generate(encoded.input_ids, max_length=10)

        # Decode
        text = lm.decode(output_ids[0])

        assert text == "decoded text"
        assert mock_model_load.called
        assert mock_tokenizer_load.called

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_type_safety(self, mock_from_pretrained):
        """Test generic type annotations work correctly."""
        from transformers import GPT2LMHeadModel

        mock_model = Mock(spec=GPT2LMHeadModel)
        mock_from_pretrained.return_value = mock_model

        # This should work with type checkers
        lm: PreTrainedCausalLM[GPT2LMHeadModel] = PreTrainedCausalLM(model_name_or_path="gpt2")

        model = lm.model
        assert isinstance(model, Mock)  # In tests
        # In real usage: assert isinstance(model, GPT2LMHeadModel)


class TestPreTrainedCausalLMEdgeCases:
    """Test edge cases and error handling."""

    def test_pathlib_path_support(self, tmp_path):
        """Test Path objects are supported."""
        model_path = tmp_path / "model"
        lm = PreTrainedCausalLM(model_name_or_path=model_path)

        assert lm.model_name_or_path == model_path
        assert isinstance(lm.model_name_or_path, Path)

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained")
    def test_model_loading_error_propagation(self, mock_from_pretrained):
        """Test model loading errors are propagated."""
        mock_from_pretrained.side_effect = RuntimeError("Model not found")

        lm = PreTrainedCausalLM(model_name_or_path="invalid-model")

        with pytest.raises(RuntimeError, match="Model not found"):
            _ = lm.model

    def test_empty_kwargs(self):
        """Test empty kwargs don't cause issues."""
        lm = PreTrainedCausalLM(**{})
        assert lm.init_kwargs == {}

    @patch("megatron.bridge.models.hf_pretrained.causal_lm.AutoTokenizer.from_pretrained")
    def test_tokenizer_with_none_pad_and_eos(self, mock_from_pretrained):
        """Test tokenizer when both pad and eos tokens are None."""
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = None
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedCausalLM(model_name_or_path="gpt2")
        _ = lm.tokenizer

        # pad_token should remain None
        assert mock_tokenizer.pad_token is None

    def test_repr_method(self, mock_model, mock_tokenizer, mock_config):
        """Test string representation."""
        lm = PreTrainedCausalLM()

        # Empty state
        repr1 = repr(lm)
        assert "PreTrainedCausalLM" in repr1
        assert "[not loaded]" in repr1

        # With components - create a new parameters iterator for each access
        def params_generator():
            return iter([torch.randn(10, 10) for _ in range(5)])

        mock_model.parameters.side_effect = params_generator
        lm._model = mock_model
        lm._tokenizer = mock_tokenizer
        lm._config = mock_config

        repr2 = repr(lm)
        assert "GPT2LMHeadModel" in repr2
        assert "GPT2TokenizerFast" in repr2
        assert "layers=12" in repr2


@pytest.fixture
def mock_config():
    """Mock AutoConfig for testing."""
    config = Mock()
    config.model_type = "gpt2"
    config.num_hidden_layers = 12
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.save_pretrained = Mock()
    return config


@pytest.fixture
def mock_tokenizer():
    """Mock PreTrainedTokenizer for testing."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.__class__.__name__ = "GPT2TokenizerFast"
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.decode.return_value = "decoded text"
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.__class__.__name__ = "GPT2LMHeadModel"
    model.to = Mock(return_value=model)
    model.half = Mock(return_value=model)
    model.float = Mock(return_value=model)
    model.save_pretrained = Mock()
    model.generation_config = None
    return model


@pytest.fixture
def mock_generation_config():
    """Mock GenerationConfig for testing."""
    config = Mock()
    config.save_pretrained = Mock()
    return config
