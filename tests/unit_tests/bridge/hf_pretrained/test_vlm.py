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
Comprehensive tests for PreTrainedVLM class.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import AutoConfig, PreTrainedTokenizer, ProcessorMixin

from megatron.hub.bridge.hf_pretrained.vlm import PreTrainedVLM


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    config = Mock(spec=AutoConfig)
    config.architectures = ["LlavaForConditionalGeneration"]
    config.model_type = "llava"
    config.hidden_size = 4096
    config.vision_config = Mock()
    config.vision_config.hidden_size = 1024
    config.text_config = Mock()
    config.text_config.hidden_size = 4096
    return config


@pytest.fixture
def mock_processor():
    """Create a mock processor with tokenizer and image processor."""
    processor = Mock(spec=ProcessorMixin)
    processor.__class__.__name__ = "LlavaProcessor"

    # Add tokenizer
    processor.tokenizer = Mock(spec=PreTrainedTokenizer)
    processor.tokenizer.__class__.__name__ = "LlamaTokenizerFast"
    processor.tokenizer.pad_token = "<pad>"
    processor.tokenizer.eos_token = "</s>"

    # Add image processor
    processor.image_processor = Mock()
    processor.image_processor.__class__.__name__ = "CLIPImageProcessor"

    return processor


@pytest.fixture
def mock_model():
    """Create a mock VLM model."""
    model = Mock()
    model.__class__.__name__ = "LlavaForConditionalGeneration"
    model.config = Mock()
    model.config.architectures = ["LlavaForConditionalGeneration"]
    model.generation_config = Mock()
    model.to = Mock(return_value=model)
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
    model.half = Mock(return_value=model)
    model.float = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 10)])
    return model


class TestPreTrainedVLMInitialization:
    """Test initialization and configuration of PreTrainedVLM."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_minimal(self, mock_cuda):
        """Test minimal initialization."""
        vlm = PreTrainedVLM()

        assert vlm._model_name_or_path is None
        assert vlm.device == "cpu"  # Default when CUDA not available
        assert vlm.torch_dtype is None
        assert vlm.trust_remote_code is False
        assert vlm.kwargs == {}

        # Lazy loaded components should not exist yet
        assert not hasattr(vlm, "_config")
        assert not hasattr(vlm, "_processor")
        assert not hasattr(vlm, "_tokenizer")
        assert not hasattr(vlm, "_image_processor")
        assert not hasattr(vlm, "_model")
        assert not hasattr(vlm, "_generation_config")

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        model_path = "llava-hf/llava-1.5-7b-hf"
        vlm = PreTrainedVLM(model_name_or_path=model_path)

        assert vlm._model_name_or_path == model_path
        assert vlm.model_name_or_path == model_path

    def test_init_with_device(self):
        """Test initialization with specific device."""
        vlm = PreTrainedVLM(device="cuda:0")
        assert vlm.device == "cuda:0"

        vlm2 = PreTrainedVLM(device=torch.device("cpu"))
        assert vlm2.device == torch.device("cpu")

    def test_init_with_dtype(self):
        """Test initialization with specific dtype."""
        vlm = PreTrainedVLM(torch_dtype=torch.float16)
        assert vlm.torch_dtype == torch.float16

    def test_init_with_trust_remote_code(self):
        """Test initialization with trust_remote_code."""
        vlm = PreTrainedVLM(trust_remote_code=True)
        assert vlm.trust_remote_code is True

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_with_kwargs(self, mock_cuda):
        """Test initialization with additional kwargs."""
        kwargs = {"use_fast": True, "padding_side": "left"}
        vlm = PreTrainedVLM(**kwargs)
        assert vlm.kwargs == kwargs

    @patch("torch.cuda.is_available", return_value=False)
    def test_from_pretrained_classmethod(self, mock_cuda):
        """Test from_pretrained class method."""
        model_path = "llava-hf/llava-1.5-7b-hf"
        vlm = PreTrainedVLM.from_pretrained(
            model_path,
            device="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_fast=True,
        )

        assert vlm._model_name_or_path == model_path
        assert vlm.device == "cuda"
        assert vlm.torch_dtype == torch.float16
        assert vlm.trust_remote_code is True
        assert vlm.kwargs == {"use_fast": True}

    @patch("torch.cuda.is_available")
    def test_cuda_device_selection(self, mock_cuda):
        """Test automatic CUDA device selection."""
        mock_cuda.return_value = True
        vlm = PreTrainedVLM()
        assert vlm.device == "cuda"

        mock_cuda.return_value = False
        vlm2 = PreTrainedVLM()
        assert vlm2.device == "cpu"


class TestPreTrainedVLMConfigProperty:
    """Test config property and lazy loading."""

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoConfig.from_pretrained")
    def test_config_lazy_load(self, mock_from_pretrained, mock_config):
        """Test config is lazy loaded on first access."""
        mock_from_pretrained.return_value = mock_config

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Config should not be loaded yet
        assert not hasattr(vlm, "_config")
        mock_from_pretrained.assert_not_called()

        # Access config
        config = vlm.config

        # Now it should be loaded
        assert config is mock_config
        assert vlm._config is mock_config
        mock_from_pretrained.assert_called_once_with("llava-hf/llava-1.5-7b-hf", trust_remote_code=False)

    def test_config_without_model_path(self):
        """Test accessing config without model_name_or_path raises error."""
        vlm = PreTrainedVLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = vlm.config

    def test_config_setter(self, mock_config):
        """Test setting config manually."""
        vlm = PreTrainedVLM()

        assert not hasattr(vlm, "_config")
        vlm.config = mock_config
        assert vlm._config is mock_config
        assert vlm.config is mock_config


class TestPreTrainedVLMProcessorProperty:
    """Test processor property and lazy loading."""

    def create_mock_processor(self):
        """Create a mock processor with image_processor and tokenizer."""
        processor = Mock(spec=ProcessorMixin)
        processor.__class__.__name__ = "LlavaProcessor"

        # Add image processor
        processor.image_processor = Mock()
        processor.image_processor.__class__.__name__ = "CLIPImageProcessor"

        # Add tokenizer
        processor.tokenizer = Mock(spec=PreTrainedTokenizer)
        processor.tokenizer.__class__.__name__ = "LlamaTokenizerFast"

        return processor

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoProcessor.from_pretrained")
    def test_processor_lazy_load(self, mock_from_pretrained):
        """Test processor is lazy loaded on first access."""
        mock_processor = self.create_mock_processor()
        mock_from_pretrained.return_value = mock_processor

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Processor should not be loaded yet
        assert not hasattr(vlm, "_processor")
        mock_from_pretrained.assert_not_called()

        # Access processor
        processor = vlm.processor

        # Now it should be loaded
        assert processor is mock_processor
        assert vlm._processor is mock_processor
        mock_from_pretrained.assert_called_once_with("llava-hf/llava-1.5-7b-hf", trust_remote_code=False)

    def test_processor_without_model_path(self):
        """Test accessing processor without model_name_or_path raises error."""
        vlm = PreTrainedVLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = vlm.processor

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoProcessor.from_pretrained")
    def test_processor_load_failure(self, mock_from_pretrained):
        """Test processor loading failure raises informative error."""
        mock_from_pretrained.side_effect = Exception("Not found")

        vlm = PreTrainedVLM(model_name_or_path="invalid-model")

        with pytest.raises(ValueError, match="Could not load processor"):
            _ = vlm.processor

    def test_processor_setter(self):
        """Test setting processor manually."""
        vlm = PreTrainedVLM()
        mock_processor = self.create_mock_processor()

        assert not hasattr(vlm, "_processor")
        vlm.processor = mock_processor
        assert vlm._processor is mock_processor
        assert vlm.processor is mock_processor


class TestPreTrainedVLMTokenizerProperty:
    """Test tokenizer property and interaction with processor."""

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoTokenizer.from_pretrained")
    def test_tokenizer_from_processor(self, mock_from_pretrained):
        """Test tokenizer accessed from processor."""
        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Create processor with tokenizer
        mock_processor = Mock(spec=ProcessorMixin)
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_processor.tokenizer = mock_tokenizer
        vlm._processor = mock_processor

        # Access tokenizer
        tokenizer = vlm.tokenizer

        # Should return processor's tokenizer, not load separately
        assert tokenizer is mock_tokenizer
        mock_from_pretrained.assert_not_called()

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoTokenizer.from_pretrained")
    def test_tokenizer_load_separately(self, mock_from_pretrained):
        """Test tokenizer loaded separately when not in processor."""
        mock_tokenizer = Mock(spec=PreTrainedTokenizer)
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_from_pretrained.return_value = mock_tokenizer

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Access tokenizer
        tokenizer = vlm.tokenizer

        # Should load separately
        assert tokenizer is mock_tokenizer
        assert tokenizer.pad_token == "</s>"  # Should be set from eos_token
        mock_from_pretrained.assert_called_once()

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoTokenizer.from_pretrained")
    def test_tokenizer_load_failure_silent(self, mock_from_pretrained):
        """Test tokenizer loading failure returns None."""
        mock_from_pretrained.side_effect = Exception("Not found")

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Should return None on failure
        tokenizer = vlm.tokenizer
        assert tokenizer is None


class TestPreTrainedVLMImageProcessorProperty:
    """Test image processor property and interaction with processor."""

    def test_image_processor_from_processor(self):
        """Test image processor accessed from processor."""
        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Create processor with image processor
        mock_processor = Mock(spec=ProcessorMixin)
        mock_image_processor = Mock()
        mock_processor.image_processor = mock_image_processor
        vlm._processor = mock_processor

        # Access image processor
        image_processor = vlm.image_processor

        # Should return processor's image processor
        assert image_processor is mock_image_processor

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoImageProcessor.from_pretrained")
    def test_image_processor_load_separately(self, mock_from_pretrained):
        """Test image processor loaded separately when not in processor."""
        mock_image_processor = Mock()
        mock_from_pretrained.return_value = mock_image_processor

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Access image processor
        image_processor = vlm.image_processor

        # Should load separately
        assert image_processor is mock_image_processor
        mock_from_pretrained.assert_called_once()


class TestPreTrainedVLMModelProperty:
    """Test model property and lazy loading."""

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoModel.from_pretrained")
    def test_model_lazy_load(self, mock_from_pretrained, mock_model):
        """Test model is lazy loaded on first access."""
        mock_from_pretrained.return_value = mock_model

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Model should not be loaded yet
        assert not hasattr(vlm, "_model")
        mock_from_pretrained.assert_not_called()

        # Access model
        model = vlm.model

        # Now it should be loaded
        assert model is mock_model
        assert vlm._model is mock_model
        mock_from_pretrained.assert_called_once()
        mock_model.to.assert_called_once()

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoModel.from_pretrained")
    def test_model_with_dtype(self, mock_from_pretrained, mock_model):
        """Test model loading with specific dtype."""
        mock_from_pretrained.return_value = mock_model

        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16)

        # Access model
        _ = vlm.model

        # Check dtype was passed
        call_kwargs = mock_from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16

    def test_model_moved_to_device(self, mock_model):
        """Test model is moved to device."""
        vlm = PreTrainedVLM(device="cuda:0")
        vlm._model = mock_model

        _ = vlm.model
        mock_model.to.assert_called_with("cuda:0")

    def test_model_setter(self, mock_model):
        """Test setting model manually."""
        vlm = PreTrainedVLM()

        assert not hasattr(vlm, "_model")
        vlm.model = mock_model
        assert vlm._model is mock_model
        assert vlm.model is mock_model


class TestPreTrainedVLMGenerationMethods:
    """Test generation-related methods."""

    def test_generate_method(self, mock_model):
        """Test generate method."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model

        # Test generation
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        outputs = vlm.generate(**inputs, max_length=50)

        mock_model.generate.assert_called_once_with(**inputs, max_length=50)
        assert outputs.shape == (1, 3)  # Based on mock return

    def test_call_method(self, mock_model):
        """Test __call__ method."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model
        mock_model.return_value = {"logits": torch.randn(1, 3, 100)}

        # Test forward pass
        inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        outputs = vlm(**inputs)

        mock_model.assert_called_once_with(**inputs)
        assert "logits" in outputs


class TestPreTrainedVLMSaveLoad:
    """Test save and load functionality."""

    def test_save_pretrained_all_components(self, tmp_path, mock_model, mock_processor, mock_config):
        """Test saving all components."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model
        vlm._processor = mock_processor
        vlm._config = mock_config
        vlm._tokenizer = mock_processor.tokenizer

        # Mock save methods
        mock_model.save_pretrained = Mock()
        mock_processor.save_pretrained = Mock()
        mock_processor.tokenizer.save_pretrained = Mock()
        mock_config.save_pretrained = Mock()

        # Save
        vlm.save_pretrained(tmp_path)

        # Check all components saved
        mock_model.save_pretrained.assert_called_once()
        mock_config.save_pretrained.assert_called_once()
        mock_processor.save_pretrained.assert_called_once()

    def test_save_pretrained_skip_processor_components(self, tmp_path, mock_model, mock_processor, mock_config):
        """Test saving skips processor sub-components when processor exists."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model
        vlm._processor = mock_processor
        vlm._config = mock_config

        # Processor contains tokenizer, so it shouldn't be saved separately
        vlm._tokenizer = mock_processor.tokenizer
        vlm._image_processor = mock_processor.image_processor

        # Mock save methods
        mock_model.save_pretrained = Mock()
        mock_processor.save_pretrained = Mock()
        mock_config.save_pretrained = Mock()

        # Save - should not try to save tokenizer/image_processor separately
        # This should work without errors
        vlm.save_artifacts(tmp_path)


class TestPreTrainedVLMDeviceManagement:
    """Test device management functionality."""

    def test_to_method(self, mock_model):
        """Test to() method."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model

        # Move to device
        result = vlm.to("cuda:1")

        assert vlm.device == "cuda:1"
        mock_model.to.assert_called_once_with("cuda:1")
        assert result is vlm  # Should return self for chaining

    def test_to_method_no_model(self):
        """Test to() method when model not loaded."""
        vlm = PreTrainedVLM()

        # Should still update device
        result = vlm.to("cuda:0")
        assert vlm.device == "cuda:0"
        assert result is vlm


class TestPreTrainedVLMPrecision:
    """Test precision conversion methods."""

    def test_half_method(self, mock_model):
        """Test half() method."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model

        result = vlm.half()

        mock_model.half.assert_called_once()
        assert vlm.torch_dtype == torch.float16
        assert result is vlm

    def test_float_method(self, mock_model):
        """Test float() method."""
        vlm = PreTrainedVLM()
        vlm._model = mock_model

        result = vlm.float()

        mock_model.float.assert_called_once()
        assert vlm.torch_dtype == torch.float32
        assert result is vlm


class TestPreTrainedVLMProperties:
    """Test various property methods."""

    def test_dtype_property(self, mock_model):
        """Test dtype property."""
        vlm = PreTrainedVLM()

        # When model not loaded, use torch_dtype
        vlm.torch_dtype = torch.float16
        assert vlm.dtype == torch.float16

        # When model loaded, use model's dtype
        param = torch.randn(10, 10, dtype=torch.bfloat16)
        mock_model.parameters.return_value = iter([param])
        vlm._model = mock_model
        assert vlm.dtype == torch.bfloat16

    def test_num_parameters_property(self, mock_model):
        """Test num_parameters method."""
        vlm = PreTrainedVLM()

        # No model loaded
        assert vlm.num_parameters() == 0

        # With model
        param1 = torch.randn(10, 10)  # 100 params
        param1.requires_grad = True  # Explicitly set to True
        param2 = torch.randn(20, 20)  # 400 params
        param2.requires_grad = False

        # Make parameters() return a new iterator each time it's called
        mock_model.parameters = Mock(side_effect=lambda: iter([param1, param2]))
        vlm._model = mock_model

        assert vlm.num_parameters() == 500
        assert vlm.num_parameters(only_trainable=True) == 100


class TestPreTrainedVLMRepr:
    """Test string representation."""

    def test_repr_empty(self):
        """Test repr with empty instance."""
        vlm = PreTrainedVLM()
        repr_str = repr(vlm)

        assert "PreTrainedVLM" in repr_str
        assert "device='cpu'" in repr_str or "device='cuda'" in repr_str

    def test_repr_with_processor(self, mock_processor):
        """Test repr with loaded components."""
        vlm = PreTrainedVLM(
            model_name_or_path="llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, trust_remote_code=True
        )
        vlm._processor = mock_processor

        repr_str = repr(vlm)

        assert "PreTrainedVLM" in repr_str
        assert "llava-hf/llava-1.5-7b-hf" in repr_str
        assert "torch.float16" in repr_str
        assert "trust_remote_code=True" in repr_str
        assert "processor" in repr_str

    def test_repr_with_vlm_model(self, mock_model, mock_processor, mock_config):
        """Test repr with all components loaded."""
        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")
        vlm._model = mock_model
        vlm._processor = mock_processor
        vlm._config = mock_config
        vlm._tokenizer = mock_processor.tokenizer

        repr_str = repr(vlm)

        assert "model" in repr_str
        assert "processor" in repr_str
        assert "tokenizer" in repr_str
        assert "config" in repr_str


class TestPreTrainedVLMStateDict:
    """Test state dict access functionality."""

    def test_state_property(self, mock_model):
        """Test state property returns StateDict."""
        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")

        # Mock model state dict
        state_dict = {
            "vision_tower.weight": torch.randn(10, 10),
            "language_model.weight": torch.randn(20, 20),
        }
        mock_model.state_dict.return_value = state_dict
        vlm._model = mock_model

        # Access state
        state = vlm.state

        # Should be a StateDict
        assert hasattr(state, "keys")
        assert hasattr(state, "glob")
        assert hasattr(state, "regex")

    def test_state_property_cached(self, mock_model):
        """Test state property is cached."""
        vlm = PreTrainedVLM(model_name_or_path="llava-hf/llava-1.5-7b-hf")
        vlm._model = mock_model
        mock_model.state_dict.return_value = {}

        # Access state twice
        state1 = vlm.state
        state2 = vlm.state

        # Should be same instance
        assert state1 is state2


class TestPreTrainedVLMIntegration:
    """Integration tests for common VLM workflows."""

    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoModel.from_pretrained")
    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoProcessor.from_pretrained")
    @patch("megatron.hub.bridge.hf_pretrained.vlm.AutoConfig.from_pretrained")
    def test_full_vlm_pipeline(
        self,
        mock_config_load,
        mock_processor_load,
        mock_model_load,
        mock_config,
        mock_processor,
        mock_model,
    ):
        """Test complete VLM pipeline from loading to generation."""
        # Setup mocks
        mock_config_load.return_value = mock_config
        mock_processor_load.return_value = mock_processor
        mock_model_load.return_value = mock_model

        # Create VLM
        vlm = PreTrainedVLM.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device="cuda")

        # Process inputs
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        inputs = vlm.process_images_and_text(images="dummy_image", text="What is this?")

        # Generate
        _ = vlm.generate(**inputs, max_length=50)

        # Verify calls
        mock_processor.assert_called_once()
        mock_model.generate.assert_called_once()


class TestPreTrainedVLMEdgeCases:
    """Test edge cases and error handling."""

    def test_processor_tokenizer_interaction(self, mock_processor):
        """Test tokenizer property when processor has tokenizer."""
        vlm = PreTrainedVLM()
        vlm._processor = mock_processor

        # First access should return processor's tokenizer
        tokenizer1 = vlm.tokenizer
        assert tokenizer1 is mock_processor.tokenizer

        # Set tokenizer separately
        new_tokenizer = Mock(spec=PreTrainedTokenizer)
        vlm.tokenizer = new_tokenizer

        # Should now return the manually set tokenizer
        tokenizer2 = vlm.tokenizer
        assert tokenizer2 is new_tokenizer

    def test_empty_kwargs(self):
        """Test that kwargs defaults to empty dict."""
        vlm = PreTrainedVLM()
        assert vlm.kwargs == {}

    @patch("megatron.hub.bridge.hf_pretrained.vlm.GenerationConfig.from_pretrained")
    def test_generation_config_silent_failure(self, mock_from_pretrained):
        """Test generation config returns None on failure."""
        mock_from_pretrained.side_effect = Exception("Not found")

        vlm = PreTrainedVLM(model_name_or_path="model-without-gen-config")
        gen_config = vlm.generation_config

        assert gen_config is None
