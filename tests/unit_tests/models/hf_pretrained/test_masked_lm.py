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

"""
Tests for PreTrainedMaskedLM class.
"""

from unittest.mock import Mock, patch

import pytest
import torch
from transformers import BertConfig, GPT2Config, PreTrainedTokenizer

from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM


class TestPreTrainedMaskedLMInitialization:
    """Test initialization and configuration of PreTrainedMaskedLM."""

    @patch("torch.cuda.is_available")
    def test_init_minimal(self, mock_cuda):
        """Test minimal initialization has no generation-specific state."""
        mock_cuda.return_value = False
        lm = PreTrainedMaskedLM()

        assert lm._model_name_or_path is None
        assert lm.device == "cpu"
        assert lm.trust_remote_code is False
        assert not hasattr(lm, "_config")
        assert not hasattr(lm, "_tokenizer")
        assert not hasattr(lm, "_model")
        # Unlike PreTrainedCausalLM, no generation_config artifact exists at all.
        assert not hasattr(lm, "generation_config")
        assert PreTrainedMaskedLM.OPTIONAL_ARTIFACTS == []

    def test_from_pretrained_classmethod(self):
        """Test from_pretrained class method wires constructor args through."""
        lm = PreTrainedMaskedLM.from_pretrained(
            "bert-base-uncased",
            device="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        assert lm._model_name_or_path == "bert-base-uncased"
        assert lm.device == "cuda"
        assert lm.torch_dtype == torch.float16
        assert lm.trust_remote_code is True


class TestPreTrainedMaskedLMConfigProperty:
    """Test config property and lazy loading."""

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.safe_load_config_with_retry")
    def test_config_lazy_load(self, mock_load_config, mock_config):
        """Test config is lazy loaded on first access."""
        mock_load_config.return_value = mock_config

        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        assert not hasattr(lm, "_config")

        config = lm.config

        assert config is mock_config
        mock_load_config.assert_called_once_with("bert-base-uncased", trust_remote_code=False)

    def test_config_without_model_path(self):
        """Test accessing config without model_name_or_path raises error."""
        lm = PreTrainedMaskedLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.config


class TestPreTrainedMaskedLMModelProperty:
    """Test model property, lazy loading, and the AutoModelForMaskedLM/AutoModel fallback."""

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.safe_load_config_with_retry")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModelForMaskedLM.from_pretrained")
    def test_model_lazy_load_via_masked_lm_head(self, mock_from_pretrained, mock_load_config, mock_model):
        """Test model is loaded via AutoModelForMaskedLM when a masked-LM head is registered."""
        mock_load_config.return_value = BertConfig()
        mock_from_pretrained.return_value = mock_model

        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        assert not hasattr(lm, "_model")

        model = lm.model

        assert model is mock_model
        mock_from_pretrained.assert_called_once()

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.safe_load_config_with_retry")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModel.from_pretrained")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModelForMaskedLM.from_pretrained")
    def test_model_falls_back_to_auto_model(
        self, mock_masked_lm_from_pretrained, mock_auto_model_from_pretrained, mock_load_config, mock_model
    ):
        """Test model loading falls back to AutoModel when the config has no registered masked-LM head."""
        # GPT2Config has no entry in MODEL_FOR_MASKED_LM_MAPPING.
        mock_load_config.return_value = GPT2Config()
        mock_auto_model_from_pretrained.return_value = mock_model

        lm = PreTrainedMaskedLM(model_name_or_path="some-encoder-only-model")
        model = lm.model

        assert model is mock_model
        mock_auto_model_from_pretrained.assert_called_once()
        # AutoModelForMaskedLM must not even be attempted for an unregistered config class.
        mock_masked_lm_from_pretrained.assert_not_called()

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.safe_load_config_with_retry")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModel.from_pretrained")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModelForMaskedLM.from_pretrained")
    def test_model_reraises_unrelated_value_error(
        self, mock_masked_lm_from_pretrained, mock_auto_model_from_pretrained, mock_load_config
    ):
        """Regression test: a ValueError unrelated to a missing masked-LM head must propagate.

        Previously, any `ValueError` raised by `AutoModelForMaskedLM.from_pretrained` (even one
        caused by invalid loader/config options or a bug in a custom MLM implementation) was
        swallowed and silently retried via `AutoModel`, which could return a model without the
        MLM head. Now that the fallback decision is based on the config's registry membership
        (BertConfig *is* registered), an unrelated `ValueError` must propagate untouched instead
        of triggering the `AutoModel` fallback.
        """
        mock_load_config.return_value = BertConfig()
        mock_masked_lm_from_pretrained.side_effect = ValueError("invalid loader option")

        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")

        with pytest.raises(ValueError, match="invalid loader option"):
            _ = lm.model

        mock_auto_model_from_pretrained.assert_not_called()

    def test_model_without_model_path(self):
        """Test accessing model without model_name_or_path raises error."""
        lm = PreTrainedMaskedLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.model

    def test_model_setter(self, mock_model):
        """Test setting model manually moves it to the configured device."""
        lm = PreTrainedMaskedLM(device="cuda")

        lm.model = mock_model

        assert lm._model is mock_model
        mock_model.to.assert_called_once_with("cuda")


class TestPreTrainedMaskedLMMethods:
    """Test forward call, encode, and decode."""

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.safe_load_config_with_retry")
    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoModelForMaskedLM.from_pretrained")
    def test_call_method(self, mock_from_pretrained, mock_load_config, mock_model):
        """Test __call__ forwards to the underlying model."""
        mock_load_config.return_value = BertConfig()
        mock_from_pretrained.return_value = mock_model
        mock_model.return_value = "model_output"

        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        result = lm(input_ids="dummy_ids")

        assert result == "model_output"
        mock_model.assert_called_once_with(input_ids="dummy_ids")

    def test_encode_method(self, mock_tokenizer):
        """Test encode tokenizes text and moves tensors to the model's device."""
        lm = PreTrainedMaskedLM(device="cpu")
        lm._tokenizer = mock_tokenizer

        lm.encode("The capital of France is [MASK].")

        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["return_tensors"] == "pt"

    def test_decode_method(self, mock_tokenizer):
        """Test decode forwards to the tokenizer's decode method."""
        lm = PreTrainedMaskedLM()
        lm._tokenizer = mock_tokenizer

        lm.decode([101, 2054, 102])

        mock_tokenizer.decode.assert_called_once_with([101, 2054, 102])


class TestPreTrainedMaskedLMDeviceManagement:
    """Test to/half/float device and precision helpers."""

    def test_to_method(self, mock_model):
        """Test moving the model to a new device."""
        lm = PreTrainedMaskedLM(device="cpu")
        lm._model = mock_model

        lm.to("cuda:0")

        assert lm.device == "cuda:0"
        mock_model.to.assert_called_once_with("cuda:0")

    def test_half_method_no_model(self):
        """Test half() is a no-op when no model has been loaded."""
        lm = PreTrainedMaskedLM()
        result = lm.half()
        assert result is lm

    def test_float_method(self, mock_model):
        """Test float() converts a loaded model to float32."""
        mock_model.float.return_value = mock_model
        lm = PreTrainedMaskedLM()
        lm._model = mock_model

        lm.float()

        mock_model.float.assert_called_once()


class TestPreTrainedMaskedLMProperties:
    """Test dtype/num_parameters/__repr__."""

    def test_dtype_and_num_parameters_without_model(self):
        """Test dtype and num_parameters are None before the model is loaded."""
        lm = PreTrainedMaskedLM()
        assert lm.dtype is None
        assert lm.num_parameters is None

    def test_repr_does_not_crash_without_loaded_components(self):
        """Test __repr__ is safe to call before any component is loaded."""
        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")

        with patch.object(PreTrainedMaskedLM, "config", new=None):
            repr_str = repr(lm)

        assert "PreTrainedMaskedLM" in repr_str


class TestPreTrainedMaskedLMHasRegisteredMaskedLMHead:
    """Test the registry / auto_map detection used to choose the loader class."""

    def test_auto_map_declares_masked_lm_head(self):
        """A trust_remote_code config with 'AutoModelForMaskedLM' in auto_map is detected."""
        config = Mock()
        config.auto_map = {"AutoModelForMaskedLM": "modeling_custom.CustomForMaskedLM"}

        assert PreTrainedMaskedLM._has_registered_masked_lm_head(config) is True

    def test_auto_map_without_masked_lm_entry_falls_back_to_registry(self):
        """An auto_map that doesn't declare AutoModelForMaskedLM falls back to the static registry."""
        config = GPT2Config()
        config.auto_map = {"AutoModelForCausalLM": "modeling_custom.CustomForCausalLM"}

        assert PreTrainedMaskedLM._has_registered_masked_lm_head(config) is False

    def test_registered_config_class_detected(self):
        """A config class registered in MODEL_FOR_MASKED_LM_MAPPING is detected without auto_map."""
        config = BertConfig()

        assert PreTrainedMaskedLM._has_registered_masked_lm_head(config) is True


class TestPreTrainedMaskedLMTokenizer:
    """Test tokenizer lazy loading and manual assignment."""

    @patch("megatron.bridge.models.hf_pretrained.masked_lm.AutoTokenizer.from_pretrained")
    def test_tokenizer_lazy_load(self, mock_from_pretrained, mock_tokenizer):
        """Test tokenizer is lazily loaded via AutoTokenizer on first access."""
        mock_from_pretrained.return_value = mock_tokenizer

        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased", trust_remote_code=True)
        assert not hasattr(lm, "_tokenizer")

        tokenizer = lm.tokenizer

        assert tokenizer is mock_tokenizer
        mock_from_pretrained.assert_called_once_with("bert-base-uncased", trust_remote_code=True)
        # Second access must not reload.
        _ = lm.tokenizer
        mock_from_pretrained.assert_called_once()

    def test_tokenizer_without_model_path(self):
        """Test accessing tokenizer without model_name_or_path raises error."""
        lm = PreTrainedMaskedLM()

        with pytest.raises(ValueError, match="model_name_or_path must be provided"):
            _ = lm.tokenizer

    def test_tokenizer_setter(self, mock_tokenizer):
        """Test the tokenizer can be set manually, bypassing lazy loading."""
        lm = PreTrainedMaskedLM()

        lm.tokenizer = mock_tokenizer

        assert lm.tokenizer is mock_tokenizer


class TestPreTrainedMaskedLMModelNameOrPathAndHasModel:
    """Test the model_name_or_path and has_model properties."""

    def test_model_name_or_path_property(self):
        """Test model_name_or_path exposes the constructor argument."""
        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        assert lm.model_name_or_path == "bert-base-uncased"

        lm_none = PreTrainedMaskedLM()
        assert lm_none.model_name_or_path is None

    def test_has_model_true_after_setting_model(self, mock_model):
        """Test has_model reports True once a model has been assigned."""
        lm = PreTrainedMaskedLM()
        assert lm.has_model is False

        lm.model = mock_model

        assert lm.has_model is True


class TestPreTrainedMaskedLMSavePretrained:
    """Test save_pretrained behavior."""

    def test_save_pretrained_saves_model_and_artifacts(self, mock_model, tmp_path):
        """Test save_pretrained saves the model and artifacts when the model is loaded."""
        lm = PreTrainedMaskedLM()
        lm._model = mock_model

        with patch.object(lm, "save_artifacts") as mock_save_artifacts:
            lm.save_pretrained(tmp_path)

        mock_model.save_pretrained.assert_called_once_with(tmp_path)
        mock_save_artifacts.assert_called_once_with(tmp_path)

    def test_save_pretrained_without_model_only_saves_artifacts(self, tmp_path):
        """Test save_pretrained skips model saving when no model has been loaded."""
        lm = PreTrainedMaskedLM()

        with patch.object(lm, "save_artifacts") as mock_save_artifacts:
            lm.save_pretrained(tmp_path)

        mock_save_artifacts.assert_called_once_with(tmp_path)


class TestPreTrainedMaskedLMDeviceManagementWithLoadedModel:
    """Test to/half/float behavior once a model is loaded, complementing the no-model cases above."""

    def test_to_method_without_model_only_updates_device(self):
        """Test to() updates the recorded device even if no model has been loaded yet."""
        lm = PreTrainedMaskedLM(device="cpu")

        result = lm.to("cuda:0")

        assert result is lm
        assert lm.device == "cuda:0"
        assert lm.has_model is False

    def test_half_method_with_loaded_model(self, mock_model):
        """Test half() converts a loaded model to float16."""
        mock_model.half.return_value = mock_model
        lm = PreTrainedMaskedLM()
        lm._model = mock_model

        result = lm.half()

        assert result is lm
        mock_model.half.assert_called_once()

    def test_float_method_no_model(self):
        """Test float() is a no-op when no model has been loaded."""
        lm = PreTrainedMaskedLM()
        result = lm.float()
        assert result is lm


class TestPreTrainedMaskedLMLoadedModelProperties:
    """Test dtype/num_parameters/__repr__ once a model has been loaded."""

    def test_dtype_and_num_parameters_with_model(self):
        """Test dtype and num_parameters reflect the loaded model's real parameters."""
        real_model = torch.nn.Linear(4, 4)
        lm = PreTrainedMaskedLM()
        lm._model = real_model

        assert lm.dtype == real_model.weight.dtype
        assert lm.num_parameters == sum(p.numel() for p in real_model.parameters())

    def test_repr_with_loaded_model(self, mock_model):
        """Test __repr__ includes the loaded model class name and parameter count."""
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(2, 2))])
        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        lm._model = mock_model

        with patch.object(PreTrainedMaskedLM, "config", new=Mock(num_hidden_layers=12, hidden_size=768)):
            repr_str = repr(lm)

        assert "PreTrainedMaskedLM" in repr_str
        assert "loaded" in repr_str

    def test_repr_with_config_only_uses_architectures(self):
        """Test __repr__ falls back to config.architectures when no model is loaded."""
        lm = PreTrainedMaskedLM(model_name_or_path="bert-base-uncased")
        lm._config = Mock(architectures=["BertForMaskedLM"], num_hidden_layers=12, hidden_size=768)

        repr_str = repr(lm)

        assert "BertForMaskedLM" in repr_str
        assert "not loaded" in repr_str


@pytest.fixture
def mock_config():
    """Mock AutoConfig for testing."""
    config = Mock()
    config.model_type = "bert"
    config.num_hidden_layers = 12
    config.hidden_size = 768
    config.save_pretrained = Mock()
    return config


@pytest.fixture
def mock_tokenizer():
    """Mock PreTrainedTokenizer for testing."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.return_value = Mock(to=Mock(return_value="encoded_inputs"))
    tokenizer.decode.return_value = "decoded text"
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.to.return_value = model
    return model
