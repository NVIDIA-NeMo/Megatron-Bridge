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
Unit tests for AutoBridge automatic bridge selection and bridge functionality.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlamaConfig,
    LlavaProcessor,
    PreTrainedTokenizerFast,
)
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import (
    AutoBridge,
    _config_disables_mtp,
    _drop_readonly_config_properties,
    _model_omits_mtp,
    _mtp_source_key_prefixes,
    _resolve_pretrained_wrapper_cls,
    _saved_config_disables_mtp,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.masked_lm import PreTrainedMaskedLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource


try:
    from transformers import MiniMaxM3VLTextConfig
except ImportError:
    MiniMaxM3VLTextConfig = None


def create_mock_pretrained_causal_lm():
    """Helper function to create a mock PreTrainedCausalLM that passes isinstance checks."""

    class MockPreTrainedCausalLM(PreTrainedCausalLM):
        def __init__(self):
            pass  # Skip actual initialization

    return MockPreTrainedCausalLM()


def _save_minimal_fast_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        models.WordLevel(
            {
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<pad>": 3,
                "hello": 4,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def _make_fake_source(present):
    """Build a ``SafeTensorsStateSource`` stand-in for ``save_hf_weights`` tests.

    Uses ``Mock(spec=...)`` so the ``isinstance(source, SafeTensorsStateSource)``
    gate in ``save_hf_weights`` stays satisfied without bypassing the real
    ``__init__``. ``has_glob`` reports which source-key globs exist; the captured
    ``save_generator`` kwargs are exposed on ``source.save_generator_kwargs`` for
    assertions.
    """
    source = Mock(spec=SafeTensorsStateSource)
    source.save_generator_kwargs = None
    source.has_glob.side_effect = lambda pattern: pattern in present

    def _capture_save_generator(generator, path, **kwargs):
        source.save_generator_kwargs = kwargs

    source.save_generator.side_effect = _capture_save_generator
    return source


class TestAutoBridge:
    """Test cases for AutoBridge automatic selection and full bridge functionality."""

    @pytest.fixture
    def llama_config(self):
        """Create a sample Llama configuration matching the provided example."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 8192,
            "max_position_embeddings": 131072,
            "mlp_bias": False,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            "rope_theta": 500000.0,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.45.0.dev0",
            "use_cache": True,
            "vocab_size": 128256,
        }

    @pytest.fixture
    def llama_config_mock(self):
        """Create a mock Llama configuration."""
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        config.model_type = "llama"
        config.vocab_size = 32000
        config.hidden_size = 2048
        config.num_hidden_layers = 16
        config.num_attention_heads = 32
        config.auto_map = None
        return config

    @pytest.fixture
    def bert_config(self):
        """Create a mock BERT configuration with no supported task head (unsupported)."""
        config = Mock()
        config.architectures = ["BertModel"]
        config.model_type = "bert"
        return config

    @pytest.fixture
    def bert_masked_lm_config(self):
        """Create a mock BERT masked-LM configuration (supported via PreTrainedMaskedLM)."""
        config = Mock()
        config.architectures = ["BertForMaskedLM"]
        config.model_type = "bert"
        config.auto_map = None
        return config

    @pytest.fixture
    def gpt2_config(self):
        """Create a mock GPT2 configuration."""
        config = Mock()
        config.architectures = ["GPT2ForCausalLM", "GPT2LMHeadModel"]
        config.model_type = "gpt2"
        return config

    def test_from_hf_pretrained_with_unsupported_model(self, bert_config):
        """Test AutoBridge raises ValueError for unsupported models."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            # Setup mocks
            mock_safe_load_config.return_value = bert_config

            # Should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("bert-base-uncased")

            assert "Model architecture not supported by AutoBridge" in str(exc_info.value)
            assert "BertModel" in str(exc_info.value)

    def test_from_hf_pretrained_with_masked_lm_architecture_and_no_registered_bridge(self, bert_masked_lm_config):
        """A '*ForMaskedLM' architecture passes the allowlist but still needs a registered bridge."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            mock_safe_load_config.return_value = bert_masked_lm_config

            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("bert-base-uncased")

            # Distinguish from the "not supported by AutoBridge" allowlist failure above:
            # the architecture is allowlisted, but no MegatronModelBridge is registered for it.
            assert "is not yet supported" in str(exc_info.value)
            assert "BertForMaskedLM" in str(exc_info.value)

    def test_drop_readonly_config_properties(self):
        """Test auto-config synthesis drops properties HuggingFace configs cannot set."""

        class CustomConfig(PretrainedConfig):
            @property
            def layers_block_type(self):
                return ["mamba", "attention"]

        config_dict = {
            "hidden_size": 768,
            "layers_block_type": ["mamba", "attention"],
            "num_hidden_layers": 2,
        }

        filtered = _drop_readonly_config_properties(config_dict, CustomConfig)

        assert filtered == {
            "hidden_size": 768,
            "num_hidden_layers": 2,
        }
        assert config_dict["layers_block_type"] == ["mamba", "attention"]

    def test_from_pretrained_config_load_failure(self):
        """Test AutoBridge handles config loading failures gracefully."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            # Setup mock to raise exception
            mock_safe_load_config.side_effect = ValueError("Failed to load configuration: Config not found")

            # Should raise ValueError with helpful message
            with pytest.raises(ValueError) as exc_info:
                AutoBridge.from_hf_pretrained("invalid/path")

            assert "Failed to load configuration" in str(exc_info.value)
            assert "Config not found" in str(exc_info.value)

    def test_mtp_disabled_helpers(self, tmp_path):
        """Detect disabled MTP in object, nested, and saved HF configs."""
        assert _config_disables_mtp(None) is False
        assert _config_disables_mtp(Mock(num_nextn_predict_layers=0)) is True
        assert _config_disables_mtp({"num_nextn_predict_layers": None, "text_config": {"mtp_num_layers": "0"}}) is True
        assert _config_disables_mtp({"text_config": {"mtp_num_hidden_layers": 0}}) is True
        assert _config_disables_mtp(Mock(num_nextn_predict_layers=1)) is False
        assert _config_disables_mtp({"mtp_num_layers": "2"}) is False
        assert _saved_config_disables_mtp(tmp_path) is False

        with open(tmp_path / "config.json", "w") as f:
            json.dump({"num_nextn_predict_layers": 0}, f)

        assert _saved_config_disables_mtp(tmp_path) is True

    def test_model_omits_mtp(self):
        """A built model with a falsy mtp_num_layers has no MTP head."""
        assert _model_omits_mtp(None) is False
        # Unset attribute -> unknown -> do not assume omitted.
        assert _model_omits_mtp(SimpleNamespace()) is False
        # SkyRL forces mtp_num_layers=None -> head omitted from export.
        assert _model_omits_mtp(Mock(mtp_num_layers=None)) is True
        assert _model_omits_mtp(Mock(mtp_num_layers=0)) is True
        assert _model_omits_mtp(Mock(mtp_num_layers=1)) is False

    def test_mtp_source_key_prefixes(self):
        """Resolve the MTP/nextn source-key prefixes to strip per architecture."""

        def src(*present_globs):
            present = set(present_globs)
            return Mock(has_glob=lambda pattern: pattern in present)

        # DeepSeek-style: dedicated mtp.* prefix.
        assert _mtp_source_key_prefixes(src("mtp.*"), {}) == ("mtp.",)

        # GLM glm4_moe_lite: nextn layer stored at index == num_hidden_layers.
        glm_src = src("model.layers.47.*")
        assert _mtp_source_key_prefixes(glm_src, {"num_hidden_layers": 47}) == ("model.layers.47.",)

        # Nested text_config carries num_hidden_layers.
        assert _mtp_source_key_prefixes(glm_src, {"text_config": {"num_hidden_layers": 47}}) == ("model.layers.47.",)

        # No matching source keys -> nothing to strip.
        assert _mtp_source_key_prefixes(src(), {"num_hidden_layers": 47}) == ()

        # Both prefixes present.
        both = src("mtp.*", "model.layers.47.*")
        assert _mtp_source_key_prefixes(both, {"num_hidden_layers": 47}) == ("mtp.", "model.layers.47.")

    def test_save_hf_weights_strips_nextn_prefix_when_mtp_omitted(self, tmp_path):
        """Regression: a model built without an MTP head must strip the GLM nextn
        layer prefix from the source map before streaming save.

        This is the actual bug being fixed (45/48-shard checkpoint dropping
        boundary shards on GLM-4.x glm4_moe_lite). Unlike the helper-level tests,
        this asserts the orchestration in ``save_hf_weights`` wires the stripped
        prefixes through to ``save_generator``. It fails if the
        ``_model_omits_mtp(...)`` branch is removed, because the HF/saved configs
        here do *not* explicitly disable MTP — the only signal is the built
        model omitting the head.
        """
        source = _make_fake_source(present={"model.layers.47.*", "model.layers.46.*"})
        # Built megatron model omits the MTP head (SkyRL forces mtp_num_layers=None).
        self._run_save_hf_weights(source, tmp_path, mtp_num_layers=None)

        assert source.save_generator_kwargs is not None
        assert source.save_generator_kwargs["ignored_source_key_prefixes"] == ("model.layers.47.",)

    def test_save_hf_weights_keeps_all_keys_when_mtp_enabled(self, tmp_path):
        """Counterpart: when the model keeps its MTP head, nothing is stripped.

        Also guards the ``if mtp_disabled`` gate: if a future refactor drops the
        gate and always calls ``_mtp_source_key_prefixes``, the helper would strip
        the real ``model.layers.47.`` layer here and this assertion would fail.
        """
        source = _make_fake_source(present={"model.layers.47.*"})
        self._run_save_hf_weights(source, tmp_path, mtp_num_layers=1)

        assert source.save_generator_kwargs["ignored_source_key_prefixes"] is None

    def test_save_hf_weights_streams_config_only_destination_shards(self, tmp_path):
        hf_config = LlamaConfig(architectures=["LlamaForCausalLM"])
        bridge_obj = object.__new__(AutoBridge)
        bridge_obj.hf_pretrained = hf_config
        expected_keys = {"model.embed_tokens.weight", "lm_head.weight"}

        class ConfigOnlyBridge:
            def stream_weights_megatron_to_hf(self, *_args, **_kwargs):
                return iter(
                    [
                        ("model.embed_tokens.weight", torch.ones(2, 2)),
                        ("lm_head.weight", torch.zeros(2, 2)),
                    ]
                )

            def get_expected_hf_export_keys(self, config):
                assert config is hf_config
                return expected_keys

        fake_model_bridge = ConfigOnlyBridge()
        model_instance = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=None))
        captured = []

        def capture_generator(generator, path, **kwargs):
            captured.extend(generator)
            assert path == tmp_path
            assert kwargs == {
                "distributed_save": True,
                "save_every_n_ranks": 2,
                "expected_keys": expected_keys,
            }

        with (
            patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock, return_value=fake_model_bridge),
            patch.object(AutoBridge, "_get_model_instance", return_value=model_instance),
            patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
            patch(
                "megatron.bridge.models.conversion.auto_bridge.save_generator_to_safetensors",
                side_effect=capture_generator,
            ) as mock_stream_save,
        ):
            bridge_obj.save_hf_weights(
                [Mock()],
                tmp_path,
                show_progress=False,
                distributed_save=True,
                save_every_n_ranks=2,
            )

        assert [name for name, _tensor in captured] == ["model.embed_tokens.weight", "lm_head.weight"]
        mock_stream_save.assert_called_once()

    def _run_save_hf_weights(self, source, tmp_path, *, mtp_num_layers):
        """Drive ``save_hf_weights`` with a stubbed bridge/model so the only
        behavior under test is the MTP prefix-resolution wiring.

        ``num_hidden_layers=47`` with no MTP-disable field means the export
        decision hinges purely on whether the *built* model omits the head
        (``mtp_num_layers``).
        """
        hf_pretrained = create_mock_pretrained_causal_lm()
        # HF config carries layer count but does NOT set any MTP-disable field.
        hf_pretrained.config = SimpleNamespace(num_hidden_layers=47)
        model_instance = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=mtp_num_layers))

        bridge_obj = object.__new__(AutoBridge)
        bridge_obj.hf_pretrained = hf_pretrained

        fake_model_bridge = Mock()
        fake_model_bridge.stream_weights_megatron_to_hf.return_value = iter([])

        with (
            # ``state`` is a read-only property on PreTrainedBase, so patch it
            # rather than assigning to the instance.
            patch.object(
                type(hf_pretrained),
                "state",
                new_callable=PropertyMock,
                return_value=SimpleNamespace(source=source),
            ),
            patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_bridge,
            patch.object(AutoBridge, "_get_model_instance", return_value=model_instance),
            patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
        ):
            mock_bridge.return_value = fake_model_bridge
            bridge_obj.save_hf_weights([Mock()], tmp_path, show_progress=False)

    def test_can_handle_supported_model(self, llama_config_mock):
        """Test can_handle returns True for supported models."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            mock_safe_load_config.return_value = llama_config_mock

            assert AutoBridge.can_handle("meta-llama/Meta-Llama-3-8B") is True
            mock_safe_load_config.assert_called_with("meta-llama/Meta-Llama-3-8B", trust_remote_code=False)

    def test_can_handle_unsupported_model(self, bert_config):
        """Test can_handle returns False for unsupported models."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            mock_safe_load_config.return_value = bert_config

            assert AutoBridge.can_handle("bert-base-uncased") is False

    def test_can_handle_masked_lm_architecture_and_no_registered_bridge(self, bert_masked_lm_config):
        """'*ForMaskedLM' passes the allowlist, but can_handle still returns False without a registered bridge.

        BERT has no registered MegatronModelBridge yet, so even though the architecture is
        allowlisted (see SUPPORTED_HF_ARCHITECTURES), can_handle must not report True for a model
        that from_hf_pretrained would then fail to load. Once a BERT bridge is registered, this
        should be updated to assert True.
        """
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            mock_safe_load_config.return_value = bert_masked_lm_config

            assert AutoBridge.can_handle("bert-base-uncased") is False

    def test_can_handle_invalid_path(self):
        """Test can_handle returns False for invalid paths."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            mock_safe_load_config.side_effect = Exception("Not found")

            assert AutoBridge.can_handle("invalid/path") is False

    # Test core bridge functionality (from original AutoBridge tests)
    def test_from_hf_pretrained_with_model_id(self):
        """Test from_hf_pretrained with model ID string."""
        # This test checks that from_hf_pretrained creates correct bridge instance
        # We'll use a mock pretrained model
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]  # Use a real architecture
        mock_model.config = mock_config

        with patch(
            "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch(
                "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
            ) as mock_safe_load_config:
                mock_safe_load_config.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(AutoBridge, "_validate_config"):
                    # Call from_hf_pretrained
                    model_id = "gpt2"
                    result = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)

                # Assertions
                assert isinstance(result, AutoBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(model_id, trust_remote_code=True)

    def test_from_hf_pretrained_dispatches_masked_lm_architecture_to_pretrained_masked_lm(self):
        """'*ForMaskedLM' architectures are loaded via PreTrainedMaskedLM, not PreTrainedCausalLM."""
        mock_model = Mock(spec=PreTrainedMaskedLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["BertForMaskedLM"]
        mock_model.config = mock_config

        with (
            patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedMaskedLM.from_pretrained"
            ) as mock_masked_lm_from_pretrained,
            patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_causal_lm_from_pretrained,
            patch(
                "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
            ) as mock_safe_load_config,
            patch.object(AutoBridge, "_validate_config"),
        ):
            mock_masked_lm_from_pretrained.return_value = mock_model
            mock_safe_load_config.return_value = mock_config

            result = AutoBridge.from_hf_pretrained("bert-base-uncased")

            assert isinstance(result, AutoBridge)
            assert result.hf_pretrained == mock_model
            mock_masked_lm_from_pretrained.assert_called_once_with("bert-base-uncased")
            mock_causal_lm_from_pretrained.assert_not_called()

    def test_resolve_pretrained_wrapper_cls(self):
        """_resolve_pretrained_wrapper_cls selects PreTrainedMaskedLM only for '*ForMaskedLM'."""
        masked_lm_config = SimpleNamespace(architectures=["BertForMaskedLM"])
        assert _resolve_pretrained_wrapper_cls(masked_lm_config) is PreTrainedMaskedLM

        causal_lm_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        assert _resolve_pretrained_wrapper_cls(causal_lm_config) is PreTrainedCausalLM

        no_arch_config = SimpleNamespace(architectures=[])
        assert _resolve_pretrained_wrapper_cls(no_arch_config) is PreTrainedCausalLM

    def test_from_hf_pretrained_passes_causal_wrapper_to_vlm_provider_bridge(self):
        """Test VLM provider construction receives the actual AutoBridge wrapper type."""
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        vlm_config = Mock(spec=PretrainedConfig)
        vlm_config.architectures = ["Qwen2_5_VLForConditionalGeneration"]
        wrapper = PreTrainedCausalLM(model_name_or_path=model_id)
        wrapper.config = vlm_config

        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with (
            patch(
                "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry",
                return_value=vlm_config,
            ),
            patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained",
                return_value=wrapper,
            ),
            patch.object(AutoBridge, "_model_bridge", mock_model_bridge),
        ):
            bridge = AutoBridge.from_hf_pretrained(model_id)
            provider = bridge.to_megatron_provider(load_weights=False)

        assert provider is mock_provider
        assert isinstance(bridge.hf_pretrained, PreTrainedCausalLM)
        assert bridge.hf_pretrained.config is vlm_config
        mock_model_bridge.provider_bridge.assert_called_once_with(wrapper)

    def test_from_hf_config_passes_causal_wrapper_to_vlm_provider_bridge(self):
        """Test config-only VLM provider construction receives a config-backed causal wrapper."""
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        vlm_config = PretrainedConfig(name_or_path=model_id)
        vlm_config.architectures = ["Qwen2_5_VLForConditionalGeneration"]

        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge.from_hf_config(vlm_config)
            provider = bridge.to_megatron_provider(load_weights=False)

        provider_input = mock_model_bridge.provider_bridge.call_args.args[0]
        assert provider is mock_provider
        assert isinstance(provider_input, PreTrainedCausalLM)
        assert not provider_input.has_model
        assert not hasattr(provider_input, "state")
        assert provider_input.config is vlm_config
        assert provider_input.model_name_or_path == model_id

    def test_from_pretrained_with_additional_kwargs(self):
        """Test from_pretrained with various kwargs."""
        # Setup mocks
        mock_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_model.config = mock_config

        with patch(
            "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
        ) as mock_from_pretrained:
            # Set up the from_pretrained class method properly
            mock_from_pretrained.return_value = mock_model

            with patch(
                "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
            ) as mock_safe_load_config:
                mock_safe_load_config.return_value = mock_config

                # Skip architecture validation for this test
                with patch.object(AutoBridge, "_validate_config"):
                    # Call with multiple kwargs
                    result = AutoBridge.from_hf_pretrained(
                        "model-id",
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2",
                    )

                # Assertions
                assert isinstance(result, AutoBridge)
                assert result.hf_pretrained == mock_model
                mock_from_pretrained.assert_called_once_with(
                    "model-id",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )

    def test_to_megatron_provider_basic(self, llama_config):
        """Test basic to_megatron_provider conversion."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = LlamaConfig(**llama_config)

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert
            bridge = AutoBridge(mock_hf_model)
            result = bridge.to_megatron_provider(load_weights=False)

            # Assertions
            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_hf_model)

    def test_to_megatron_provider_with_different_model_types(self):
        """Test to_megatron_provider with different model architectures."""
        # Test with GPT2 model
        mock_gpt2_model = Mock(spec=PreTrainedCausalLM)
        mock_gpt2_model.config = Mock(model_type="gpt2")

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_gpt2_model)
            result = bridge.to_megatron_provider(load_weights=False)

            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_gpt2_model)

    def test_to_megatron_provider_with_custom_kwargs(self, llama_config):
        """Test to_megatron_provider with custom keyword arguments."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = LlamaConfig(**llama_config)

        # Mock model bridge
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_provider.register_pre_wrap_hook = Mock()
        mock_model_bridge.provider_bridge.return_value = mock_provider
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            # Create bridge and convert with load_weights=True
            bridge = AutoBridge(mock_hf_model)
            result = bridge.to_megatron_provider(load_weights=True)

            # Assertions
            assert result == mock_provider
            mock_model_bridge.provider_bridge.assert_called_once_with(mock_hf_model)
            # Check that a pre-wrap hook was registered for loading weights
            mock_provider.register_pre_wrap_hook.assert_called_once()

    def test_to_megatron_provider_sets_hf_model_id_from_path(self):
        """to_megatron_provider tags providers with the provided HF path."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_provider.hf_model_id = None
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            provider = bridge.to_megatron_provider(load_weights=False, hf_path="local/hf/path")

        assert provider.hf_model_id == "local/hf/path"

    def test_to_megatron_provider_sets_hf_model_id_from_pretrained(self):
        """to_megatron_provider falls back to HF model name_or_path."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        type(mock_hf_model).model_name_or_path = PropertyMock(return_value="hf/model-id")
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_provider.hf_model_id = None
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            provider = bridge.to_megatron_provider(load_weights=False)

        assert provider.hf_model_id == "hf/model-id"

    def test_to_megatron_provider_persists_hf_model_revision(self):
        """to_megatron_provider records the immutable revision used for config loading."""
        revision = "b968826d9c46dd6066d109eabc6255188de91218"  # pragma: allowlist secret
        mock_hf_model = PreTrainedCausalLM(model_name_or_path="hf/model-id", revision=revision)
        mock_model_bridge = Mock()
        mock_provider = Mock(spec=GPTModelProvider)
        mock_provider.hf_model_id = None
        mock_provider.hf_model_revision = None
        mock_model_bridge.provider_bridge.return_value = mock_provider

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            provider = bridge.to_megatron_provider(load_weights=False)

        assert provider.hf_model_id == "hf/model-id"
        assert provider.hf_model_revision == revision

    def test_get_hf_model_id_from_checkpoint_delegates(self):
        """AutoBridge helper delegates to checkpoint utilities."""
        with patch(
            "megatron.bridge.training.utils.checkpoint_utils.get_hf_model_id_from_checkpoint",
            return_value="delegated/model",
        ) as mock_infer:
            result = AutoBridge.get_hf_model_id_from_checkpoint("/tmp/checkpoint")

        assert result == "delegated/model"
        mock_infer.assert_called_once_with("/tmp/checkpoint")

    def test_to_megatron_provider_error_handling(self):
        """Test to_megatron_provider error handling."""
        # Setup mock to raise an exception
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        # Mock model bridge to raise error
        mock_model_bridge = Mock()
        mock_model_bridge.provider_bridge.side_effect = ValueError("Unsupported model type")

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)

            # Should propagate the exception
            with pytest.raises(ValueError, match="Unsupported model type"):
                bridge.to_megatron_provider()

    def test_bridge_instance_creation(self):
        """Test AutoBridge instance creation."""
        # Test with PreTrainedCausalLM
        mock_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge(mock_model)

        # Should have the expected methods
        assert hasattr(bridge, "from_hf_pretrained")
        assert hasattr(bridge, "to_megatron_provider")
        assert hasattr(bridge, "load_hf_weights")
        assert hasattr(bridge, "export_hf_weights")
        assert hasattr(bridge, "save_hf_pretrained")
        assert bridge.hf_pretrained == mock_model

        # Test with PretrainedConfig
        mock_config = Mock(spec=PretrainedConfig)
        bridge_config = AutoBridge(mock_config)
        assert bridge_config.hf_pretrained == mock_config

        # Test with PreTrainedMaskedLM
        mock_masked_lm = Mock(spec=PreTrainedMaskedLM)
        bridge_masked_lm = AutoBridge(mock_masked_lm)
        assert bridge_masked_lm.hf_pretrained == mock_masked_lm

        # Test with invalid type
        with pytest.raises(
            ValueError,
            match="hf_pretrained must be a PreTrainedCausalLM, PreTrainedMaskedLM, or PretrainedConfig instance",
        ):
            AutoBridge("invalid")

        # from_hf_pretrained should be a classmethod
        import inspect

        assert inspect.ismethod(AutoBridge.from_hf_pretrained)

    def test_pretrained_wrapper_cls_property(self):
        """Test _pretrained_wrapper_cls resolves the wrapper class for each hf_pretrained kind."""
        # A PreTrainedMaskedLM instance resolves directly, regardless of its config.
        mock_masked_lm = Mock(spec=PreTrainedMaskedLM)
        bridge = AutoBridge(mock_masked_lm)
        assert bridge._pretrained_wrapper_cls is PreTrainedMaskedLM

        # A PreTrainedCausalLM instance resolves directly, regardless of its config.
        mock_causal_lm = Mock(spec=PreTrainedCausalLM)
        bridge = AutoBridge(mock_causal_lm)
        assert bridge._pretrained_wrapper_cls is PreTrainedCausalLM

        # A bare PretrainedConfig falls back to resolving from the config's architectures.
        masked_lm_config = Mock(spec=PretrainedConfig)
        masked_lm_config.architectures = ["BertForMaskedLM"]
        bridge = AutoBridge(masked_lm_config)
        assert bridge._pretrained_wrapper_cls is PreTrainedMaskedLM

        causal_lm_config = Mock(spec=PretrainedConfig)
        causal_lm_config.architectures = ["LlamaForCausalLM"]
        bridge = AutoBridge(causal_lm_config)
        assert bridge._pretrained_wrapper_cls is PreTrainedCausalLM

    def test_from_hf_config(self):
        """Test creating bridge from config only."""
        # Create a mock config
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["GPT2LMHeadModel"]

        # Skip architecture validation for this test
        with patch.object(AutoBridge, "_validate_config"):
            bridge = AutoBridge.from_hf_config(config)
            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == config

    def test_from_hf_config_invalid_architecture(self):
        """Test from_hf_config with unsupported architecture."""
        config = Mock(spec=PretrainedConfig)
        config.architectures = ["BertModel"]  # No supported task-head suffix

        with pytest.raises(ValueError, match="Model architecture not supported by AutoBridge"):
            AutoBridge.from_hf_config(config)

    def test_from_auto_config_happy_path(self, tmp_path):
        """from_auto_config synthesizes config and tags bridge with source model id."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_config.yaml").write_text("dummy: true\n")

        mock_hf_cfg = Mock()
        mock_hf_cfg.to_dict.return_value = {"vocab_size": 32000}

        first_bridge = Mock()
        first_bridge._model_bridge.megatron_to_hf_config.return_value = {"vocab_size": 64000}
        second_bridge = Mock()

        hf_model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_hf_cfg) as mock_auto_cfg:
            with patch(
                "megatron.bridge.training.model_load_save.load_model_config",
                return_value=(Mock(name="megatron_cfg"), None),
            ) as mock_load_cfg:
                with patch(
                    "megatron.bridge.models.conversion.utils.conform_config_to_reference",
                    return_value={"vocab_size": 64000},
                ) as mock_conform:
                    with patch.object(AutoBridge, "from_hf_config", side_effect=[first_bridge, second_bridge]):
                        bridge = AutoBridge.from_auto_config(str(ckpt_dir), hf_model_id)

        assert bridge is second_bridge
        assert second_bridge.hf_model_id == hf_model_id
        mock_auto_cfg.assert_called_once_with(hf_model_id, trust_remote_code=False)
        mock_load_cfg.assert_called_once_with(str(ckpt_dir))
        mock_conform.assert_called_once_with({"vocab_size": 64000}, {"vocab_size": 32000})

    def test_from_auto_config_switches_to_export_model_type(self, tmp_path):
        """A component bridge can synthesize a different standalone HF config class."""
        if MiniMaxM3VLTextConfig is None:
            pytest.skip("Native MiniMax-M3 text config requires Transformers 5.12.1 or newer.")

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "run_config.yaml").write_text("dummy: true\n")

        source_config = SimpleNamespace(
            model_type="minimax_m3_vl",
            to_dict=lambda: {"model_type": "minimax_m3_vl", "text_config": {}},
        )
        first_bridge = Mock()
        first_bridge._model_bridge.megatron_to_hf_config.return_value = {
            "architectures": ["MiniMaxM3VLForCausalLM"],
            "model_type": "minimax_m3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 32,
            "dense_intermediate_size": 128,
            "shared_intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "vocab_size": 1024,
            "bos_token_id": 1000,
            "eos_token_id": 1001,
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "layer_types": ["full_attention"] * 4,
        }
        second_bridge = Mock()
        seen_configs = []

        def capture_config(config):
            seen_configs.append(config)
            return first_bridge if len(seen_configs) == 1 else second_bridge

        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=source_config),
            patch(
                "megatron.bridge.training.model_load_save.load_model_config",
                return_value=(Mock(name="megatron_cfg"), None),
            ),
            patch("megatron.bridge.models.conversion.utils.conform_config_to_reference") as mock_conform,
            patch.object(AutoBridge, "from_hf_config", side_effect=capture_config),
        ):
            result = AutoBridge.from_auto_config(str(ckpt_dir), "MiniMaxAI/MiniMax-M3")

        assert result is second_bridge
        assert isinstance(seen_configs[1], MiniMaxM3VLTextConfig)
        assert seen_configs[1].model_type == "minimax_m3_vl_text"
        assert seen_configs[1].layer_types == ["full_attention"] * 4
        mock_conform.assert_not_called()

    def test_from_auto_config_uses_latest_iter_run_config(self, tmp_path):
        """from_auto_config falls back to latest iter_* directory for run_config.yaml."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "iter_0000001").mkdir()
        iter_latest = ckpt_dir / "iter_0000003"
        iter_latest.mkdir()
        (iter_latest / "run_config.yaml").write_text("dummy: true\n")

        mock_hf_cfg = Mock()
        mock_hf_cfg.to_dict.return_value = {"vocab_size": 32000}
        first_bridge = Mock()
        first_bridge._model_bridge.megatron_to_hf_config.return_value = {"vocab_size": 64000}
        second_bridge = Mock()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_hf_cfg):
            with patch(
                "megatron.bridge.training.model_load_save.load_model_config",
                return_value=(Mock(name="megatron_cfg"), None),
            ) as mock_load_cfg:
                with patch(
                    "megatron.bridge.models.conversion.utils.conform_config_to_reference",
                    return_value={"vocab_size": 64000},
                ):
                    with patch.object(AutoBridge, "from_hf_config", side_effect=[first_bridge, second_bridge]):
                        AutoBridge.from_auto_config(str(ckpt_dir), "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

        mock_load_cfg.assert_called_once_with(str(iter_latest))

    def test_from_auto_config_missing_checkpoint_path(self):
        """from_auto_config fails with clear message for nonexistent checkpoint root."""
        with pytest.raises(FileNotFoundError, match="Megatron checkpoint not found"):
            AutoBridge.from_auto_config("/definitely/not/a/path", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

    def test_from_auto_config_missing_run_config(self, tmp_path):
        """from_auto_config fails if no run_config.yaml is found."""
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "iter_0000001").mkdir()

        with pytest.raises(FileNotFoundError, match="Could not find run_config.yaml"):
            AutoBridge.from_auto_config(str(ckpt_dir), "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

    def test_supports_method(self):
        """Test the supports class method."""
        # Supported config
        config = Mock()
        config.architectures = ["LlamaForCausalLM"]
        assert AutoBridge.supports(config) is True

        # Multiple architectures, one supported
        config.architectures = ["LlamaModel", "LlamaForCausalLM"]
        assert AutoBridge.supports(config) is True

        # '*ForMaskedLM' is an allowlisted non-causal architecture
        config.architectures = ["BertForMaskedLM"]
        assert AutoBridge.supports(config) is True

        # Bare encoder with no supported task head
        config.architectures = ["BertModel"]
        assert AutoBridge.supports(config) is False

        # No architectures
        config.architectures = []
        assert AutoBridge.supports(config) is False

        # Missing architectures attribute
        config_no_arch = Mock(spec=[])
        assert AutoBridge.supports(config_no_arch) is False

    def test_list_supported_models(self):
        """Test listing supported models."""
        # Since this method looks at internal dispatch registry,
        # we'll just test that it returns a list
        with patch("megatron.bridge.models.conversion.auto_bridge.model_bridge") as mock_bridge:
            # Mock to avoid AttributeError
            mock_bridge.get_model_bridge = Mock()
            mock_bridge.get_model_bridge._exact_types = {}
            supported = AutoBridge.list_supported_models()
            assert isinstance(supported, list)
            # The list might be empty if no models are registered in test environment

    def test_load_hf_weights(self):
        """Test loading weights into a Megatron model."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]  # List of model instances

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            bridge.load_hf_weights(mock_megatron_model)

            mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                mock_hf_model, mock_megatron_model, allowed_mismatched_params=None
            )

    def test_load_hf_weights_with_allowed_mismatched_params(self):
        """Test loading weights with allowed_mismatched_params."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)
            whitelist = ["*.bias", "layer.1.weight"]
            bridge.load_hf_weights(mock_megatron_model, allowed_mismatched_params=whitelist)

            mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                mock_hf_model, mock_megatron_model, allowed_mismatched_params=whitelist
            )

    def test_load_hf_weights_from_path(self):
        """Test loading weights from a different path."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        # Create bridge first, then patch the from_pretrained method
        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)

            # Now patch the from_pretrained method
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_from_pretrained:
                mock_loaded_model = Mock(spec=PreTrainedCausalLM)
                mock_from_pretrained.return_value = mock_loaded_model

                bridge.load_hf_weights(mock_megatron_model, "./custom_model")

                mock_from_pretrained.assert_called_once_with("./custom_model", trust_remote_code=False)
                mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                    mock_loaded_model,
                    mock_megatron_model,
                    allowed_mismatched_params=None,
                )

    def test_load_hf_weights_from_path_with_masked_lm(self):
        """Test loading weights from a path dispatches to PreTrainedMaskedLM when hf_pretrained is one."""
        mock_hf_model = Mock(spec=PreTrainedMaskedLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        mock_model_bridge = Mock()
        mock_model_bridge.load_weights_hf_to_megatron = Mock()

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge(mock_hf_model)

            with (
                patch(
                    "megatron.bridge.models.conversion.auto_bridge.PreTrainedMaskedLM.from_pretrained"
                ) as mock_masked_lm_from_pretrained,
                patch(
                    "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
                ) as mock_causal_lm_from_pretrained,
            ):
                mock_loaded_model = Mock(spec=PreTrainedMaskedLM)
                mock_masked_lm_from_pretrained.return_value = mock_loaded_model

                bridge.load_hf_weights(mock_megatron_model, "./custom_model")

                mock_masked_lm_from_pretrained.assert_called_once_with("./custom_model", trust_remote_code=False)
                mock_causal_lm_from_pretrained.assert_not_called()
                mock_model_bridge.load_weights_hf_to_megatron.assert_called_once_with(
                    mock_loaded_model,
                    mock_megatron_model,
                    allowed_mismatched_params=None,
                )

    def test_load_hf_weights_no_path_config_only(self):
        """Test load_hf_weights fails when bridge has config only and no path provided."""
        mock_config = Mock(spec=PretrainedConfig)
        bridge = AutoBridge(mock_config)

        with pytest.raises(
            ValueError,
            match="hf_path is required when hf_pretrained is not a PreTrainedCausalLM",
        ):
            bridge.load_hf_weights([Mock()])

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    def test_save_hf_pretrained(self, mock_is_init, mock_is_avail, mock_barrier, mock_get_rank):
        """Test saving a model in HuggingFace format."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.save_artifacts = Mock()
        mock_hf_model.state = Mock()
        mock_hf_model.state.source = Mock(spec=["save_generator"])

        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

        mock_hf_model.state.source = Mock(spec=SafeTensorsStateSource)
        mock_hf_model.state.source.save_generator = Mock()

        mock_megatron_model = [Mock()]

        with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = AutoBridge(mock_hf_model)

            # Mock _model_bridge to have no ADDITIONAL_FILE_PATTERNS
            with patch.object(
                type(bridge),
                "_model_bridge",
                PropertyMock(return_value=Mock(ADDITIONAL_FILE_PATTERNS=None)),
            ):
                bridge.save_hf_pretrained(mock_megatron_model, "./output_dir")

                # Check artifacts were saved on rank 0
                mock_hf_model.save_artifacts.assert_called_once_with(
                    "./output_dir", original_source_path=None, additional_files=None
                )
                mock_save_hf_weights.assert_called_once_with(
                    mock_megatron_model,
                    "./output_dir",
                    True,
                    True,
                    merge_adapter_weights=True,
                    distributed_save=False,
                    save_every_n_ranks=1,
                    weight_dtype=None,
                )

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only(self, _mock_dist_avail, _mock_dist_init, tmp_path):
        """Config-only save without a reference writes config.json and calls save_hf_weights."""
        bridge = AutoBridge(PretrainedConfig())

        with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge.save_hf_pretrained([Mock()], str(tmp_path))

        assert (tmp_path / "config.json").exists()
        mock_save_hf_weights.assert_called_once()

    @pytest.mark.parametrize("artifact_source", ["hf_model_id", "source_path"])
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only_saves_reference_artifacts(
        self, _mock_dist_avail, _mock_dist_init, tmp_path, artifact_source
    ):
        """Config-only save preserves processor artifacts without loading reference weights."""
        source_path = tmp_path / "source"
        output_path = tmp_path / "output"
        source_path.mkdir()

        source_config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            max_position_embeddings=2048,
            vocab_size=5,
        )
        source_config.save_pretrained(source_path)
        tokenizer = _save_minimal_fast_tokenizer(source_path)
        processor = LlavaProcessor(image_processor=CLIPImageProcessor(), tokenizer=tokenizer)
        processor.save_pretrained(source_path)

        synthesized_config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            max_position_embeddings=4096,
            vocab_size=5,
        )
        bridge = AutoBridge(synthesized_config)
        save_kwargs = {}
        if artifact_source == "hf_model_id":
            bridge.hf_model_id = str(source_path)
        else:
            save_kwargs["source_path"] = source_path

        with patch(
            "megatron.bridge.models.hf_pretrained.causal_lm.AutoModelForCausalLM.from_pretrained"
        ) as mock_load_weights:
            with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
                bridge.save_hf_pretrained([Mock()], output_path, **save_kwargs)

        exported_tokenizer = AutoTokenizer.from_pretrained(output_path, local_files_only=True)
        exported_processor = AutoProcessor.from_pretrained(output_path, local_files_only=True)
        assert exported_tokenizer("hello")["input_ids"] == [4]
        assert isinstance(exported_processor, LlavaProcessor)
        assert (output_path / "tokenizer.json").is_file()
        assert (output_path / "processor_config.json").is_file()
        assert json.loads((output_path / "config.json").read_text())["max_position_embeddings"] == 4096
        mock_load_weights.assert_not_called()
        mock_save_hf_weights.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only_uses_reference_artifact_saver(self, _mock_dist_avail, _mock_dist_init):
        """Config-only save delegates all artifacts and model-specific options to the lazy HF wrapper."""
        config = LlamaConfig(architectures=["LlamaForCausalLM"])
        bridge = AutoBridge(config)
        bridge.hf_model_id = "some-org/some-model"

        mock_artifact_source = Mock(spec=PreTrainedCausalLM)
        mock_model_bridge = Mock(ADDITIONAL_FILE_PATTERNS=["chat_template.jinja"])
        mock_model_bridge.get_hf_tokenizer_kwargs.return_value = {"use_fast": True}

        with patch.object(
            type(bridge),
            "_model_bridge",
            PropertyMock(return_value=mock_model_bridge),
        ):
            with patch.object(
                PreTrainedCausalLM,
                "from_pretrained",
                return_value=mock_artifact_source,
            ) as mock_from_pretrained:
                with patch.object(AutoBridge, "save_hf_weights"):
                    bridge.save_hf_pretrained(
                        [Mock()],
                        "./output_dir",
                        source_path="./custom_files",
                    )

        mock_from_pretrained.assert_called_once_with(
            "some-org/some-model",
            use_fast=True,
            trust_remote_code=False,
        )
        assert mock_artifact_source.config is config
        mock_artifact_source.save_artifacts.assert_called_once_with(
            "./output_dir",
            original_source_path="./custom_files",
            additional_files=["chat_template.jinja"],
        )

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only_honors_export_artifact_policy(self, _mock_dist_avail, _mock_dist_init):
        """A standalone component export can omit source processor and custom-code artifacts."""

        class TextOnlyBridge:
            ADDITIONAL_FILE_PATTERNS = ["tokenizer_config.json"]
            HF_EXPORT_OPTIONAL_ARTIFACTS = ("generation_config",)
            HF_EXPORT_TRUST_REMOTE_CODE = False
            SUPPORTS_HF_PRETRAINED_EXPORT = True

            @staticmethod
            def get_hf_tokenizer_kwargs():
                return {}

        config = LlamaConfig(architectures=["LlamaForCausalLM"])
        bridge = AutoBridge(config)
        bridge.hf_model_id = "some-org/composite-model"
        bridge.trust_remote_code = True
        mock_artifact_source = Mock(spec=PreTrainedCausalLM)

        with patch.object(
            type(bridge),
            "_model_bridge",
            PropertyMock(return_value=TextOnlyBridge()),
        ):
            with patch.object(
                PreTrainedCausalLM,
                "from_pretrained",
                return_value=mock_artifact_source,
            ) as mock_from_pretrained:
                with patch.object(AutoBridge, "save_hf_weights"):
                    bridge.save_hf_pretrained([Mock()], "./output_dir")

        mock_from_pretrained.assert_called_once_with(
            "some-org/composite-model",
            trust_remote_code=False,
        )
        assert mock_artifact_source.OPTIONAL_ARTIFACTS == ["generation_config"]
        mock_artifact_source.save_artifacts.assert_called_once_with(
            "./output_dir",
            original_source_path=None,
            additional_files=["tokenizer_config.json"],
        )

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only_strips_auto_map_without_remote_code(
        self, _mock_dist_avail, _mock_dist_init, tmp_path
    ):
        """Config-only save omits stale remote-code metadata when remote code is not preserved."""
        config = LlamaConfig(architectures=["LlamaForCausalLM"])
        config.auto_map = {
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
        }
        bridge = AutoBridge(config)

        with patch.object(AutoBridge, "save_hf_weights"):
            bridge.save_hf_pretrained([Mock()], str(tmp_path))

        saved_config = json.loads((tmp_path / "config.json").read_text())
        assert "auto_map" not in saved_config

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.is_available", return_value=False)
    def test_save_hf_pretrained_config_only_preserves_auto_map_with_remote_code(
        self, _mock_dist_avail, _mock_dist_init, tmp_path
    ):
        """Config-only save keeps auto_map and copies code when remote code is preserved."""
        source_path = tmp_path / "source"
        output_path = tmp_path / "output"
        source_path.mkdir()
        _save_minimal_fast_tokenizer(source_path)
        (source_path / "modeling_custom.py").write_text("# custom modeling code\n")

        config = LlamaConfig(architectures=["LlamaForCausalLM"])
        config.auto_map = {
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
        }
        bridge = AutoBridge(config)
        bridge.hf_model_id = str(source_path)
        bridge.trust_remote_code = True

        mock_model_bridge = Mock(ADDITIONAL_FILE_PATTERNS=None)
        mock_model_bridge.get_hf_tokenizer_kwargs.return_value = {}
        with patch.object(
            type(bridge),
            "_model_bridge",
            PropertyMock(return_value=mock_model_bridge),
        ):
            with patch.object(AutoBridge, "save_hf_weights"):
                bridge.save_hf_pretrained([Mock()], output_path)

        saved_config = json.loads((output_path / "config.json").read_text())
        assert saved_config["auto_map"] == {
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
        }
        assert (output_path / "modeling_custom.py").exists()
        assert not (output_path / ".cache").exists()

    @patch("torch.distributed.get_rank", return_value=1)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.barrier")
    def test_save_hf_pretrained_non_zero_rank(
        self, mock_barrier, mock_is_available, mock_is_initialized, mock_get_rank
    ):
        """Test save_hf_pretrained on non-zero rank (should not save artifacts)."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.save_artifacts = Mock()

        mock_megatron_model = [Mock()]

        with patch.object(AutoBridge, "save_hf_weights") as mock_save_hf_weights:
            bridge = AutoBridge(mock_hf_model)

            # Mock _model_bridge to have no ADDITIONAL_FILE_PATTERNS
            with patch.object(
                type(bridge),
                "_model_bridge",
                PropertyMock(return_value=Mock(ADDITIONAL_FILE_PATTERNS=None)),
            ):
                bridge.save_hf_pretrained(mock_megatron_model, "./output_dir")

                # Artifacts should NOT be saved on non-zero rank
                mock_hf_model.save_artifacts.assert_not_called()
                mock_save_hf_weights.assert_called_once_with(
                    mock_megatron_model,
                    "./output_dir",
                    True,
                    True,
                    merge_adapter_weights=True,
                    distributed_save=False,
                    save_every_n_ranks=1,
                    weight_dtype=None,
                )

    def test_export_hf_weights(self):
        """Test exporting weights from Megatron to HF format."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        mock_megatron_model = [object()]

        with patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_model_bridge_prop:
            mock_model_bridge = Mock()
            mock_weight_iter = [("weight1", torch.randn(10, 10)), ("weight2", torch.randn(5, 5))]
            mock_model_bridge.stream_weights_megatron_to_hf.return_value = iter(mock_weight_iter)
            mock_model_bridge_prop.return_value = mock_model_bridge

            with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
                mock_arch_class = Mock()
                mock_transformers.LlamaForCausalLM = mock_arch_class

                bridge = AutoBridge(mock_hf_model)

                # Mock the cached property to avoid accessing transformers
                with patch.object(AutoBridge, "_causal_lm_architecture", new_callable=PropertyMock) as mock_prop:
                    mock_prop.return_value = mock_arch_class
                    weights = list(bridge.export_hf_weights(mock_megatron_model, cpu=True))

                    assert len(weights) == 2
                    assert weights[0][0] == "weight1"
                    assert weights[1][0] == "weight2"
                    assert isinstance(weights[0][1], torch.Tensor)
                    assert isinstance(weights[1][1], torch.Tensor)
                    mock_model_bridge.stream_weights_megatron_to_hf.assert_called_once_with(
                        mock_megatron_model,
                        mock_hf_model,
                        cpu=True,
                        show_progress=True,
                        conversion_tasks=None,
                        merge_adapter_weights=True,
                        weight_dtype=None,
                    )

    def test_export_adapter_weights(self):
        """Test exporting adapter weights from Megatron to HF format."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        mock_megatron_model = [object()]

        with patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_model_bridge_prop:
            mock_model_bridge = Mock()
            mock_weight_iter = [("adapter.weight", torch.randn(4, 4))]
            mock_model_bridge.stream_adapter_weights_megatron_to_hf.return_value = iter(mock_weight_iter)
            mock_model_bridge_prop.return_value = mock_model_bridge

            with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
                mock_arch_class = Mock()
                mock_transformers.LlamaForCausalLM = mock_arch_class

                bridge = AutoBridge(mock_hf_model)

                with patch.object(AutoBridge, "_causal_lm_architecture", new_callable=PropertyMock) as mock_prop:
                    mock_prop.return_value = mock_arch_class
                    weights = list(bridge.export_adapter_weights(mock_megatron_model, cpu=False, show_progress=False))

                    assert len(weights) == 1
                    assert weights[0][0] == "adapter.weight"
                    assert isinstance(weights[0][1], torch.Tensor)
                    mock_model_bridge.stream_adapter_weights_megatron_to_hf.assert_called_once_with(
                        mock_megatron_model,
                        cpu=False,
                        show_progress=False,
                        exclude_adapter_base_prefixes=None,
                    )

    def test_get_causal_lm_architecture(self):
        """Test getting the CausalLM architecture class."""
        # Test with model that has architectures
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            mock_arch_class = Mock()
            mock_transformers.LlamaForCausalLM = mock_arch_class

            # Create bridge instance directly without isinstance validation
            bridge = AutoBridge.__new__(AutoBridge)
            bridge.hf_pretrained = mock_hf_model

            arch = bridge._causal_lm_architecture
            assert arch == mock_arch_class

    def test_get_causal_lm_architecture_with_masked_lm_wrapper(self):
        """Test _causal_lm_architecture reads config off a PreTrainedMaskedLM wrapper too."""
        mock_hf_model = Mock(spec=PreTrainedMaskedLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["BertForMaskedLM"]
        mock_hf_model.config.auto_map = None

        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            mock_arch_class = Mock()
            mock_transformers.BertForMaskedLM = mock_arch_class

            bridge = AutoBridge.__new__(AutoBridge)
            bridge.hf_pretrained = mock_hf_model

            arch = bridge._causal_lm_architecture
            assert arch == mock_arch_class

    def test_get_causal_lm_architecture_no_architectures(self):
        """Test error when no architectures found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = []
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        with pytest.raises(ValueError, match="No architectures found in model config"):
            bridge._causal_lm_architecture

    def test_get_causal_lm_architecture_no_causal_lm(self):
        """Test error when no supported architecture is found."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["BertModel"]
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        with pytest.raises(ValueError, match="No supported architecture found"):
            bridge._causal_lm_architecture

    def test_get_causal_lm_architecture_not_in_transformers(self):
        """Test that custom-registered arch names (not in transformers) are returned as strings.

        Custom models registered via AutoConfig.register / AutoModelForCausalLM.register
        (e.g. BailingMoeV2ForCausalLM) are not present in the standard transformers module
        but are still valid — the bridge dispatch supports string-based source matching.
        """
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["CustomForCausalLM"]
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        # Mock transformers to not have the CustomForCausalLM attribute
        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            del mock_transformers.CustomForCausalLM

            # Falls back to string class name for custom-registered models
            result = bridge._causal_lm_architecture
            assert result == "CustomForCausalLM"

    def test_get_causal_lm_architecture_string_registered_fallback(self):
        """Test that a string-registered architecture resolves via the _exact_types fallback."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["Qwen3ASRForConditionalGeneration"]
        mock_hf_model.config.auto_map = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            # Architecture not available in transformers
            del mock_transformers.Qwen3ASRForConditionalGeneration

            with patch("megatron.bridge.models.conversion.auto_bridge.model_bridge") as mock_model_bridge:
                # Simulate a registry that contains the architecture as a string key
                mock_get_bridge = Mock()
                mock_get_bridge._exact_types = {"Qwen3ASRForConditionalGeneration": Mock()}
                mock_model_bridge.get_model_bridge = mock_get_bridge

                arch = bridge._causal_lm_architecture
                assert arch == "Qwen3ASRForConditionalGeneration"

    def test_repr(self):
        """Test string representation of AutoBridge."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.__repr__ = Mock(return_value="PreTrainedCausalLM(\n  config=...\n)")

        mock_model_bridge = Mock()
        mock_model_bridge.__repr__ = Mock(return_value="ModelBridge(\n  mappings=...\n)")

        with patch.object(AutoBridge, "_model_bridge", mock_model_bridge):
            bridge = AutoBridge.__new__(AutoBridge)
            bridge.hf_pretrained = mock_hf_model
            repr_str = repr(bridge)

            assert "AutoBridge(" in repr_str
            assert "(hf_pretrained):" in repr_str
            assert "(model_bridge):" in repr_str
            assert "PreTrainedCausalLM" in repr_str
            assert "ModelBridge" in repr_str

    @patch("torch.cuda.current_device", return_value=0)
    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_compatibility(self, mock_cuda_avail, mock_cuda_device):
        """Test that bridge works on CPU-only systems."""
        # This test ensures the bridge doesn't require CUDA
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["GPT2ForCausalLM"]
        mock_hf_model.config.auto_map = None

        # Create bridge - should work without CUDA
        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        assert bridge.hf_pretrained == mock_hf_model

        # Test methods that might use device
        with patch("megatron.bridge.models.conversion.auto_bridge.transformers") as mock_transformers:
            mock_transformers.GPT2ForCausalLM = Mock()

            # These operations should work on CPU
            arch = bridge._causal_lm_architecture
            assert arch is not None

    def test_kwargs_passed_through(self, gpt2_config):
        """Test that all kwargs are properly passed to the underlying loader."""
        with patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry"
        ) as mock_safe_load_config:
            with patch(
                "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained"
            ) as mock_from_pretrained:
                with patch.object(AutoBridge, "_validate_config"):
                    mock_safe_load_config.return_value = gpt2_config
                    mock_model = create_mock_pretrained_causal_lm()
                    mock_from_pretrained.return_value = mock_model

                    # Call with various kwargs
                    AutoBridge.from_hf_pretrained(
                        "gpt2",
                        trust_remote_code=True,
                        device_map="balanced",
                        torch_dtype="bfloat16",
                        custom_param="test",
                    )

                    # Verify all kwargs were passed
                    mock_from_pretrained.assert_called_once_with(
                        "gpt2",
                        trust_remote_code=True,
                        device_map="balanced",
                        torch_dtype="bfloat16",
                        custom_param="test",
                    )

    @patch.object(AutoBridge, "save_megatron_model")
    @patch.object(AutoBridge, "to_megatron_model")
    @patch.object(AutoBridge, "from_hf_pretrained")
    def test_import_ckpt_basic(self, mock_from_hf_pretrained, mock_to_megatron_model, mock_save_megatron_model):
        """Test basic import_ckpt functionality."""
        # Setup mocks
        mock_bridge = Mock(spec=AutoBridge)
        mock_from_hf_pretrained.return_value = mock_bridge

        mock_megatron_model = [Mock()]
        mock_bridge.to_megatron_model.return_value = mock_megatron_model
        mock_bridge.save_megatron_model = Mock()

        # Test import_ckpt
        AutoBridge.import_ckpt("meta-llama/Meta-Llama-3-8B", "./megatron_checkpoint")

        # Assertions
        mock_from_hf_pretrained.assert_called_once_with("meta-llama/Meta-Llama-3-8B")
        mock_bridge.to_megatron_model.assert_called_once_with(wrap_with_ddp=False, use_cpu_initialization=True)
        mock_bridge.save_megatron_model.assert_called_once_with(
            mock_megatron_model,
            "./megatron_checkpoint",
            hf_tokenizer_path="meta-llama/Meta-Llama-3-8B",
            hf_tokenizer_kwargs=mock_bridge._model_bridge.get_hf_tokenizer_kwargs(),
            low_memory_save=True,
        )

    @patch.object(AutoBridge, "save_megatron_model")
    @patch.object(AutoBridge, "to_megatron_model")
    @patch.object(AutoBridge, "from_hf_pretrained")
    def test_import_ckpt_with_kwargs(self, mock_from_hf_pretrained, mock_to_megatron_model, mock_save_megatron_model):
        """Test import_ckpt with custom kwargs."""
        # Setup mocks
        mock_bridge = Mock(spec=AutoBridge)
        mock_from_hf_pretrained.return_value = mock_bridge

        mock_megatron_model = [Mock()]
        mock_bridge.to_megatron_model.return_value = mock_megatron_model
        mock_bridge.save_megatron_model = Mock()
        mock_bridge._model_bridge.get_hf_tokenizer_kwargs.return_value = {}

        # Test import_ckpt with kwargs
        AutoBridge.import_ckpt(
            "./local_model",
            "./megatron_checkpoint",
            torch_dtype=torch.float16,
            device_map="auto",
            revision="0123456789abcdef",  # pragma: allowlist secret
        )

        # Assertions
        mock_from_hf_pretrained.assert_called_once_with(
            "./local_model",
            torch_dtype=torch.float16,
            device_map="auto",
            revision="0123456789abcdef",  # pragma: allowlist secret
        )
        mock_bridge.to_megatron_model.assert_called_once_with(wrap_with_ddp=False, use_cpu_initialization=True)
        mock_bridge.save_megatron_model.assert_called_once_with(
            mock_megatron_model,
            "./megatron_checkpoint",
            hf_tokenizer_path="./local_model",
            hf_tokenizer_kwargs={"revision": "0123456789abcdef"},  # pragma: allowlist secret
            low_memory_save=True,
        )

    def test_export_ckpt_basic(self):
        """Test basic export_ckpt functionality."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch.object(bridge, "load_megatron_model") as mock_load_megatron_model:
            with patch.object(bridge, "save_hf_pretrained") as mock_save_hf_pretrained:
                mock_load_megatron_model.return_value = mock_megatron_model

                # Test export_ckpt
                bridge.export_ckpt("./megatron_checkpoint", "./hf_export")

                # Assertions
                mock_load_megatron_model.assert_called_once_with("./megatron_checkpoint", wrap_with_ddp=False)
                mock_save_hf_pretrained.assert_called_once_with(
                    mock_megatron_model,
                    "./hf_export",
                    show_progress=True,
                    source_path=None,
                    strict=False,
                )

    def test_export_ckpt_with_kwargs(self):
        """Test export_ckpt with custom kwargs."""
        # Setup mocks
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch.object(bridge, "load_megatron_model") as mock_load_megatron_model:
            with patch.object(bridge, "save_hf_pretrained") as mock_save_hf_pretrained:
                mock_load_megatron_model.return_value = mock_megatron_model

                # Test export_ckpt with kwargs
                bridge.export_ckpt("./megatron_checkpoint", "./hf_export", show_progress=False)

                # Assertions
                mock_load_megatron_model.assert_called_once_with("./megatron_checkpoint", wrap_with_ddp=False)
                mock_save_hf_pretrained.assert_called_once_with(
                    mock_megatron_model,
                    "./hf_export",
                    show_progress=False,
                    source_path=None,
                    strict=False,
                )

    def test_save_megatron_model_basic(self):
        """Test save_megatron_model method."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.save_megatron_model") as mock_save_megatron_model:
            bridge.save_megatron_model(mock_megatron_model, "./checkpoint_path", low_memory_save=True)

            mock_save_megatron_model.assert_called_once_with(
                mock_megatron_model,
                "./checkpoint_path",
                hf_tokenizer_path=None,
                low_memory_save=True,
                hf_tokenizer_kwargs=None,
            )

    def test_save_megatron_model_with_tokenizer(self):
        """Test save_megatron_model method with tokenizer path."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        mock_megatron_model = [Mock()]

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with patch("megatron.bridge.training.model_load_save.save_megatron_model") as mock_save_megatron_model:
            bridge.save_megatron_model(
                mock_megatron_model,
                "./checkpoint_path",
                hf_tokenizer_path="meta-llama/Meta-Llama-3-8B",
                low_memory_save=True,
            )

            mock_save_megatron_model.assert_called_once_with(
                mock_megatron_model,
                "./checkpoint_path",
                hf_tokenizer_path="meta-llama/Meta-Llama-3-8B",
                low_memory_save=True,
                hf_tokenizer_kwargs=None,
            )

    def test_save_megatron_model_import_error(self):
        """Test save_megatron_model import error handling."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        # Create a mock that raises ImportError when accessed
        def mock_import(*args, **kwargs):
            if "megatron.bridge.training.model_load_save" in args[0]:
                raise ImportError("No module named 'megatron.bridge.training.model_load_save'")
            return __import__(*args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="megatron.bridge.training is not available"):
                bridge.save_megatron_model([Mock()], "./path")

    def test_load_megatron_model_basic(self):
        """Test load_megatron_model method."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        bridge.trust_remote_code = False

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            from pathlib import Path

            with patch.object(Path, "iterdir") as mock_iterdir:
                # Setup mocks
                mock_model = Mock()
                mock_load_megatron_model.return_value = mock_model

                # Mock iterdir to return empty list (no iter_ folders)
                mock_iterdir.return_value = []

                result = bridge.load_megatron_model("./checkpoint_path")

                assert result == [mock_model]
                mock_load_megatron_model.assert_called_once()
                mock_iterdir.assert_called_once()

    def test_load_megatron_model_with_iter_folder(self):
        """Test load_megatron_model with iter_ folders."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        bridge.trust_remote_code = False

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            from pathlib import Path

            # Create mock folder objects
            mock_iter_folder_1 = Mock()
            mock_iter_folder_1.is_dir.return_value = True
            mock_iter_folder_1.name = "iter_0000010"

            mock_iter_folder_2 = Mock()
            mock_iter_folder_2.is_dir.return_value = True
            mock_iter_folder_2.name = "iter_0000020"

            # Mock path.iterdir()
            with patch.object(Path, "iterdir") as mock_iterdir:
                # Setup mocks
                mock_model = Mock()
                mock_load_megatron_model.return_value = mock_model

                # Mock iterdir to return the iter folders
                mock_iterdir.return_value = [mock_iter_folder_1, mock_iter_folder_2]

                result = bridge.load_megatron_model("./checkpoint_path")

                assert result == [mock_model]
                mock_load_megatron_model.assert_called_once()
                mock_iterdir.assert_called_once()
                # Should use the latest iteration (iter_0000020)

    def test_load_megatron_model_with_mp_overrides(self):
        """Test load_megatron_model with model-parallel overrides argument."""

        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        bridge.trust_remote_code = False

        # Create model-parallel overrides
        mp_overrides = {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        }

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            with patch("torch.distributed.is_available", return_value=False):
                with patch("torch.distributed.is_initialized", return_value=False):
                    from pathlib import Path

                    with patch.object(Path, "iterdir") as mock_iterdir:
                        # Setup mocks
                        mock_model = Mock()
                        mock_load_megatron_model.return_value = mock_model

                        # Mock iterdir to return empty list (no iter_ folders)
                        mock_iterdir.return_value = []

                        # Call load_megatron_model with model-parallel overrides
                        result = bridge.load_megatron_model(
                            "checkpoint_path",
                            mp_overrides=mp_overrides,
                            wrap_with_ddp=False,
                        )

                        # Verify the result
                        assert result == [mock_model]

                        # Verify that load_megatron_model was called with mp_overrides
                        mock_load_megatron_model.assert_called_once()
                        call_args = mock_load_megatron_model.call_args

                        # Check that mp_overrides was passed correctly
                        assert call_args.kwargs["mp_overrides"] == mp_overrides

                        # Check other expected arguments
                        assert call_args.args[0] == "checkpoint_path"  # path argument
                        assert "skip_temp_dist_context" in call_args.kwargs

    def test_load_megatron_model_registers_prefix_when_trust_remote_code(self):
        """Test that load_megatron_model registers transformers_modules prefix when trust_remote_code=True."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_config = Mock(spec=PretrainedConfig)
        mock_config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config = mock_config

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model
        bridge.trust_remote_code = True

        with patch("megatron.bridge.training.model_load_save.load_megatron_model") as mock_load_megatron_model:
            with patch("megatron.bridge.utils.instantiate_utils.register_allowed_target_prefix") as mock_register:
                from pathlib import Path

                with patch.object(Path, "iterdir") as mock_iterdir:
                    mock_load_megatron_model.return_value = Mock()
                    mock_iterdir.return_value = []

                    bridge.load_megatron_model("./checkpoint_path")

                    mock_register.assert_called_once_with("transformers_modules.")

    @patch("torch.distributed.is_available")
    @patch("torch.distributed.is_initialized")
    def test_save_hf_pretrained_uses_bridge_additional_file_patterns(self, mock_is_init, mock_is_avail):
        """Test that save_hf_pretrained uses bridge-level ADDITIONAL_FILE_PATTERNS."""
        # Setup distributed mocks
        mock_is_avail.return_value = False
        mock_is_init.return_value = False

        # Create a mock PreTrainedCausalLM
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.save_artifacts = Mock()

        # Create AutoBridge
        bridge = AutoBridge(mock_pretrained)

        # Mock the _model_bridge to have ADDITIONAL_FILE_PATTERNS
        mock_model_bridge = Mock()
        mock_model_bridge.ADDITIONAL_FILE_PATTERNS = [
            "*reasoning_parser.py",
            "custom_file.txt",
        ]

        # Patch _model_bridge as a property
        with patch.object(type(bridge), "_model_bridge", PropertyMock(return_value=mock_model_bridge)):
            # Call save_hf_pretrained
            mock_model = Mock()

            with patch.object(bridge, "save_hf_weights"):
                bridge.save_hf_pretrained(mock_model, "/tmp/output")

            # Verify save_artifacts was called with the bridge-level patterns
            mock_pretrained.save_artifacts.assert_called_once()
            call_kwargs = mock_pretrained.save_artifacts.call_args.kwargs

            assert call_kwargs["additional_files"] == [
                "*reasoning_parser.py",
                "custom_file.txt",
            ]

    @patch("torch.distributed.is_available")
    @patch("torch.distributed.is_initialized")
    def test_save_hf_pretrained_without_additional_file_patterns(self, mock_is_init, mock_is_avail):
        """Test that save_hf_pretrained works when bridge has no ADDITIONAL_FILE_PATTERNS."""
        # Setup distributed mocks
        mock_is_avail.return_value = False
        mock_is_init.return_value = False

        # Create a mock PreTrainedCausalLM
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.save_artifacts = Mock()

        # Create AutoBridge
        bridge = AutoBridge(mock_pretrained)

        # Mock the _model_bridge without ADDITIONAL_FILE_PATTERNS
        mock_model_bridge = Mock()
        mock_model_bridge.ADDITIONAL_FILE_PATTERNS = None

        # Patch _model_bridge as a property
        with patch.object(type(bridge), "_model_bridge", PropertyMock(return_value=mock_model_bridge)):
            # Call save_hf_pretrained
            mock_model = Mock()

            with patch.object(bridge, "save_hf_weights"):
                bridge.save_hf_pretrained(mock_model, "/tmp/output")

            # Verify save_artifacts was called with None for additional_files
            mock_pretrained.save_artifacts.assert_called_once()
            call_kwargs = mock_pretrained.save_artifacts.call_args.kwargs

            assert call_kwargs["additional_files"] is None

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_save_hf_weights_filters_quantizer_tensors(self, mock_get_rank, mock_is_init, mock_is_avail, mock_barrier):
        """Test that save_hf_weights separates _quantizer. tensors into a sidecar file."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

        mock_source = Mock(spec=SafeTensorsStateSource)
        mock_hf_model.state = Mock()
        mock_hf_model.state.source = mock_source

        normal_tensor = torch.randn(4, 4)
        quant_tensor = torch.randn(1)
        weight_iter = [
            ("model.layers.0.self_attn.q_proj.weight", normal_tensor),
            ("model.layers.0.self_attn.q_proj.input_quantizer._amax", quant_tensor),
        ]

        mock_megatron_model = [Mock()]
        mock_megatron_model[0].module = Mock()
        mock_megatron_model[0].module.module = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with (
            patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_model_bridge_prop,
            patch(
                "modelopt.torch.quantization.utils.is_quantized",
                return_value=True,
            ),
            patch("torch.save") as mock_torch_save,
        ):
            mock_model_bridge = Mock()
            mock_model_bridge.stream_weights_megatron_to_hf.return_value = iter(weight_iter)
            mock_model_bridge_prop.return_value = mock_model_bridge

            # Capture what save_generator receives by consuming the generator it's passed
            saved_pairs = []

            def fake_save_generator(gen, *args, **kwargs):
                for pair in gen:
                    saved_pairs.append(pair)

            mock_source.save_generator = fake_save_generator

            bridge.save_hf_weights(mock_megatron_model, "/tmp/output")

            # Only the normal weight should have passed through to save_generator
            assert len(saved_pairs) == 1
            assert saved_pairs[0][0] == "model.layers.0.self_attn.q_proj.weight"
            mock_model_bridge.stream_weights_megatron_to_hf.assert_called_once_with(
                mock_megatron_model,
                mock_hf_model,
                cpu=True,
                show_progress=True,
                merge_adapter_weights=True,
                weight_dtype=None,
            )

            # The quantizer tensor should have been saved via torch.save sidecar
            mock_torch_save.assert_called_once()
            sidecar_dict = mock_torch_save.call_args[0][0]
            assert "model.layers.0.self_attn.q_proj.input_quantizer._amax" in sidecar_dict

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_save_hf_weights_no_sidecar_when_not_quantized(
        self, mock_get_rank, mock_is_init, mock_is_avail, mock_barrier
    ):
        """Test that save_hf_weights skips sidecar logic when model is not quantized."""
        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = Mock()
        mock_hf_model.config.architectures = ["LlamaForCausalLM"]
        mock_hf_model.config.auto_map = None

        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

        mock_source = Mock(spec=SafeTensorsStateSource)
        mock_hf_model.state = Mock()
        mock_hf_model.state.source = mock_source

        weight_iter = [("model.weight", torch.randn(4, 4))]

        mock_megatron_model = [Mock()]
        mock_megatron_model[0].module = Mock()
        mock_megatron_model[0].module.module = None

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        with (
            patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_model_bridge_prop,
            patch(
                "modelopt.torch.quantization.utils.is_quantized",
                return_value=False,
            ),
            patch("torch.save") as mock_torch_save,
        ):
            mock_model_bridge = Mock()
            mock_model_bridge.stream_weights_megatron_to_hf.return_value = iter(weight_iter)
            mock_model_bridge_prop.return_value = mock_model_bridge

            mock_source.save_generator = Mock()
            bridge.save_hf_weights(mock_megatron_model, "/tmp/output")

            mock_model_bridge.stream_weights_megatron_to_hf.assert_called_once_with(
                mock_megatron_model,
                mock_hf_model,
                cpu=True,
                show_progress=True,
                merge_adapter_weights=True,
                weight_dtype=None,
            )
            mock_torch_save.assert_not_called()

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_available", return_value=True)
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_save_hf_weights_ignores_mtp_source_keys_when_mtp_disabled(
        self, mock_get_rank, mock_is_init, mock_is_avail, mock_barrier, tmp_path
    ):
        """Pass MTP source-key ignore prefixes when the exported config disables MTP."""
        from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

        class ModelWrapper:
            pass

        class ModelInstance:
            pass

        model_instance = ModelInstance()
        model_instance.config = {"num_nextn_predict_layers": 0}
        wrapper = ModelWrapper()
        wrapper.module = model_instance

        mock_hf_model = Mock(spec=PreTrainedCausalLM)
        mock_hf_model.config = {"num_nextn_predict_layers": 1}
        mock_hf_model.state = Mock()
        mock_source = Mock(spec=SafeTensorsStateSource)
        mock_source.has_glob.return_value = True
        mock_hf_model.state.source = mock_source

        bridge = AutoBridge.__new__(AutoBridge)
        bridge.hf_pretrained = mock_hf_model

        mock_model_bridge = Mock()
        mock_model_bridge.stream_weights_megatron_to_hf.return_value = iter([("model.weight", torch.ones(1))])

        with (
            patch.object(AutoBridge, "_model_bridge", new_callable=PropertyMock) as mock_model_bridge_prop,
            patch("modelopt.torch.quantization.utils.is_quantized", return_value=False),
        ):
            mock_model_bridge_prop.return_value = mock_model_bridge
            bridge.save_hf_weights([wrapper], tmp_path)

        assert mock_source.save_generator.call_args.kwargs["ignored_source_key_prefixes"] == ("mtp.",)
        mock_source.has_glob.assert_called_once_with("mtp.*")
