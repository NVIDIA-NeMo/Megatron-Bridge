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

from unittest.mock import patch, sentinel

import pytest
from transformers import AutoTokenizer

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


class TestTokenizers:
    @pytest.mark.parametrize("vocab_size", [32000])
    def test_build_null_tokenizer(self, vocab_size):
        # Setup
        config = TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=vocab_size,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "null-text"
        assert tokenizer.vocab_size == vocab_size

    @patch("megatron.core.tokenizers.text.libraries.MegatronHFTokenizer")
    @pytest.mark.parametrize("use_fast", [True])
    @pytest.mark.parametrize("include_special_tokens", [False])
    def test_build_megatron_tokenizer(self, mock_hf_tokenizer_class, use_fast, include_special_tokens):
        # Setup
        custom_kwargs = {
            "use_fast": use_fast,
            "include_special_tokens": include_special_tokens,
        }
        config = TokenizerConfig(
            tokenizer_type="GPT2BPETokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "megatron"
        assert tokenizer.path == "GPT2BPETokenizer"
        assert tokenizer.additional_args["use_fast"] == use_fast
        assert tokenizer.additional_args["include_special_tokens"] == include_special_tokens

    @patch("megatron.core.tokenizers.text.libraries.HuggingFaceTokenizer")
    @pytest.mark.parametrize("chat_template", ["{% for message in messages %}{{ message.content }}{% endfor %}"])
    def test_build_hf_tokenizer(self, mock_hf_tokenizer_class, chat_template):
        # Setup
        metadata_path = {"library": "huggingface", "chat_template": chat_template}
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Meta-Llama-3-8B-Instruct",
            metadata_path=metadata_path,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "huggingface"
        assert tokenizer.chat_template == chat_template

    @pytest.mark.parametrize(
        ("model_id", "revision", "trust_remote_code"),
        [
            ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", False),  # pragma: allowlist secret
            (
                "moonshotai/Moonlight-16B-A3B",
                "476b36a473d4467f94469414bef6cee75c9c8172",  # pragma: allowlist secret
                True,
            ),
        ],
    )
    @patch("megatron.bridge.training.tokenizers.tokenizer.build_mcore_tokenizer")
    @patch("huggingface_hub.snapshot_download")
    def test_build_hf_tokenizer_resolves_immutable_revision(
        self,
        mock_snapshot_download,
        mock_build_mcore_tokenizer,
        model_id,
        revision,
        trust_remote_code,
    ):
        mock_snapshot_download.return_value = f"/cache/models--resolved/snapshots/{revision}"
        mock_build_mcore_tokenizer.return_value = sentinel.tokenizer
        hf_tokenizer_kwargs = {"revision": revision, "trust_remote_code": trust_remote_code}
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=model_id,
            hf_tokenizer_kwargs=hf_tokenizer_kwargs,
        )

        tokenizer = build_tokenizer(config)

        assert tokenizer is sentinel.tokenizer
        mock_snapshot_download.assert_called_once()
        snapshot_kwargs = mock_snapshot_download.call_args.kwargs
        assert snapshot_kwargs["repo_id"] == model_id
        assert snapshot_kwargs["revision"] == revision
        assert {"config.json", "tokenizer*", "*.py", "*.jinja"}.issubset(snapshot_kwargs["allow_patterns"])
        assert {"*.safetensors", "*.bin", "*.pt", "*.gguf"}.issubset(snapshot_kwargs["ignore_patterns"])

        resolved_config = mock_build_mcore_tokenizer.call_args.args[0]
        assert resolved_config is not config
        assert resolved_config.tokenizer_model == mock_snapshot_download.return_value
        assert resolved_config.trust_remote_code is trust_remote_code
        assert config.tokenizer_model == model_id
        assert config.hf_tokenizer_kwargs == hf_tokenizer_kwargs

    @patch("megatron.bridge.training.tokenizers.tokenizer.build_mcore_tokenizer")
    @patch("huggingface_hub.snapshot_download")
    def test_build_tokenizer_skips_snapshot_resolution_without_remote_hf_revision(
        self, mock_snapshot_download, mock_build_mcore_tokenizer, tmp_path
    ):
        mock_build_mcore_tokenizer.return_value = sentinel.tokenizer
        configs = [
            TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model="Qwen/Qwen3-8B"),
            TokenizerConfig(
                tokenizer_type="NullTokenizer",
                vocab_size=32,
                hf_tokenizer_kwargs={"revision": "not-used"},
            ),
            TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=tmp_path,
                hf_tokenizer_kwargs={"revision": "local-path-is-already-resolved"},
            ),
        ]

        for config in configs:
            assert build_tokenizer(config) is sentinel.tokenizer
            assert mock_build_mcore_tokenizer.call_args.args[0] is config

        mock_snapshot_download.assert_not_called()

    @patch("megatron.core.tokenizers.text.libraries.TikTokenTokenizer")
    @pytest.mark.parametrize("pattern", ["v1"])
    @pytest.mark.parametrize("num_special_tokens", [2000])
    def test_build_tiktoken_tokenizer(self, mock_tiktoken_tokenizer, pattern, num_special_tokens):
        # Setup
        config = TokenizerConfig(
            tokenizer_type="TikTokenizer",
            tokenizer_model="tiktoken.json",
            tiktoken_pattern=pattern,
            tiktoken_num_special_tokens=num_special_tokens,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        assert tokenizer.library == "tiktoken"
        assert tokenizer.path == "tiktoken.json"
        assert tokenizer.additional_args["pattern"] == pattern
        assert tokenizer.additional_args["num_special_tokens"] == num_special_tokens

    @pytest.mark.timeout(30)
    def test_hf_tokenizer_as_local_path_object(self, tmp_path):
        # Cover the user case where a user has made a local path object of a WIP tokenizer and wants
        #  to use that in some megatron model at train time.

        # First as a proxy download a tokenizer from HF and save it to a local path. A user would
        #  do this differently by exporting their WIP tokenizer to a local path.

        # 1. Download a common, small tokenizer from the Hub
        # "bert-base-uncased" is a safe choice as it's small and standard.
        model_id = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 2. Define a local path in the temporary directory
        local_model_path = tmp_path / "my_local_tokenizer"

        # 3. Save the tokenizer to disk
        # This creates tokenizer_config.json, vocab.txt, special_tokens_map.json, etc.
        tokenizer.save_pretrained(str(local_model_path))

        # 4. Load it back using the local path
        # This simulates the user providing a path to a folder instead of a Hub ID
        cfg = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=local_model_path,
            hf_tokenizer_kwargs={
                "trust_remote_code": True,
                "include_special_tokens": True,
            },
        )
        loaded_tokenizer = build_tokenizer(cfg)

        # 5. Verify it functions identically
        test_text = "Unit testing is important."

        original_tokens = tokenizer.encode(test_text)
        reloaded_tokens = loaded_tokenizer.tokenize(test_text)

        assert original_tokens == reloaded_tokens
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size

        # verify that the directory actually contains files (sanity check)
        assert (local_model_path / "tokenizer_config.json").exists()
        assert (local_model_path / "tokenizer.json").exists()


CHAT_TEMPLATE = "{% generation %}{{ messages }}{% endgeneration %}"


class TestChatTemplatePathOverride:
    """`TokenizerConfig.chat_template_path` loads a jinja template from a local or msc:// path."""

    def test_resolve_from_local_file(self, tmp_path):
        from megatron.bridge.training.tokenizers import tokenizer as tok_mod

        template_file = tmp_path / "template.jinja"
        template_file.write_text(CHAT_TEMPLATE)
        config = TokenizerConfig(chat_template_path=str(template_file))

        with patch.object(tok_mod.MultiStorageClientFeature, "is_enabled", return_value=False):
            assert tok_mod._resolve_chat_template(config) == CHAT_TEMPLATE

    def test_inline_chat_template_passthrough(self):
        from megatron.bridge.training.tokenizers.tokenizer import _resolve_chat_template

        config = TokenizerConfig(chat_template=CHAT_TEMPLATE)

        assert _resolve_chat_template(config) == CHAT_TEMPLATE

    def test_none_when_neither_set(self):
        from megatron.bridge.training.tokenizers.tokenizer import _resolve_chat_template

        assert _resolve_chat_template(TokenizerConfig()) is None

    def test_inline_and_path_are_mutually_exclusive(self, tmp_path):
        from megatron.bridge.training.tokenizers.tokenizer import _resolve_chat_template

        template_file = tmp_path / "template.jinja"
        template_file.write_text(CHAT_TEMPLATE)
        config = TokenizerConfig(chat_template=CHAT_TEMPLATE, chat_template_path=str(template_file))

        with pytest.raises(ValueError):
            _resolve_chat_template(config)

    def test_resolve_via_msc_when_enabled(self):
        from unittest.mock import MagicMock

        from megatron.bridge.training.tokenizers import tokenizer as tok_mod

        fake_handle = MagicMock()
        fake_handle.read.return_value = CHAT_TEMPLATE
        fake_msc = MagicMock()
        fake_msc.open.return_value.__enter__.return_value = fake_handle

        config = TokenizerConfig(chat_template_path="msc://bucket/template.jinja")
        with patch.object(tok_mod, "MultiStorageClientFeature") as msc_feat:
            msc_feat.is_enabled.return_value = True
            msc_feat.import_package.return_value = fake_msc
            assert tok_mod._resolve_chat_template(config) == CHAT_TEMPLATE
        fake_msc.open.assert_called_once_with("msc://bucket/template.jinja", "r")

    def test_build_hf_tokenizer_passes_resolved_template(self, tmp_path):
        from megatron.bridge.training.tokenizers import tokenizer as tok_mod

        template_file = tmp_path / "template.jinja"
        template_file.write_text(CHAT_TEMPLATE)
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            chat_template_path=str(template_file),
        )

        with (
            patch.object(tok_mod.MultiStorageClientFeature, "is_enabled", return_value=False),
            patch.object(tok_mod, "build_mcore_tokenizer") as mock_build_mcore,
        ):
            build_tokenizer(config)

        (called_config,), _ = mock_build_mcore.call_args
        assert called_config.chat_template == CHAT_TEMPLATE
        assert called_config.chat_template_path is None
