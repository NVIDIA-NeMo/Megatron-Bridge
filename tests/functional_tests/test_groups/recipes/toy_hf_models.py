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

"""Offline Hugging Face artifacts shared by recipe functional tests."""

from pathlib import Path


GLM_45V_TOY_CHAT_TEMPLATE = """\
{% for message in messages %}<|{{ message['role'] }}|>
\
{% if message['content'] is string %}\
{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}\
{% else %}{{ message['content'] }}{% endif %}\
{% else %}\
{% for content in message['content'] %}\
{% if content['type'] == 'image' %}<|image|>\
{% elif content['type'] == 'video' %}<|video|>\
{% elif content['type'] == 'text' %}\
{% if message['role'] == 'assistant' %}{% generation %}{{ content['text'] }}{% endgeneration %}\
{% else %}{{ content['text'] }}{% endif %}\
{% endif %}\
{% endfor %}\
{% endif %}{% if not (loop.last and message['role'] == 'assistant') %}<|endoftext|>{% endif %}\
{% endfor %}\
{% if add_generation_prompt %}<|assistant|>
{% endif %}"""


def _save_minimal_tokenizer(model_dir: Path, *, image_tokens: bool = False, chat_template: str | None = None) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    eos_token = "<eos>"
    if image_tokens:
        vocab.update(
            {
                "<|image|>": 4,
                "<|video|>": 5,
                "<|system|>": 6,
                "<|user|>": 7,
                "<|assistant|>": 8,
                "<|observation|>": 9,
                "<|endoftext|>": 10,
            }
        )
        eos_token = "<|endoftext|>"

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token=eos_token,
        pad_token="<pad>",
        unk_token="<unk>",
    )
    if image_tokens:
        hf_tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<|image|>",
                    "<|video|>",
                    "<|system|>",
                    "<|user|>",
                    "<|assistant|>",
                    "<|observation|>",
                ]
            }
        )
    hf_tokenizer.chat_template = chat_template
    hf_tokenizer.save_pretrained(model_dir)


def create_deepseek_v4_toy_artifacts(root: Path) -> str:
    """Create the config and tokenizer needed to construct a DSv4 recipe offline."""
    from transformers import DeepseekV4Config

    from tests.functional_tests.test_groups.models.deepseek.test_deepseek_v4_conversion import (
        HF_DEEPSEEK_V4_TOY_MODEL_CONFIG,
    )

    model_dir = root / "deepseek_v4_toy"
    model_dir.mkdir(parents=True, exist_ok=True)
    DeepseekV4Config(**HF_DEEPSEEK_V4_TOY_MODEL_CONFIG).save_pretrained(model_dir)
    _save_minimal_tokenizer(model_dir)
    return str(model_dir)


def create_glm_45v_toy_artifacts(root: Path) -> str:
    """Create the config, tokenizer, and image processor needed by GLM recipe tests."""
    from transformers.models.glm4v.configuration_glm4v import Glm4vConfig
    from transformers.models.glm4v.image_processing_glm4v import Glm4vImageProcessor

    from tests.functional_tests.test_groups.models.glm_vl.test_glm_45v_conversion import (
        HF_GLM_45V_TOY_MODEL_CONFIG,
    )

    model_dir = root / "glm_45v_toy"
    model_dir.mkdir(parents=True, exist_ok=True)
    Glm4vConfig(**HF_GLM_45V_TOY_MODEL_CONFIG).save_pretrained(model_dir)
    _save_minimal_tokenizer(model_dir, image_tokens=True, chat_template=GLM_45V_TOY_CHAT_TEMPLATE)
    Glm4vImageProcessor().save_pretrained(model_dir)
    return str(model_dir)
