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

"""Megatron-Core tokenizer for Inkling2 text training.

The ``thinkingmachines/Inkling-2-Small`` tokenizer configuration assigns no BOS or EOS token, which pretraining
datasets and SFT padding need, and its chat template marks no assistant span for the SFT loss.
``Inkling2MegatronTokenizer`` loads the release's Hugging Face tokenizer with the reference tokenizer's roles and
declares the chat template's assistant-turn boundaries. Recipes select it through Megatron-Core's tokenizer metadata:
``cfg.tokenizer.metadata_path = inkling2_tokenizer_metadata()``.
"""

from typing import Any

from megatron.core.tokenizers.text.libraries.abstract_tokenizer import MegatronTokenizerTextAbstract
from megatron.core.tokenizers.text.text_tokenizer import MegatronTokenizerText

from megatron.bridge.data.conversation_processing import (
    CHATML_ASSISTANT_ROLE,
    AssistantMaskBoundaryConfig,
    assistant_mask_boundary_config_from_markers,
)


# The reference tokenizer's roles. The release's vocabulary holds both tokens, but its configuration assigns no role.
INKLING2_BOS_TOKEN = "<|begin_of_text|>"
INKLING2_EOS_TOKEN = "<|endoftext|>"


class Inkling2MegatronTokenizer(MegatronTokenizerText):
    """The Inkling2 Hugging Face tokenizer with BOS and EOS roles and the chat template's assistant-turn boundaries."""

    def _restore_model(self, **kwargs: Any) -> MegatronTokenizerTextAbstract:
        """Load the Hugging Face tokenizer with the BOS and EOS roles; their tokens must already be in its vocabulary."""
        if self.library != "huggingface":
            raise ValueError(f"Inkling2MegatronTokenizer loads a Hugging Face tokenizer, not a {self.library!r} one.")
        tokenizer = super()._restore_model(
            **{**kwargs, "bos_token": INKLING2_BOS_TOKEN, "eos_token": INKLING2_EOS_TOKEN}
        )
        if tokenizer.vocab_size != tokenizer.original_vocab_size:
            raise ValueError(
                f"{INKLING2_BOS_TOKEN} and {INKLING2_EOS_TOKEN} must already be in the tokenizer's vocabulary; "
                f"assigning their roles added {tokenizer.vocab_size - tokenizer.original_vocab_size} tokens."
            )
        return tokenizer

    def assistant_mask_boundary_config(self) -> AssistantMaskBoundaryConfig:
        """Return the chat template's assistant-turn boundaries for the SFT loss mask.

        The template renders every message as role-tagged blocks such as
        ``<|message_model|><|content_text|>...<|end_message|>`` and ends an assistant turn with
        ``<|content_model_end_sampling|>``. The model writes its own ``<|message_model|>`` headers, so the loss
        covers the whole turn, from its first header through the end-of-sampling token.
        """
        return assistant_mask_boundary_config_from_markers(
            self,
            assistant_start="<|message_model|>",
            assistant_end="<|content_model_end_sampling|>",
            role_start_markers={
                "system": "<|message_system|>",
                "user": "<|message_user|>",
                "tool": "<|message_tool|>",
            },
            include_start_tokens_for_roles=(CHATML_ASSISTANT_ROLE,),
        )


def inkling2_tokenizer_metadata() -> dict[str, str]:
    """Return the Megatron-Core tokenizer metadata that loads a Hugging Face tokenizer as ``Inkling2MegatronTokenizer``.

    ``TokenizerConfig.metadata_path`` takes this metadata in place of a path to a metadata file.
    """
    return {"library": "huggingface", "class_path": __name__, "class_name": Inkling2MegatronTokenizer.__name__}


__all__ = [
    "INKLING2_BOS_TOKEN",
    "INKLING2_EOS_TOKEN",
    "Inkling2MegatronTokenizer",
    "inkling2_tokenizer_metadata",
]
