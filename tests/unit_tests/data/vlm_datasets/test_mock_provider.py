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

import types

import torch

import megatron.bridge.data.vlm_datasets.mock_provider as mock
from megatron.bridge.training.config import DatasetBuildContext


class _Proc:
    class _Tok:
        pad_token_id = 0
        eos_token_id = 2
        added_tokens_decoder = {}

        def __init__(self):
            self.vocab = {"a": 1, "b": 2, "c": 3}

        @property
        def vocab_size(self):
            return len(self.vocab)

        def get_vocab(self):
            return self.vocab

    def __init__(self):
        self.tokenizer = self._Tok()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):  # noqa: ARG002
        # Return a simple deterministic string
        return "<bos> user: x"

    def __call__(self, **kwargs):  # text/images/padding/return_tensors
        # Minimal tensors required by dataset
        out = {"input_ids": torch.tensor([[1, 2, 3]])}
        images = kwargs.get("images")
        if images is not None:
            n = len(images)
            out["pixel_values"] = torch.randn(1, n, 3, 4, 4)
            out["image_grid_thw"] = torch.tensor([[[1, 2, 2]] * n])
        return types.SimpleNamespace(**out)


def test_mock_provider_builds_splits(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _Proc()))
    provider = mock.MockVLMConversationProvider(sequence_length=16, hf_processor_path="dummy/model", num_images=0)
    ctx = DatasetBuildContext(train_samples=2, valid_samples=1, test_samples=0)
    train_ds, valid_ds, test_ds = provider.build_datasets(ctx)
    assert train_ds is not None and valid_ds is not None and test_ds is None
