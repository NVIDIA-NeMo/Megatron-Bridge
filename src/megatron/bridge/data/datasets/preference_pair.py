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

import logging
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

from torch.utils.data import Dataset

from megatron.bridge.data.datasets.preference import REF_LOGPROB_COLUMNS, preference_collate_fn
from megatron.bridge.data.datasets.preference_tokenization import build_pair_conversations, tokenize_conversation


logger = logging.getLogger(__name__)


class PreferencePairDataset(Dataset):
    """Preference-pair dataset that tokenizes rows on fetch.

    Fields may be message lists or plain strings; nothing is tokenized at init,
    and each ``__getitem__`` runs the chat template on that pair. ``__len__`` must stay fixed for the sampler, so invalid pairs are
    not dropped - they are emitted as two-token stubs with ``loss_multiplier=0.0``
    and contribute nothing to the loss. ``pair_id`` is the source row index, which
    is what ``ref_logprobs`` (the scorer's output) is keyed by.
    """

    def __init__(
        self,
        source: Sequence[Mapping[str, Any]],
        tokenizer,
        max_seq_length: int,
        pad_token_id: int | None = None,
        pad_seq_length_to_mult: int = 1,
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        prompt_key: str | None = None,
        ref_logprobs: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> None:
        if len(source) == 0:
            raise ValueError("Preference dataset is empty.")
        self.source = source
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.prompt_key = prompt_key
        self.ref_logprobs = ref_logprobs
        self.require_ref_logprobs = ref_logprobs is not None

        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self.pad_token_id = pad_token_id

        self.collate_fn = partial(
            preference_collate_fn,
            pad_token_id=pad_token_id,
            pad_seq_length_to_mult=pad_seq_length_to_mult,
            require_ref_logprobs=self.require_ref_logprobs,
        )

    def __len__(self) -> int:
        return len(self.source)

    def _stub_record(self, idx: int, reason: str) -> dict[str, Any]:
        logger.warning("Pair %d unusable (%s); emitting a zero-loss stub.", idx, reason)
        stub_ids = [self.pad_token_id, self.pad_token_id]

        return {
            "pair_id": idx,
            "chosen_input_ids": stub_ids,
            "chosen_context_len": 1,
            "rejected_input_ids": stub_ids,
            "rejected_context_len": 1,
            "loss_multiplier": 0.0,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.source[idx]
        pair = build_pair_conversations(
            row, chosen_key=self.chosen_key, rejected_key=self.rejected_key, prompt_key=self.prompt_key
        )
        if isinstance(pair, str):
            record = self._stub_record(idx, pair)
        else:
            side_c = tokenize_conversation(self.tokenizer, pair[0], self.max_seq_length)
            side_r = tokenize_conversation(self.tokenizer, pair[1], self.max_seq_length)
            if isinstance(side_c, str) or isinstance(side_r, str):
                record = self._stub_record(idx, side_c if isinstance(side_c, str) else side_r)
            else:
                record = {
                    "pair_id": idx,
                    "chosen_input_ids": side_c[0],
                    "chosen_context_len": side_c[1],
                    "rejected_input_ids": side_r[0],
                    "rejected_context_len": side_r[1],
                    "loss_multiplier": 1.0,
                }
        if self.ref_logprobs is not None:
            try:
                ref = self.ref_logprobs[idx]
            except KeyError:
                raise ValueError(f"ref_logprobs has no entry for pair_id={idx}.") from None
            record.update({col: ref[col] for col in REF_LOGPROB_COLUMNS})
        return record
