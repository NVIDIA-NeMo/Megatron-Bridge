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

import os
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from megatron.bridge.data.builders import DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.data.utils import get_dataset_provider


@pytest.mark.run_only_on("GPU")
class TestVLMDirectHFMasking:
    def test_direct_hf_vlm_label_masking_and_alignment(self):
        try:
            from transformers import AutoProcessor  # noqa: F401
        except Exception:
            pytest.skip("transformers not available")

        hf_processor = os.environ.get("HF_VLM_PROCESSOR", "Qwen/Qwen2.5-VL-3B-Instruct")

        def make_short_conversations(**kwargs):
            del kwargs
            return [
                {
                    "conversation": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": Image.new("RGB", (16, 16), color="red")},
                                {"type": "text", "text": "Answer with the word red."},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "red"}]},
                    ]
                },
                {
                    "conversation": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": Image.new("RGB", (16, 16), color="blue")},
                                {"type": "text", "text": "Answer with the word blue."},
                            ],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": "blue"}]},
                    ]
                },
            ]

        config = DirectHFSFTDatasetConfig(
            # Qwen's default minimum image resolution expands this tiny image
            # to roughly 256 visual tokens. Leave room for the assistant turn
            # so this test validates masking rather than truncating all labels.
            seq_length=512,
            hf_processor_path=hf_processor,
            source=HFDatasetSourceConfig(path_or_dataset="synthetic/vlm", schema_adapter="rdr"),
            num_workers=0,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=False,
            persistent_workers=False,
        )

        provider = get_dataset_provider(config)
        with (
            patch(
                "megatron.bridge.data.builders.direct_hf_sft.prepare_hf_dataset_sources",
                return_value=None,
            ),
            patch(
                "megatron.bridge.data.builders.direct_hf_sft.load_and_adapt_hf_dataset",
                side_effect=lambda _: make_short_conversations(),
            ),
        ):
            train_ds, _, _ = provider([2, 0, 0], config, tokenizer=None)
        assert train_ds is not None

        def _collate_with_capture(batch_examples):
            setattr(train_ds, "_last_batch_examples", batch_examples)
            return train_ds.collate_fn(batch_examples)

        loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=_collate_with_capture)

        try:
            batch = next(iter(loader))
        except ImportError as e:
            pytest.skip(f"qwen-vl-utils likely missing: {e}")

        assert "input_ids" in batch
        assert "labels" in batch
        assert "loss_mask" in batch
        assert batch["visual_inputs"] is not None
        assert batch["visual_inputs"].pixel_values is not None
        assert batch["visual_inputs"].pixel_values.numel() > 0

        labels = batch["labels"]
        loss_mask = batch["loss_mask"].to(dtype=torch.bool)

        # Where loss_mask == 0, labels must be -100
        assert torch.all(labels[~loss_mask] == -100)

        # Where loss_mask == 1, labels should not be -100
        has_unmasked = torch.any(loss_mask, dim=1)
        assert torch.all(has_unmasked)
        assert torch.all(labels[loss_mask] != -100)

        # Token-level 1:1 match of assistant replies with unmasked labels
        processor = getattr(train_ds, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", processor)

        def gather_assistant_texts(example: dict):
            out = []
            for turn in example.get("conversation", []):
                if turn.get("role") != "assistant":
                    continue
                parts = turn.get("content", [])
                if isinstance(parts, list):
                    buf = []
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                            buf.append(p["text"])
                    if buf:
                        out.append("".join(buf))
                elif isinstance(parts, str):
                    out.append(parts)
            return out

        unmasked_rows = torch.nonzero(has_unmasked, as_tuple=False).flatten().tolist()
        assert unmasked_rows
        examples_batch = getattr(train_ds, "_last_batch_examples")
        for i in unmasked_rows:
            label_ids = [int(t) for t in labels[i].tolist() if int(t) != -100]
            pos = 0
            turns = gather_assistant_texts(examples_batch[i])
            ok = True
            for t in turns:
                tok0 = tokenizer(t, add_special_tokens=False)["input_ids"]
                tok1 = tokenizer(t + "\n", add_special_tokens=False)["input_ids"]
                matched = False
                for cand in (tok0, tok1):
                    L = len(cand)
                    if label_ids[pos : pos + L] == cand:
                        pos += L
                        matched = True
                        break
                if not matched:
                    ok = False
                    break
            if ok and pos != len(label_ids):
                terminal_candidates = (
                    tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"],
                    tokenizer("<|im_end|>\n", add_special_tokens=False)["input_ids"],
                    tokenizer("\n<|im_end|>", add_special_tokens=False)["input_ids"],
                )
                ok = any(label_ids[pos:] == candidate for candidate in terminal_candidates)
            assert ok
