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

import pytest
import torch
from torch.utils.data import DataLoader

from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.training.config import DatasetBuildContext


@pytest.mark.run_only_on("GPU")
class TestVLMHFMasking:
    def test_hf_vlm_label_masking_and_alignment(self):
        try:
            from transformers import AutoProcessor  # noqa: F401
        except Exception:
            pytest.skip("transformers not available")

        hf_processor = os.environ.get("HF_VLM_PROCESSOR", "Qwen/Qwen2.5-VL-3B-Instruct")

        provider = HFDatasetConversationProvider(
            sequence_length=256,
            hf_processor_path=hf_processor,
            maker_name="rdr",  # small and public HF dataset
            num_workers=0,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=False,
            persistent_workers=False,
        )

        context = DatasetBuildContext(train_samples=16, valid_samples=0, test_samples=0, tokenizer=None)
        train_ds, _, _ = provider.build_datasets(context)
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

        labels = batch["labels"]
        loss_mask = batch["loss_mask"].to(dtype=torch.bool)

        # Where loss_mask == 0, labels must be -100
        assert torch.all(labels[~loss_mask] == -100)

        # Where loss_mask == 1, labels should not be -100
        has_unmasked = torch.any(loss_mask, dim=1)
        if torch.any(has_unmasked):
            assert torch.all(labels[loss_mask] != -100)

        # At least one unmasked token in batch
        per_sample_unmasked = torch.sum(loss_mask, dim=1)
        assert torch.any(per_sample_unmasked > 0)

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

        examples_batch = getattr(train_ds, "_last_batch_examples")
        for i in range(labels.size(0)):
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
                ok = False
            assert ok
