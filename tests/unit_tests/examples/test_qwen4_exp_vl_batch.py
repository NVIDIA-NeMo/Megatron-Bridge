# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU regressions for whole-sample admission and EP padding in the VL recipe."""

import argparse
import ast
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3] / "examples/models/qwen/qwen4_exp"
pytestmark = pytest.mark.unit


def load_guard(path):
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("require", "inspect_batch")
    ]
    namespace = {"torch": torch, "argparse": argparse, "Any": Any, "hashlib": hashlib, "SELECTED_RECORDS": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class PaddingTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_guard(ROOT / "finetune_nemotron_image_v3.py")
        self.args = argparse.Namespace(sequence_length=2048)
        self.model = SimpleNamespace(vocab_size=100, image_token_id=7, vision_config={"spatial_merge_size": 2})
        self.tokens = torch.zeros((1, 2048), dtype=torch.int64)
        self.tokens[:, :9] = 1
        self.tokens[:, 0] = 7
        self.labels = torch.full_like(self.tokens, -100)
        self.labels[:, :8] = 2
        self.mask = torch.zeros((1, 2048), dtype=torch.float32)
        self.mask[:, :8] = 1
        self.attention = torch.zeros_like(self.tokens)
        self.attention[:, :9] = 1
        self.batch = {
            "input_ids": self.tokens,
            "labels": self.labels,
            "loss_mask": self.mask,
            "attention_mask": self.attention,
            "visual_inputs": SimpleNamespace(
                normalized_for_model=lambda: {
                    "pixel_values": torch.ones((4, 12)),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                }
            ),
        }
        records = {}
        for name, tensor in (("input_ids", self.tokens), ("labels", self.labels), ("loss_mask", self.mask)):
            prefix = tensor[:, :16].contiguous()
            records[name] = {
                "sha256": hashlib.sha256(prefix.numpy().tobytes()).hexdigest(),
                "shape": [1, 16],
                "dtype": str(prefix.dtype),
            }
        self.ns["SELECTED_RECORDS"] = {records["input_ids"]["sha256"]: records}

    def run_guard(self):
        return self.ns["inspect_batch"](self.batch, self.args, self.model)

    def test_native_ep_padding_passes_with_distinct_full_and_prefix_records(self):
        result = self.run_guard()
        self.assertEqual((result["raw_sample_tokens"], result["selected_extent"]), (9, 16))
        self.assertEqual(result["selected_prefix_records"]["input_ids"]["shape"], [1, 16])
        self.assertEqual(result["native_padded_records"]["input_ids"]["shape"], [1, 2048])
        self.assertNotEqual(result["selected_prefix_records"]["input_ids"]["sha256"], result["input_ids_sha256"])

    def test_token_label_and_loss_prefix_changes_rejected(self):
        for tensor, value in ((self.tokens, 2), (self.labels, 3), (self.mask, 0)):
            before = tensor[0, 3].item()
            tensor[0, 3] = value
            with self.assertRaises(ValueError):
                self.run_guard()
            tensor[0, 3] = before

    def test_extra_ep_padding_changes_rejected(self):
        for tensor, value in ((self.tokens, 1), (self.labels, 0), (self.mask, 1)):
            before = tensor[0, 16].item()
            tensor[0, 16] = value
            with self.assertRaises(ValueError):
                self.run_guard()
            tensor[0, 16] = before

    def test_preselected_alignment_padding_is_hash_authenticated(self):
        self.tokens[0, 12] = 1
        with self.assertRaises(ValueError):
            self.run_guard()

    def test_attention_hole_nonbinary_and_missing_rejected(self):
        self.attention[0, 2] = 0
        with self.assertRaises(ValueError):
            self.run_guard()
        self.attention[0, 2] = 2
        with self.assertRaises(ValueError):
            self.run_guard()
        del self.batch["attention_mask"]
        with self.assertRaises(ValueError):
            self.run_guard()

    def test_short_native_extent_rejected(self):
        for key in ("input_ids", "labels", "loss_mask", "attention_mask"):
            self.batch[key] = self.batch[key][:, :16]
        with self.assertRaises(ValueError):
            self.run_guard()


if __name__ == "__main__":
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("CPU-only tests require no CUDA/PG")
    unittest.main()
