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
Unit tests for Qwen3VL Self Attention implementation.

Run with: uv run pytest tests/unit_tests/models/qwen_vl/modelling_qwen3_vl/test_attention.py"""

import datetime
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import AttnMaskType

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import attention as qwen_attention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import (
    Qwen3VLSelfAttention,
    _get_packed_rotary_metadata,
)


class TestQwen3VLSelfAttention:
    @classmethod
    def setup_class(cls):
        """Setup distributed process group once for all tests in this class."""
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"

            device_count = torch.cuda.device_count()
            if device_count > 0:
                torch.cuda.set_device(0)

            dist.init_process_group(
                backend="nccl" if device_count > 0 else "gloo",
                world_size=1,
                rank=0,
                timeout=datetime.timedelta(minutes=30),
            )

    @classmethod
    def teardown_class(cls):
        """Teardown distributed process group once after all tests in this class."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_parallel_state(self, tp_size=1, ep_size=1, pp_size=1, cp_size=1):
        """Setup Megatron parallel state with specified parallelism configuration.

        Args:
            tp_size: Tensor model parallel size
            ep_size: Expert model parallel size
            pp_size: Pipeline model parallel size
            cp_size: Context parallel size
        """
        # Clean up any existing parallel state before initializing
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=1,
        )

        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        """Teardown Megatron parallel state after each test method."""
        parallel_state.destroy_model_parallel()

    def test_local_causal_attention_matches_reference_with_gqa(self):
        query = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]],
                [[[0.5, 0.5], [1.0, 0.0], [0.5, -0.5], [0.0, 1.0]]],
                [[[1.5, -0.5], [0.5, 1.0], [1.0, 0.0], [-0.5, 1.5]]],
            ]
        )
        key = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 1.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [1.0, -1.0]]],
            ]
        )
        value = torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[5.0, 6.0], [7.0, 8.0]]],
                [[[9.0, 10.0], [11.0, 12.0]]],
            ]
        )
        scale = 0.5

        repeated_key = key.repeat_interleave(2, dim=2)
        repeated_value = value.repeat_interleave(2, dim=2)
        scores = torch.einsum("sbhd,tbhd->bhst", query.float(), repeated_key.float()) * scale
        causal_mask = torch.ones((query.shape[0], key.shape[0]), dtype=torch.bool).triu(
            1 + key.shape[0] - query.shape[0]
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        expected = torch.einsum("bhst,tbhd->sbhd", probs, repeated_value).reshape(query.shape[0], 1, -1)

        actual = Qwen3VLSelfAttention._local_causal_attention(query, key, value, scale)

        assert actual.shape == (3, 1, 8)
        torch.testing.assert_close(actual, expected)

    def test_local_causal_attention_aligns_static_cache_mask(self):
        query = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
        key = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[1.0, 1.0]]]])
        value = torch.tensor([[[[1.0, 2.0]]], [[[3.0, 4.0]]], [[[5.0, 6.0]]]])
        scale = 1.0

        scores = torch.einsum("sbhd,tbhd->bhst", query.float(), key.float()) * scale
        causal_mask = torch.tensor([[False, False, True], [False, False, False]])
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        expected = torch.einsum("bhst,tbhd->sbhd", probs, value).reshape(query.shape[0], 1, -1)

        actual = Qwen3VLSelfAttention._local_causal_attention(query, key, value, scale)

        assert actual.shape == (2, 1, 2)
        torch.testing.assert_close(actual, expected)

    def test_packed_rotary_metadata_propagates_q_and_kv_max_seqlen(self):
        """Qwen's direct rotary call receives layout-specific padded metadata."""
        q = torch.tensor([0, 5, 12], dtype=torch.int32)
        kv = torch.tensor([0, 7, 16], dtype=torch.int32)
        q_padded = torch.tensor([0, 8, 16], dtype=torch.int32)
        kv_padded = torch.tensor([0, 10, 20], dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=q,
            cu_seqlens_kv=kv,
            cu_seqlens_q_padded=q_padded,
            cu_seqlens_kv_padded=kv_padded,
            max_seqlen_q=8,
            max_seqlen_kv=10,
        )

        actual_q, actual_max_q = _get_packed_rotary_metadata(packed, for_query=True)
        actual_kv, actual_max_kv = _get_packed_rotary_metadata(packed, for_query=False)

        assert actual_q is q_padded
        assert actual_kv is kv_padded
        assert actual_max_q == 8
        assert actual_max_kv == 10

    def test_inference_materializes_raw_mrope_and_propagates_q_k_metadata(self, monkeypatch):
        """Static inference materializes raw mRoPE before cache adjustment and rotary calls."""
        sequence_length = 4
        hidden_size = 8
        raw_freqs = torch.randn(3, 1, sequence_length, hidden_size // 2)
        expected_freqs = qwen_attention.materialize_mrope_freqs(
            raw_freqs,
            [2, 1, 1],
            interleaved_mrope=True,
        )
        cu_seqlens_q = torch.tensor([0, sequence_length], dtype=torch.int32)
        cu_seqlens_kv = torch.tensor([0, sequence_length], dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q_padded=cu_seqlens_q,
            cu_seqlens_kv_padded=cu_seqlens_kv,
            max_seqlen_q=7,
            max_seqlen_kv=11,
        )
        cp_group = object()
        rotary_calls = []

        def fake_apply_rotary(tensor, freqs, config, cu_seqlens=None, *, cp_group=None, max_seqlen=None):
            rotary_calls.append((freqs, cu_seqlens, cp_group, max_seqlen))
            return tensor

        def adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        ):
            del (
                inference_context,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
            assert isinstance(rotary_pos_emb, tuple)
            assert len(rotary_pos_emb) == 2
            torch.testing.assert_close(rotary_pos_emb[0], expected_freqs)
            torch.testing.assert_close(rotary_pos_emb[1], expected_freqs)
            return query, key, value, rotary_pos_emb, AttnMaskType.causal, None

        monkeypatch.setattr(qwen_attention, "apply_rotary_pos_emb_absolute", fake_apply_rotary)
        monkeypatch.setattr(qwen_attention, "nvtx_range_push", lambda **kwargs: None)
        monkeypatch.setattr(qwen_attention, "nvtx_range_pop", lambda **kwargs: None)

        query = torch.randn(sequence_length, 1, 1, hidden_size)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        attention = SimpleNamespace(
            config=SimpleNamespace(
                no_rope_freq=None,
                flash_decode=False,
                mrope_section=[2, 1, 1],
                mrope_interleaved=True,
                rotary_interleaved=False,
                attention_output_gate=False,
            ),
            layer_number=1,
            training=False,
            pg_collection=SimpleNamespace(cp=cp_group),
            get_query_key_value_tensors=lambda hidden_states, key_value_states: (query, key, value),
            _adjust_key_value_for_inference=adjust_key_value_for_inference,
            core_attention=lambda query, key, value, attention_mask, **kwargs: query,
            checkpoint_core_attention=False,
            linear_proj=lambda context: (context, None),
        )
        inference_context = SimpleNamespace(
            is_dynamic_batching=lambda: False,
            is_decode_only=lambda: False,
            is_static_batching=lambda: True,
        )

        output, bias = Qwen3VLSelfAttention.forward(
            attention,
            torch.randn(sequence_length, 1, hidden_size),
            attention_mask=None,
            inference_context=inference_context,
            rotary_pos_emb=raw_freqs,
            packed_seq_params=packed,
        )

        assert output.shape == (sequence_length, 1, hidden_size)
        assert bias is None
        assert len(rotary_calls) == 2
        torch.testing.assert_close(rotary_calls[0][0], expected_freqs)
        torch.testing.assert_close(rotary_calls[1][0], expected_freqs)
        assert rotary_calls[0][1:] == (cu_seqlens_q, cp_group, 7)
        assert rotary_calls[1][1:] == (cu_seqlens_kv, cp_group, 11)

    def run_self_attention(self, pg_collection):
        tensor_model_parallel_size = torch.distributed.get_world_size(pg_collection.tp)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=tensor_model_parallel_size,
            use_cpu_initialization=False,
        )
        self.self_attention = Qwen3VLSelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pg_collection,
        )

        config = self.self_attention.config
        sequence_length = 127
        micro_batch_size = 2

        self.self_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.self_attention.config.hidden_size),
            device="cuda",
        )

        output, bias = self.self_attention(hidden_states, None)
        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_self_attention_mpu(self):
        self._setup_parallel_state(tp_size=1, ep_size=1, pp_size=1)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        assert pg_collection is not None
        assert pg_collection.tp is not None
        assert pg_collection.pp is not None
        assert pg_collection.cp is not None
        assert pg_collection.embd is not None

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.run_self_attention(pg_collection)
