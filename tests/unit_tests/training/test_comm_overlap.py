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
from unittest.mock import MagicMock, patch

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.t5_provider import T5ModelProvider
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    TransformerLayerTPOverlapCfg,
    _CommOverlapConfig,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import DistributedDataParallelConfig, OptimizerConfig


def create_gpt_config(**kwargs):
    """Helper function to create a valid GPTConfig with defaults."""
    defaults = {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_layers": 12,
        "ffn_hidden_size": None,
        "kv_channels": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "sequence_parallel": False,
        "context_parallel_size": 1,
    }
    # Add pipeline_dtype if using pipeline parallelism
    if kwargs.get("pipeline_model_parallel_size", defaults["pipeline_model_parallel_size"]) > 1:
        defaults["pipeline_dtype"] = "fp32"
    defaults.update(kwargs)
    return GPTModelProvider(**defaults)


def create_t5_config(**kwargs):
    """Helper function to create a valid T5Config with defaults."""
    defaults = {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_layers": 12,
        "ffn_hidden_size": None,
        "kv_channels": None,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "sequence_parallel": False,
        "context_parallel_size": 1,
        "apply_rope_fusion": False,  # Disable RoPE fusion to avoid dependency issues
    }
    # Add pipeline_dtype if using pipeline parallelism
    if kwargs.get("pipeline_model_parallel_size", defaults["pipeline_model_parallel_size"]) > 1:
        defaults["pipeline_dtype"] = "fp32"
    defaults.update(kwargs)
    return T5ModelProvider(**defaults)


class TestMegatronCommOverlapConfig:
    def test_finalize(self):
        cfg = CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
            tp_comm_bootstrap_backend="nccl",
            data_parallel_size=2,
        )
        cfg.finalize()

        assert cfg.tp_comm_overlap is True
        assert cfg.tp_comm_overlap_cfg is None  # Should be reset to None in finalize()
        assert cfg.tp_comm_bootstrap_backend is None  # Should be reset to None in finalize()
        assert cfg.user_comm_overlap_cfg.tp_comm_overlap is True
        assert isinstance(cfg.user_comm_overlap_cfg.tp_comm_overlap_cfg, TransformerLayerTPOverlapCfg)

    def test_get_model_comm_overlap_cfgs_with_tp_disabled(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
        assert result.tp_comm_overlap is False
        assert result.overlap_p2p_comm is False
        assert result.batch_p2p_comm is False

    @patch("megatron.bridge.training.comm_overlap.HAVE_TE", False)
    def test_get_model_comm_overlap_cfgs_no_te(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=True,
            num_attention_heads=16,  # Must be divisible by TP size
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("megatron.bridge.training.comm_overlap.logging.warning") as mock_warning:
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
            assert result.tp_comm_overlap is False
            mock_warning.assert_called_with("Disabling tensor parallel communication overlap due to TE not detected.")

    def test_get_model_comm_overlap_cfgs_tp_size_too_small(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,  # Cannot use sequence_parallel with TP size 1
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("megatron.bridge.training.comm_overlap.logging.warning") as mock_warning:
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
            assert result.tp_comm_overlap is False
            mock_warning.assert_called_with("Disabling tensor parallel communication overlap due to TP size < 2.")

    def test_get_model_comm_overlap_cfgs_no_sequence_parallel(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            num_attention_heads=16,  # Must be divisible by TP size
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)
        with patch("megatron.bridge.training.comm_overlap.logging.warning") as mock_warning:
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
            assert result.tp_comm_overlap is False
            mock_warning.assert_called_with(
                "Disabling tensor parallel communication overlap due to sequence_parallel=False."
            )

    def test_get_model_comm_overlap_cfgs_pp_with_vp(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            sequence_parallel=False,
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)
        result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
        assert result.overlap_p2p_comm is True
        assert result.batch_p2p_comm is False

    def test_get_model_comm_overlap_cfgs_pp_without_vp(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=1,
            sequence_parallel=False,
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)
        result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
        assert result.overlap_p2p_comm is False
        assert result.batch_p2p_comm is True

    def test_get_optimizer_overlap_cfgs_dp_enabled(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=4)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )

        result = comm_cfg._get_optimizer_overlap_cfgs(model_cfg)
        assert result.bucket_size == 128 * 1024 * 1024
        assert result.overlap_grad_reduce is True
        assert result.overlap_param_gather is True
        assert result.overlap_param_gather_with_optimizer_step is False
        assert result.align_param_gather is False

    def test_get_optimizer_overlap_cfgs_dp_disabled(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )

        result = comm_cfg._get_optimizer_overlap_cfgs(model_cfg)
        assert result.bucket_size is None
        assert result.overlap_grad_reduce is False
        assert result.overlap_param_gather is False

    def test_get_optimizer_overlap_cfgs_with_pp_vp(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=4)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            sequence_parallel=False,
        )

        result = comm_cfg._get_optimizer_overlap_cfgs(model_cfg)
        assert result.align_param_gather is True

    def test_apply_cfgs(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()

        src_cfg = _CommOverlapConfig(
            tp_comm_overlap=True, overlap_p2p_comm=True, batch_p2p_comm=False, bucket_size=1024
        )

        dest_cfg = MagicMock()
        dest_cfg.tp_comm_overlap = False
        dest_cfg.overlap_p2p_comm = False
        dest_cfg.batch_p2p_comm = True
        dest_cfg.bucket_size = 0

        comm_cfg._apply_cfgs(src_cfg, dest_cfg)

        assert dest_cfg.tp_comm_overlap is True
        assert dest_cfg.overlap_p2p_comm is True
        assert dest_cfg.batch_p2p_comm is False
        assert dest_cfg.bucket_size == 1024

    def test_override_user_cfgs(self):
        user_cfg = _CommOverlapConfig(tp_comm_overlap=True, overlap_p2p_comm=True, bucket_size=2048)

        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        comm_cfg.user_comm_overlap_cfg = user_cfg

        default_cfg = _CommOverlapConfig(
            tp_comm_overlap=False, overlap_p2p_comm=False, batch_p2p_comm=True, bucket_size=1024
        )

        result = comm_cfg._override_user_cfgs(default_cfg)

        assert result.tp_comm_overlap is True  # Overridden by user
        assert result.overlap_p2p_comm is True  # Overridden by user
        assert result.batch_p2p_comm is True  # Not overridden (user didn't specify)
        assert result.bucket_size == 2048  # Overridden by user

    @patch("torch.cuda.get_device_capability", return_value=(10, 0))
    def test_set_num_cuda_device_max_connections_hopper_multi_parallel(self, mock_cuda):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=2)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=True,
            num_attention_heads=16,  # Must be divisible by TP size
        )

        with patch.dict(os.environ, {}, clear=True):
            comm_cfg._set_num_cuda_device_max_connections(model_cfg)
            assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "32"

    @patch("torch.cuda.get_device_capability", return_value=(10, 0))
    def test_set_num_cuda_device_max_connections_hopper_single_parallel(self, mock_cuda):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )

        with patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "16"}, clear=False):
            comm_cfg._set_num_cuda_device_max_connections(model_cfg)
            assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ

    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_set_num_cuda_device_max_connections_ampere_with_tp(self, mock_cuda):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=True,
            num_attention_heads=16,  # Must be divisible by TP size
        )

        with patch.dict(os.environ, {}, clear=True):
            comm_cfg._set_num_cuda_device_max_connections(model_cfg)
            assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") == "1"

    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_set_num_cuda_device_max_connections_ampere_no_tp(self, mock_cuda):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=1)
        comm_cfg.finalize()
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )

        with patch.dict(os.environ, {"CUDA_DEVICE_MAX_CONNECTIONS": "8"}, clear=False):
            comm_cfg._set_num_cuda_device_max_connections(model_cfg)
            assert "CUDA_DEVICE_MAX_CONNECTIONS" not in os.environ

    @patch("megatron.bridge.training.comm_overlap.HAVE_TE", True)
    def test_setup_method_complete(self):
        tp_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=tp_overlap_cfg,
            tp_comm_bootstrap_backend="nccl",
            overlap_p2p_comm=True,
            data_parallel_size=4,
        )
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=True,
            num_attention_heads=16,  # Must be divisible by TP size
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=True)

        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            with patch.dict(os.environ, {}, clear=True):
                comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

        # Check model config was updated
        assert model_cfg.tp_comm_overlap is True
        assert isinstance(model_cfg.tp_comm_overlap_cfg, dict)
        assert model_cfg.tp_comm_bootstrap_backend == "nccl"

        # Check optimizer config was updated (if attributes exist)
        if hasattr(optimizer_cfg, "overlap_grad_reduce"):
            assert optimizer_cfg.overlap_grad_reduce is True
            assert optimizer_cfg.overlap_param_gather is True
            assert optimizer_cfg.bucket_size == 128 * 1024 * 1024

        # Check DDP config was updated
        assert ddp_cfg.overlap_grad_reduce is True
        assert ddp_cfg.bucket_size == 128 * 1024 * 1024

    def test_setup_with_t5_config(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=2)
        comm_cfg.finalize()

        model_cfg = create_t5_config(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=True,
            num_attention_heads=16,  # Must be divisible by TP size
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=True)

        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            with patch.dict(os.environ, {}, clear=True):
                comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

        # Check configs were updated appropriately
        assert model_cfg.tp_comm_overlap is False

        # Check optimizer config was updated (if attributes exist)
        if hasattr(optimizer_cfg, "overlap_grad_reduce"):
            assert optimizer_cfg.overlap_grad_reduce is True

        # Check DDP config was updated
        if hasattr(ddp_cfg, "overlap_param_gather"):
            assert ddp_cfg.overlap_param_gather is True

    def test_setup_without_distributed_optimizer(self):
        comm_cfg = CommOverlapConfig(tp_comm_overlap=False, data_parallel_size=4)
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        # Store original values
        orig_overlap_grad_reduce = getattr(optimizer_cfg, "overlap_grad_reduce", None)

        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

        # Check that optimizer config was NOT updated
        assert getattr(optimizer_cfg, "overlap_grad_reduce", None) == orig_overlap_grad_reduce

    def test_user_override_pp_overlap(self):
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=False,
            overlap_p2p_comm=False,  # User explicitly sets to False
            batch_p2p_comm=False,  # User explicitly sets to False
            data_parallel_size=1,
        )
        comm_cfg.finalize()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=4,
            virtual_pipeline_model_parallel_size=2,
            sequence_parallel=False,
        )

        result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
        # Even though PP > 1 and VP > 1, user override should take precedence
        assert result.overlap_p2p_comm is False
        assert result.batch_p2p_comm is False

    def test_tp_overlap_config_conversion_to_dict(self):
        tp_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, tp_comm_overlap_cfg=tp_overlap_cfg, data_parallel_size=1)
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            num_attention_heads=16,
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            with patch("megatron.bridge.training.comm_overlap.HAVE_TE", True):
                comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

        # Check that tp_comm_overlap_cfg was converted to dict
        assert isinstance(model_cfg.tp_comm_overlap_cfg, dict)
        assert "qkv_dgrad" in model_cfg.tp_comm_overlap_cfg
        assert "proj_fprop" in model_cfg.tp_comm_overlap_cfg

        # Check that None values were filtered out (TE expectation)
        for key, value in model_cfg.tp_comm_overlap_cfg.items():
            assert value is not None, f"Found None value for key '{key}' - should have been filtered out"

    @patch("megatron.bridge.training.comm_overlap.HAVE_TE", True)
    def test_tp_overlap_with_no_config_provided(self):
        """Test that TP overlap handles the case when no tp_comm_overlap_cfg is provided."""
        # Create CommOverlapConfig with tp_comm_overlap=True but no tp_comm_overlap_cfg
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, data_parallel_size=1)
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            num_attention_heads=16,
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            with patch("megatron.bridge.training.comm_overlap.logging.warning") as mock_warning:
                comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

                # Check that warning was logged for missing tp_comm_overlap_cfg
                mock_warning.assert_called_with(
                    "Tensor parallel overlap: No overlap config provided. "
                    "Initializing TP comm overlap with the default config."
                )

        # Check that model config was updated correctly
        assert model_cfg.tp_comm_overlap is True
        assert model_cfg.tp_comm_overlap_cfg is None
        assert model_cfg.tp_comm_bootstrap_backend is None

    def test_tp_overlap_config_filters_none_values(self):
        """Test that None values are filtered from tp_comm_overlap_cfg dict."""
        from dataclasses import dataclass

        # Create a mock config with some None values
        @dataclass
        class MockTPOverlapCfg:
            qkv_dgrad: str = "qkv_dgrad"
            proj_fprop: str = "proj_fprop"
            none_field1: str = None
            valid_field: str = "valid"
            none_field2: int = None

        mock_tp_cfg = MockTPOverlapCfg()
        comm_cfg = CommOverlapConfig(tp_comm_overlap=True, tp_comm_overlap_cfg=mock_tp_cfg, data_parallel_size=1)
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            sequence_parallel=True,
            num_attention_heads=16,
        )

        optimizer_cfg = OptimizerConfig()
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
            comm_cfg.setup(model_cfg, optimizer_cfg, ddp_cfg)

        # Check that config was converted to dict and None values filtered
        assert isinstance(model_cfg.tp_comm_overlap_cfg, dict)
        expected_keys = {"qkv_dgrad", "proj_fprop", "valid_field"}
        filtered_keys = {"none_field1", "none_field2"}

        # Check that valid keys are present
        for key in expected_keys:
            assert key in model_cfg.tp_comm_overlap_cfg
            assert model_cfg.tp_comm_overlap_cfg[key] is not None

        # Check that None value keys are filtered out
        for key in filtered_keys:
            assert key not in model_cfg.tp_comm_overlap_cfg

    def test_moe_ep_overlap_config_validation(self):
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=False,
            data_parallel_size=1,
            overlap_moe_expert_parallel_comm=True,
        )
        comm_cfg.finalize()

        # Minimal valid config to pass MOE EP overlap assertions
        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            expert_model_parallel_size=2,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            bf16=True,
            fp16=False,
            recompute_granularity=None,
            recompute_method=None,
            recompute_num_layers=None,
            moe_shared_expert_overlap=False,
            mtp_num_layers=None,
            moe_use_legacy_grouped_gemm=False,
            add_bias_linear=False,
        )

        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("megatron.bridge.training.comm_overlap.is_torch_min_version", return_value=True):
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)

        assert result.overlap_moe_expert_parallel_comm is True

    def test_delay_wgrad_config_validation(self):
        """delay_wgrad_compute passes when TE and EP overlap conditions are met."""
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=False,
            data_parallel_size=1,
            delay_wgrad_compute=True,
            overlap_moe_expert_parallel_comm=True,
        )
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            expert_model_parallel_size=2,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            bf16=True,
            moe_use_legacy_grouped_gemm=False,
            add_bias_linear=False,
        )

        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("megatron.bridge.training.comm_overlap.is_te_min_version", return_value=True):
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
            assert result.delay_wgrad_compute is True

    def test_delay_wgrad_config_validation_with_overlap_grad_reduce(self):
        """delay_wgrad_compute passes when TE and EP overlap conditions are met."""
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=False,
            data_parallel_size=1,
            delay_wgrad_compute=True,
            overlap_moe_expert_parallel_comm=True,
        )
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            expert_model_parallel_size=2,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            bf16=True,
            moe_use_legacy_grouped_gemm=False,
            add_bias_linear=False,
            gradient_accumulation_fusion=True,
        )

        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=True, overlap_grad_reduce=True)

        with patch("megatron.bridge.training.comm_overlap.is_te_min_version", return_value=True):
            result = comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
            assert result.delay_wgrad_compute is True

    def test_delay_wgrad_requires_ep_overlap(self):
        """delay_wgrad_compute requires EP overlap to be enabled."""
        comm_cfg = CommOverlapConfig(
            tp_comm_overlap=False,
            data_parallel_size=1,
            delay_wgrad_compute=True,
            overlap_moe_expert_parallel_comm=False,
        )
        comm_cfg.finalize()

        model_cfg = create_gpt_config(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            sequence_parallel=False,
            expert_model_parallel_size=2,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            bf16=True,
            moe_use_legacy_grouped_gemm=False,
            add_bias_linear=False,
        )
        ddp_cfg = DistributedDataParallelConfig(use_distributed_optimizer=False)

        with patch("megatron.bridge.training.comm_overlap.is_te_min_version", return_value=True):
            try:
                comm_cfg._get_model_comm_overlap_cfgs(model_cfg, ddp_cfg)
                assert False, "Expected AssertionError when EP overlap is not enabled"
            except AssertionError:
                pass
