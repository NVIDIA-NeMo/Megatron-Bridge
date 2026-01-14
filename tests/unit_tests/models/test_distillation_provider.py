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

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models.distillation_provider import DistillationProvider, convert_to_distillation_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.qwen import Qwen3MoEModelProvider
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig


class TestGPTDistillationProvider:
    """Test cases for GPTDistillationProvider class."""

    def test_initialization_with_teacher(self):
        """Test GPTDistillationProvider can be initialized with a teacher."""
        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = convert_to_distillation_provider(student_base, teacher)

        assert student.teacher is teacher
        assert student.num_layers == 12
        assert student.hidden_size == 2048
        assert student.num_attention_heads == 16

    def test_initialization_without_teacher_raises_error(self):
        """Test GPTDistillationProvider raises error when teacher is None."""
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        with pytest.raises(AssertionError):
            convert_to_distillation_provider(student_base, None)

    def test_post_init_validates_shared_attributes(self):
        """Test __post_init__ validates that shared attributes match between student and teacher."""
        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=torch.float32,
        )

        # Test mismatched tensor_model_parallel_size
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=2,  # Different from teacher
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        with pytest.raises(ValueError):
            convert_to_distillation_provider(student_base, teacher)

    def test_post_init_validates_seq_length(self):
        """Test __post_init__ validates seq_length."""
        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=2048,
            pipeline_dtype=torch.float32,
        )

        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,  # Different from teacher
            pipeline_dtype=torch.float32,
        )
        with pytest.raises(ValueError):
            convert_to_distillation_provider(student_base, teacher)

    @patch("modelopt.torch.distill.plugins.megatron.parallel_state")
    @patch("megatron.bridge.models.gpt_provider.calculate_padded_vocab_size", return_value=1024)
    @patch("megatron.bridge.models.gpt_provider.MCoreGPTModel")
    def test_provide_method_creates_distillation_model(
        self,
        mock_mcore_gpt,
        mock_calc_vocab,
        mock_mtd_parallel_state,
    ):
        """Test provide method creates a ModelOpt DistillationModel."""
        mock_mtd_parallel_state.is_pipeline_first_stage.return_value = True
        mock_mtd_parallel_state.is_pipeline_last_stage.return_value = True

        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=4096,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = convert_to_distillation_provider(student_base, teacher, kd_config=ModelOptDistillConfig())

        # Attach minimal pg_collection needed by provider.provide
        pg = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()
        teacher._pg_collection = pg
        student._pg_collection = pg

        # Mock the provide method calls and modelopt functions
        mock_student_model = Mock()
        mock_teacher_model = Mock()
        mock_student_model.config = Mock()
        mock_teacher_model.config = Mock()
        # Avoid ProjectionLayer being created here
        mock_student_model.config.hidden_size = mock_teacher_model.config.hidden_size = 4096
        mock_kd_model = Mock()
        # Ensure that .parameters() callable returns an empty iterator
        mock_teacher_model.parameters.return_value = iter(())
        mock_kd_model.parameters.return_value = iter(())

        # Set the side effects for the model provider - student first, then teacher
        mock_mcore_gpt.side_effect = [mock_student_model, mock_teacher_model]
        with patch("megatron.bridge.models.distillation_provider.mtd.convert", return_value=mock_kd_model):
            result = student.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)

        # Verify that both student and teacher models were created
        assert mock_mcore_gpt.call_count == 2
        assert result[0] is mock_kd_model

    def test_setattr_mirrors_to_teacher(self):
        """Test __setattr__ mirrors attributes to teacher when teacher has that attribute."""
        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = convert_to_distillation_provider(student_base, teacher)

        student.num_layers = 10  # This exists on teacher, so it should be mirrored
        assert student.num_layers == 10
        assert teacher.num_layers == 10

    def test_setattr_does_not_mirror_when_teacher_lacks_attribute(self):
        """Test __setattr__ does not mirror attributes that teacher doesn't have."""
        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student_base = GPTModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = convert_to_distillation_provider(student_base, teacher)

        student.new_attribute = "test_value"  # Should not be reflected on teacher
        assert student.new_attribute == "test_value"
        assert not hasattr(teacher, "new_attribute")

    def test_convert_to_distillation_provider_preserves_original_provider(self):
        """Ensure convert_to_distillation_provider retains original provider behavior."""

        class CustomProvider(GPTModelProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.extra_attr = "custom-attr"
                self.custom_provide_calls = 0

            def provide(self, pre_process=None, post_process=None, vp_stage=None):
                self.custom_provide_calls += 1
                return "custom-result"

        teacher = GPTModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = CustomProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )

        original_bases = DistillationProvider.__bases__
        try:
            converted = convert_to_distillation_provider(student, teacher)

            assert converted is student
            assert isinstance(converted, DistillationProvider)
            assert isinstance(converted, CustomProvider)
            assert converted.extra_attr == "custom-attr"

            result = converted._super_class.provide(converted)
            assert result == "custom-result"
            assert converted.custom_provide_calls == 1
        finally:
            # Restore original bases since it was modified globally for the entire class
            DistillationProvider.__bases__ = original_bases

    def test_converted_provider_to_cfg_dict_preserves_original_provider(self):
        """Ensure converted provider to_cfg_dict retains original provider behavior."""

        teacher = Qwen3MoEModelProvider(
            num_layers=24,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student = Qwen3MoEModelProvider(
            num_layers=12,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )

        original_bases = DistillationProvider.__bases__
        try:
            converted = convert_to_distillation_provider(student, teacher)
            cfg_dict = converted.to_cfg_dict()

            assert cfg_dict["_target_"] == "megatron.bridge.models.qwen.qwen_provider.Qwen3MoEModelProvider"
        finally:
            # Restore original bases since it was modified globally for the entire class
            DistillationProvider.__bases__ = original_bases

    def test_convert_mamba_provider_to_distillation_provider(self):
        """Test that MambaModelProvider can be converted to DistillationProvider."""
        from megatron.bridge.models.distillation_provider import DistillationProvider

        teacher = MambaModelProvider(
            num_layers=48,
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        student_base = MambaModelProvider(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )

        original_bases = DistillationProvider.__bases__
        try:
            converted = convert_to_distillation_provider(student_base, teacher)

            # Verify conversion succeeded
            assert converted is student_base
            assert isinstance(converted, DistillationProvider)
            assert isinstance(converted, MambaModelProvider)
            assert converted.teacher is teacher
            assert converted.num_layers == 24
            assert converted.hidden_size == 2048
            assert converted.num_attention_heads == 16
        finally:
            # Restore original bases since it was modified globally for the entire class
            DistillationProvider.__bases__ = original_bases
