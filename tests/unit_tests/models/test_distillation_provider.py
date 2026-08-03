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
from transformers import LlamaConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.distillation_provider import (
    DistillationModelConfig,
    DistillationProvider,
    convert_to_distillation_provider,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig


def _llama_builder_config(hidden_size: int, num_hidden_layers: int):
    """Create a builder-backed Llama config with non-default runtime fields."""
    hf_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=16384,
        num_attention_heads=8,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=4,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
        },
        rope_theta=500000.0,
        tie_word_embeddings=True,
        vocab_size=128256,
    )
    return AutoBridge.from_hf_config(hf_config).get_model_config()


class TestDistillationProvider:
    """Test cases for DistillationProvider class."""

    def test_builder_config_preserves_runtime_fields(self):
        """Builder distillation keeps the exact student and teacher configs."""
        student = _llama_builder_config(hidden_size=1024, num_hidden_layers=8)
        teacher = _llama_builder_config(hidden_size=2048, num_hidden_layers=16)
        student_fields = {
            "rotary_base": student.rotary_base,
            "rope_scaling": student.rope_scaling,
            "rope_scaling_factor": student.rope_scaling_factor,
            "share_embeddings_and_output_weights": student.share_embeddings_and_output_weights,
        }

        converted = convert_to_distillation_provider(student, teacher)
        converted.finalize()

        assert converted is not student
        assert isinstance(converted, DistillationModelConfig)
        serialized = converted.as_dict()
        assert serialized["_target_"] == f"{type(student).__module__}.{type(student).__qualname__}"
        assert "teacher" not in serialized
        assert converted.teacher is teacher
        assert converted.rotary_base == student_fields["rotary_base"]
        assert converted.rope_scaling == student_fields["rope_scaling"]
        assert converted.rope_scaling_factor == student_fields["rope_scaling_factor"]
        assert converted.share_embeddings_and_output_weights is student_fields["share_embeddings_and_output_weights"]
        assert converted.pre_wrap_hooks[-1] == converted._convert_hook
        converted.seq_length = 4096
        assert converted.teacher.seq_length == 4096

    def test_builder_config_converts_student_in_place(self):
        """Builder conversion preserves the chunk identity expected by training."""
        student = _llama_builder_config(hidden_size=1024, num_hidden_layers=8)
        teacher = _llama_builder_config(hidden_size=2048, num_hidden_layers=16)
        converted = convert_to_distillation_provider(student, teacher)
        student_model = Mock()
        teacher_model = Mock()
        student_model.config = student.transformer
        teacher_model.config = teacher.transformer
        student_chunk = Mock()
        teacher_chunk = Mock()
        student_chunk.module = student_model
        teacher_chunk.module = teacher_model
        teacher_builder = Mock()
        teacher_builder.build_distributed_models.return_value = [teacher_chunk]
        kd_model = Mock()

        with (
            patch.object(type(teacher), "get_builder_cls", return_value=Mock(return_value=teacher_builder)),
            patch("megatron.bridge.models.distillation_provider.ProcessGroupCollection.use_mpu_process_groups"),
            patch("megatron.bridge.models.distillation_provider.mtd.convert", return_value=kd_model),
            patch("megatron.bridge.models.distillation_provider.mtd_mcore.setup_distillation_config") as setup,
            patch("megatron.bridge.models.distillation_provider.mtd_mcore.adjust_distillation_model_for_mcore"),
        ):
            setup.return_value = Mock(criterion=None, loss_balancer=None)
            result = converted._convert_hook([student_chunk])

        assert result == [kd_model]

    def test_builder_config_rejects_mismatched_parallelism(self):
        """Builder distillation validates settings shared by both models."""
        student = _llama_builder_config(hidden_size=1024, num_hidden_layers=8)
        teacher = _llama_builder_config(hidden_size=2048, num_hidden_layers=16)
        teacher.tensor_model_parallel_size = 2

        with pytest.raises(ValueError, match="tensor_model_parallel_size"):
            converted = convert_to_distillation_provider(student, teacher)
            converted.finalize()

    def test_initialization_with_teacher(self):
        """Test DistillationProvider can be initialized with a teacher."""
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
        """Test DistillationProvider raises error when teacher is None."""
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

    @patch("megatron.bridge.models.model_provider.ProcessGroupCollection.use_mpu_process_groups")
    @patch("megatron.bridge.models.model_provider.parallel_state.is_initialized", return_value=True)
    @patch("megatron.bridge.models.model_provider.torch.distributed.is_initialized", return_value=True)
    @patch("megatron.bridge.models.model_provider.torch.cuda.set_device")
    @patch("modelopt.torch.distill.plugins.megatron.parallel_state")
    @patch("megatron.bridge.models.gpt_provider.calculate_padded_vocab_size", return_value=1024)
    @patch("megatron.bridge.models.gpt_provider.MCoreGPTModel")
    def test_provide_method_creates_distillation_model(
        self,
        mock_mcore_gpt,
        mock_calc_vocab,
        mock_mtd_parallel_state,
        mock_set_device,
        mock_distributed_initialized,
        mock_model_parallel_initialized,
        mock_pg_collection,
    ):
        """Test the KD conversion runs (in the deferred pre-wrap hook) and yields a DistillationModel."""
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
        process_group = Mock()
        process_group.size.return_value = 1
        process_group.rank.return_value = 0
        pg = type(
            "PG",
            (),
            {"pp": process_group, "tp": process_group, "cp": process_group, "dp": process_group},
        )()
        teacher._pg_collection = pg
        student._pg_collection = pg
        mock_pg_collection.return_value = pg

        # Mock the provide method calls and modelopt functions
        mock_student_model = Mock()
        mock_teacher_model = Mock()
        mock_student_model.config = Mock()
        mock_teacher_model.config = Mock()
        # Avoid ProjectionLayer being created here
        mock_student_model.config.hidden_size = mock_teacher_model.config.hidden_size = 4096
        mock_kd_model = Mock()
        # Ensure that .parameters() callable returns an empty iterator. The KD conversion is deferred,
        # so provide() now returns the un-converted student, which get_model iterates before the hook.
        mock_student_model.parameters.return_value = iter(())
        mock_teacher_model.parameters.return_value = iter(())
        mock_kd_model.parameters.return_value = iter(())

        # Set the side effects for the model provider - student first, then teacher
        mock_mcore_gpt.side_effect = [mock_student_model, mock_teacher_model]
        # The KD conversion is deferred to the ``_convert_hook`` pre-wrap hook, which unwraps the
        # (mocked) student/teacher chunks; make ``unwrap_model`` an identity for the Mock chunks.
        with (
            patch("megatron.bridge.models.distillation_provider.mtd.convert", return_value=mock_kd_model),
            patch("megatron.bridge.models.distillation_provider.unwrap_model", side_effect=lambda m: m),
        ):
            result = student.provide_distributed_model(wrap_with_ddp=False, mixed_precision_wrapper=None)

        # Verify that both student and teacher models were created
        assert mock_mcore_gpt.call_count == 2
        assert result[0] is mock_kd_model

    def _make_pair(self):
        kwargs = dict(
            hidden_size=4096,
            num_attention_heads=32,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            seq_length=1024,
            pipeline_dtype=None,
        )
        teacher = GPTModelProvider(num_layers=24, **kwargs)
        student_base = GPTModelProvider(num_layers=12, **kwargs)
        return student_base, teacher

    def test_convert_hook_registered_last_for_qad_ordering(self):
        """_convert_hook is appended last so a prepend=True QAD hook runs before the KD conversion."""
        student_base, teacher = self._make_pair()
        student = convert_to_distillation_provider(student_base, teacher)

        assert student._pre_wrap_hooks[-1] == student._convert_hook

        def restore_hook(model_chunks):
            return model_chunks

        student.register_pre_wrap_hook(restore_hook, prepend=True)
        assert student._pre_wrap_hooks[0] is restore_hook
        assert student._pre_wrap_hooks[-1] == student._convert_hook

    def test_convert_hook_distill_submodule(self):
        """distill_submodule distills only the submodule; full_model is retained for export."""
        student_base, teacher = self._make_pair()
        student = convert_to_distillation_provider(
            student_base, teacher, kd_config=ModelOptDistillConfig(), distill_submodule="language_model"
        )

        # Full (VLM-like) models: each carries a ``.language_model`` submodule with a config.
        student_full, teacher_full = Mock(), Mock()
        student_full.language_model.config = Mock()
        teacher_full.language_model.config = Mock()
        student_full.language_model.config.hidden_size = teacher_full.language_model.config.hidden_size = 4096
        student_full.language_model.parameters.return_value = iter(())
        teacher_full.language_model.parameters.return_value = iter(())
        # The teacher's full model is built inside _convert_hook.
        teacher.provide_distributed_model = Mock(return_value=[teacher_full])

        with (
            patch("modelopt.torch.distill.plugins.megatron.parallel_state") as mock_ps,
            patch("megatron.bridge.models.distillation_provider.unwrap_model", side_effect=lambda m: m),
            # mtd.convert mutates in place -> return the same submodule object.
            patch("megatron.bridge.models.distillation_provider.mtd.convert", side_effect=lambda m, mode: m),
        ):
            mock_ps.is_pipeline_first_stage.return_value = True
            mock_ps.is_pipeline_last_stage.return_value = True
            result = student._convert_hook([student_full])

        # The full model is retained for export; only the distilled submodule is returned as the model.
        assert student.full_model is student_full
        assert result[0] is student_full.language_model

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
        student = GPTModelProvider(
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

            assert cfg_dict["_target_"] == "megatron.bridge.models.gpt_provider.GPTModelProvider"

            # Verify GPTModelProvider-level fields are present and match the student config.
            assert cfg_dict["vocab_size"] == student.vocab_size
            assert cfg_dict["share_embeddings_and_output_weights"] == student.share_embeddings_and_output_weights
        finally:
            # Restore original bases since it was modified globally for the entire class
            DistillationProvider.__bases__ = original_bases

    def test_convert_hybrid_provider_to_distillation_provider(self):
        """Test that HybridModelProvider can be converted to DistillationProvider."""
        from megatron.bridge.models.distillation_provider import DistillationProvider

        teacher = HybridModelProvider(
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
        student_base = HybridModelProvider(
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
            assert isinstance(converted, HybridModelProvider)
            assert converted.teacher is teacher
            assert converted.num_layers == 24
            assert converted.hidden_size == 2048
            assert converted.num_attention_heads == 16
        finally:
            # Restore original bases since it was modified globally for the entire class
            DistillationProvider.__bases__ = original_bases
