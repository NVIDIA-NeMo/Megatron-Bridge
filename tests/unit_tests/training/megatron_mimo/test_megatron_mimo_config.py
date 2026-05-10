# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import pytest
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


def test_module_parallelism_finalize_computes_dp():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    parallelism.finalize(world_size=16)
    assert parallelism.data_parallel_size == 4
    assert parallelism.total_model_parallel_size == 4
    assert parallelism.total_ranks == 16


def test_module_parallelism_finalize_invalid_world_size():
    parallelism = ModuleParallelismConfig(tensor_model_parallel_size=3, pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="world_size .* not divisible"):
        parallelism.finalize(world_size=10)


def test_megatron_mimo_heterogeneous_rank_offset_overlap():
    """Test that overlapping rank ranges are detected in heterogeneous deployment."""
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=2),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    with pytest.raises(ValueError, match="overlap"):
        megatron_mimo_parallelism_config.finalize(world_size=6)


def test_megatron_mimo_heterogeneous_valid_contiguous():
    """Test that contiguous rank allocation works correctly."""
    # Note: encoder DP must be >= LLM DP for embedding alignment
    module_parallelisms = {
        "encoder": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=4, rank_offset=0),
        "language": ModuleParallelismConfig(tensor_model_parallel_size=1, data_parallel_size=2, rank_offset=4),
    }
    megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms=module_parallelisms,
    )
    # No gaps, no overlap, encoder DP >= LLM DP - should pass
    megatron_mimo_parallelism_config.finalize(world_size=6)
    assert megatron_mimo_parallelism_config.total_world_size == 6


def test_megatron_mimo_rejects_single_language_module_pp1():
    """Single-module ``{"language": ...}`` with otherwise-valid colocated geometry
    must fail with the required-modules error (not silently pass).

    The required-modules validator should run before colocated placement checks
    so malformed single-module configs surface the right reason.
    """
    module_parallelisms = {
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="at least one modality module"):
        cfg.finalize(world_size=2)


def test_megatron_mimo_rejects_no_language_module():
    """Configs without the language module are rejected by required-modules
    before any geometry-shape validator runs."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match=f"'{MIMO_LANGUAGE_MODULE_KEY}' must be in module_parallelisms"):
        cfg.finalize(world_size=2)


# ── Colocated heterogeneous TP/DP validator ───────────────────────────────────
#
# These tests cover ``_validate_colocated_constraints``, the dedicated
# colocated-mode validator for heterogeneous TP/DP. They
# verify what's rejected today and pin the eventually-allowed shapes via xfail
# so that a follow-up PR dropping the asymmetric-DP/TP guards flips them green
# automatically.


def test_colocated_validator_accepts_equal_dp_tp_pp1_cp1():
    """Symmetric colocated stays accepted (regression baseline)."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=2)
    assert cfg.total_world_size == 2

def test_colocated_validator_accepts_language_pp_gt_1():
    """Language PP>1 in colocated is accepted by the colocated language-PP adapter.

    Encoder PP remains 1. Data-parallel sizes are chosen so both modules span
    the same four colocated ranks.
    """
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=4,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=4)
    assert cfg.total_world_size == 4


def test_colocated_validator_rejects_encoder_pp_gt_1():
    """Encoder PP>1 in colocated is still rejected."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="encoder PP=1"):
        cfg.finalize(world_size=4)


def test_colocated_validator_rejects_language_pp_with_multiple_modality_modules():
    """Language PP>1 v1 supports exactly one colocated modality module."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=4,
            rank_offset=0,
        ),
        "audio": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=4,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="exactly one modality module"):
        cfg.finalize(world_size=4)


def test_colocated_validator_accepts_language_cp_gt_1():
    """Language CP>1 is accepted for the colocated CP-only path."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=2)
    assert cfg.total_world_size == 2


def test_colocated_validator_rejects_encoder_cp_gt_1():
    """Encoder CP>1 in colocated is still rejected."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="encoder CP=1"):
        cfg.finalize(world_size=2)


def test_colocated_validator_rejects_language_cp_with_multiple_modality_modules():
    """Language CP>1 v1 shares the single-modality gate with language PP>1."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "audio": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="PP>1 or CP>1"):
        cfg.finalize(world_size=2)


def test_colocated_validator_rejects_combined_language_pp_and_cp():
    """CP+PP is rejected until MCore language-PP CP label/loss-mask sharding lands."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=4,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="does not support combining language PP>1 with language CP>1"):
        cfg.finalize(world_size=4)


def test_colocated_validator_rejects_etp_gt_1():
    """ETP>1 in colocated is rejected — EP/ETP coverage is a separate task."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=2,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="ETP=1"):
        cfg.finalize(world_size=2)


# Heterogeneous TP/DP shapes are supported in colocated mode.


def test_colocated_validator_accepts_fan_in_2gpu():
    """Target heterogeneous shape: enc(TP=1,DP=2) × llm(TP=2,DP=1) on 2 GPUs.

    This locks the fan-in geometry used by colocated heterogeneous TP/DP.
    """
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=2)
    assert cfg.total_world_size == 2
