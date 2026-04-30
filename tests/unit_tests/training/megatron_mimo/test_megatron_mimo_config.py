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
    """Single-module ``{"language": ...}`` with otherwise-valid stage-2 geometry
    must fail with the required-modules error (not silently pass).

    Pre-fix this slipped through: only-language passed `_validate_module_placement`
    (one module → no overlap), and stage-2 `_is_colocated()` returned True
    because the placement set was a singleton, but PP=1/CP=1/ETP=1 didn't
    trigger any rejection. The required-modules validator catches this
    explicitly with the right reason.
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


def test_megatron_mimo_rejects_single_language_module_pp_gt1():
    """Single-module language config with PP>1 must surface the required-modules
    error, NOT the colocated stage-2 PP=1 error.

    Pre-fix the only-language config tripped `_validate_colocated_stage2_constraints`
    (because `_is_colocated()` returned True for a singleton placement set) and
    raised "PP=1" — a true statement about the geometry but the wrong reason
    for the failure. The user needs to add a modality module, not lower PP.
    """
    module_parallelisms = {
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=1,
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


# ── Stage-2 colocated heterogeneous TP/DP validator (Step 1 of plan) ──────────
#
# These tests cover ``_validate_colocated_stage2_constraints``, the dedicated
# colocated-mode validator added for the heterogeneous TP/DP rollout. They
# verify what's rejected today and pin the eventually-allowed shapes via xfail
# so that a follow-up PR dropping the asymmetric-DP/TP guards flips them green
# automatically.


def test_colocated_stage2_accepts_equal_dp_tp_pp1_cp1():
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


def test_colocated_stage2_accepts_language_pp_gt_1():
    """Language PP>1 in colocated is accepted by the three-phase schedule scope.

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


def test_colocated_stage2_rejects_encoder_pp_gt_1():
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


def test_colocated_stage2_rejects_language_pp_with_multiple_modality_modules():
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


def test_colocated_stage2_rejects_cp_gt_1():
    """CP>1 in colocated is rejected — asymmetric CP is a separate task."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            data_parallel_size=1,
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
    with pytest.raises(ValueError, match="CP=1"):
        cfg.finalize(world_size=2)


def test_colocated_stage2_rejects_etp_gt_1():
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


def test_colocated_stage2_does_not_run_on_non_colocated():
    """Stage-2 colocated checks must NOT fire for disjoint (non-colocated) layouts.

    Non-colocated supports PP>1 / CP>1 today and must keep doing so.
    """
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=2,
            data_parallel_size=1,
            rank_offset=2,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    # Non-colocated (disjoint ranges) — stage-2 check is a no-op; no PP=1 error.
    cfg.finalize(world_size=4)
    assert cfg.total_world_size == 4


def test_colocated_stage2_rejects_unequal_total_ranks_pure_dp():
    """Pure asymmetric DP `enc(TP=1,DP=2) × llm(TP=1,DP=1)` is not a valid colocated
    layout because total_ranks differ (2 vs 1).

    Caught by ``_validate_module_placement`` (partial overlap), before stage-2.
    Pinned here to make the rejection-path explicit and document why this
    "DP-only" 2-GPU shape is not a valid functional smoke target.
    """
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            data_parallel_size=1,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    with pytest.raises(ValueError, match="partial overlap"):
        cfg.finalize(world_size=2)


# ── Heterogeneous TP/DP shapes — now accepted under the Step-3 RNG plumbing ──


def test_colocated_stage2_accepts_fan_in_2gpu():
    """Target heterogeneous shape: enc(TP=1,DP=2) × llm(TP=2,DP=1) on 2 GPUs.

    Was xfail until the asymmetric-TP/DP short-term guards dropped (alongside
    the Step 3 module-scoped CUDA RNG plumbing in MimoModel + provider).
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


def test_colocated_stage2_accepts_fan_in_4gpu():
    """Target shape: enc(TP=1,DP=4) × llm(TP=2,DP=2) on 4 GPUs."""
    module_parallelisms = {
        "vision": ModuleParallelismConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=4,
            rank_offset=0,
        ),
        "language": ModuleParallelismConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            data_parallel_size=2,
            rank_offset=0,
        ),
    }
    cfg = MegatronMIMOParallelismConfig(module_parallelisms=module_parallelisms)
    cfg.finalize(world_size=4)
    assert cfg.total_world_size == 4
