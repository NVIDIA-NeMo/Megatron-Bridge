# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Precedence tests for the perf-script override pipeline.

Expected precedence (highest wins): argparse > Hydra CLI > workload base > recipe default.

Exercises the real 4-step override pipeline from run_script.py:main:
    1. get_perf_optimized_recipe()      # recipe default + workload base
    2. set_cli_overrides(recipe, cli)   # Hydra/OmegaConf merge
    3. set_user_overrides(recipe, args) # argparse Namespace
    4. set_post_overrides(recipe, ...)  # post-processing (GBS auto-scale etc.)

Focus on recompute_* and global_batch_size — the two fields where we observed
silent wipes after upstream PR #3470 (2026-04-23).
"""

import sys
from pathlib import Path


SCRIPTS_PERF_PATH = Path(__file__).parents[3] / "scripts" / "performance"
sys.path.insert(0, str(SCRIPTS_PERF_PATH))


def _build_base_args(**overrides):
    """Return a fully-populated argparse.Namespace using the real parser defaults.

    Tests override only the fields they care about; every other field takes its
    argparse-default (None for optional flags), matching what a real user CLI
    invocation with only the required args would produce.
    """
    from argument_parser import parse_cli_args

    parser = parse_cli_args()
    # minimum required: -m, -mr, -g, -ng
    argv = [
        "-m",
        "deepseek",
        "-mr",
        "deepseek_v3",
        "--task",
        "pretrain",
        "-g",
        "gb200",
        "-ng",
        "64",
    ]
    args, _ = parser.parse_known_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _fresh_recipe():
    from utils.utils import get_perf_optimized_recipe

    return get_perf_optimized_recipe(
        model_family_name="deepseek",
        model_recipe_name="deepseek_v3",
        train_task="pretrain",
        gpu="gb200",
        compute_dtype="nvfp4",
        mock=True,
        config_variant="v1",
    )


def _apply(recipe, cli_overrides=None, args_overrides=None, run_post=True, num_gpus=64):
    from utils.overrides import set_cli_overrides, set_post_overrides, set_user_overrides

    cli_overrides = cli_overrides or []
    args = _build_base_args(**(args_overrides or {}))
    recipe = set_cli_overrides(recipe, cli_overrides)
    recipe = set_user_overrides(recipe, args)
    if run_post:
        recipe = set_post_overrides(
            recipe,
            model_family_name="deepseek",
            model_recipe_name="deepseek_v3",
            gpu="gb200",
            num_gpus=num_gpus,
            compute_dtype="nvfp4",
            task="pretrain",
            user_gbs=args.global_batch_size,
            config_variant="v1",
        )
    return recipe


# ---------------------------------------------------------------------------
# Recompute precedence
# ---------------------------------------------------------------------------


class TestRecipeDefault:
    def test_recipe_default_disables_recompute(self):
        """DSv3 recipe default must be 'no recompute' (granularity=None).

        Rationale: MCore's transformer_config post-init fills
        recompute_modules=None with ['core_attn'] when granularity is set.
        Leaving granularity='selective' with modules=None would therefore
        silently turn on core_attn recompute across all layers for any run
        that doesn't explicitly configure recompute — defeating #3470's
        stated 'no recompute default' intent."""
        from megatron.bridge.recipes.deepseek.deepseek_v3 import deepseek_v3_pretrain_config

        cfg = deepseek_v3_pretrain_config()
        assert cfg.model.recompute_granularity is None
        assert cfg.model.recompute_num_layers is None


class TestRecomputePrecedence:
    def test_A_workload_default_survives_when_nothing_else_set(self):
        """Workload base for GB200 NVFP4 V1 has recompute_modules=['mlp']. With
        no Hydra and no argparse override, that value must reach the final
        recipe."""
        recipe = _fresh_recipe()
        recipe = _apply(recipe, run_post=False)
        assert recipe.model.recompute_modules == ["mlp"]
        assert recipe.model.recompute_granularity == "selective"

    def test_B_hydra_overrides_workload(self):
        """Hydra `model.recompute_modules=[mlp,mla_up_proj]` must replace the
        workload's ['mlp']."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            cli_overrides=["model.recompute_modules=[mlp,mla_up_proj]"],
            run_post=False,
        )
        assert recipe.model.recompute_modules == ["mlp", "mla_up_proj"]
        assert recipe.model.recompute_granularity == "selective"

    def test_C_argparse_overrides_hydra(self):
        """argparse `--recompute_modules foo,bar` must beat a Hydra override."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            cli_overrides=["model.recompute_modules=[mla_up_proj]"],
            args_overrides={"recompute_modules": ["foo", "bar"]},
            run_post=False,
        )
        assert recipe.model.recompute_modules == ["foo", "bar"]
        assert recipe.model.recompute_granularity == "selective"

    def test_D_explicit_disable_via_hydra(self):
        """Explicit Hydra disable (used by GB300 dsv3.sh pattern) must produce
        granularity=None and an empty module list."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            cli_overrides=[
                "model.recompute_granularity=null",
                "model.recompute_modules=[]",
            ],
            run_post=False,
        )
        assert recipe.model.recompute_granularity is None
        assert recipe.model.recompute_modules == []

    def test_E_argparse_recompute_num_layers_switches_to_full(self):
        """argparse `--recompute_num_layers 5` must switch to full-block
        recompute."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            args_overrides={"recompute_num_layers": 5},
            run_post=False,
        )
        assert recipe.model.recompute_granularity == "full"
        assert recipe.model.recompute_method == "block"
        assert recipe.model.recompute_num_layers == 5


# ---------------------------------------------------------------------------
# GBS auto-rescale precedence
# ---------------------------------------------------------------------------


class TestGbsPrecedence:
    def test_F_autoscale_fires_when_no_one_sets(self):
        """When neither Hydra nor argparse sets GBS and num_gpus differs from
        the workload default, set_post_overrides should rescale. This is the
        existing intentional feature; verify it still works after the fix."""
        recipe = _fresh_recipe()
        # Workload default for GB200 V1 is GBS=2048 at 256 GPUs. At 64 GPUs,
        # gbs_scaling_factor * 64 should be applied.
        recipe = _apply(recipe, num_gpus=64)
        assert recipe.train.global_batch_size != 2048, "GBS auto-scale did not fire for num_gpus=64 (default=256)"

    def test_G_hydra_overrides_autoscale(self):
        """Hydra `train.global_batch_size=128` must survive set_post_overrides
        even though num_gpus differs from the workload default."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            cli_overrides=["train.global_batch_size=128"],
            num_gpus=64,
        )
        assert recipe.train.global_batch_size == 128

    def test_H_argparse_overrides_autoscale(self):
        """argparse `-gb 256` must survive set_post_overrides."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            args_overrides={"global_batch_size": 256},
            num_gpus=64,
        )
        assert recipe.train.global_batch_size == 256


# ---------------------------------------------------------------------------
# Consistency check: full override chain produces expected final state
# ---------------------------------------------------------------------------


class TestFullChainSanity:
    def test_I_combined_overrides_end_to_end(self):
        """Simulate the GB200 proxy script: Hydra for recompute modules +
        pipeline/cuda_graph; argparse only for required fields. After the
        fix, the Hydra recompute override must survive the full 4-step chain."""
        recipe = _fresh_recipe()
        recipe = _apply(
            recipe,
            cli_overrides=[
                "model.num_layers=16",
                "model.pipeline_model_parallel_size=1",
                "model.virtual_pipeline_model_parallel_size=null",
                "model.pipeline_model_parallel_layout=null",
                "model.recompute_modules=[mlp,mla_up_proj]",
                "model.cuda_graph_impl=none",
                "model.cuda_graph_scope=[]",
                "train.micro_batch_size=2",
                "train.global_batch_size=128",
            ],
            num_gpus=64,
        )
        assert recipe.model.num_layers == 16
        assert recipe.model.pipeline_model_parallel_size == 1
        assert recipe.model.virtual_pipeline_model_parallel_size is None
        assert recipe.model.recompute_modules == ["mlp", "mla_up_proj"]
        assert recipe.model.recompute_granularity == "selective"
        assert recipe.model.cuda_graph_impl == "none"
        assert recipe.model.cuda_graph_scope == []
        assert recipe.train.micro_batch_size == 2
        assert recipe.train.global_batch_size == 128
