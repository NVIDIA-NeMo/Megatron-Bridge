"""
Compare the recipe's ConfigContainer vs manually built one to find the difference.
"""

import torch.distributed as dist

from megatron.bridge.recipes.llama import llama32_1b_pretrain_config as pretrain_config


def print_config_structure(cfg, name):
    """Print key aspects of a config."""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")

    print("\nModel:")
    print(f"  Class: {cfg.model.__class__}")
    print(f"  ID: {id(cfg.model)}")

    print("\nOptimizer:")
    print(f"  Class: {cfg.optimizer.__class__}")
    print(f"  use_distributed_optimizer: {cfg.optimizer.use_distributed_optimizer}")

    print("\nDDP:")
    print(f"  Class: {cfg.ddp.__class__}")
    print(f"  use_megatron_fsdp: {cfg.ddp.use_megatron_fsdp}")
    print(f"  data_parallel_sharding_strategy: {cfg.ddp.data_parallel_sharding_strategy}")

    print("\nDataset:")
    print(f"  Class: {cfg.dataset.__class__}")
    print(f"  Type: {type(cfg.dataset).__name__}")

    print("\nCheckpoint:")
    print(f"  ckpt_format: {cfg.checkpoint.ckpt_format}")
    print(f"  save: {cfg.checkpoint.save}")
    print(f"  load: {cfg.checkpoint.load}")


if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if dist.get_rank() == 0:
        # Create recipe config
        print("\n" + "#" * 80)
        print("# CREATING RECIPE CONFIG")
        print("#" * 80)
        cfg_recipe = pretrain_config(seq_length=1024, use_megatron_fsdp=True)

        # Apply FSDP modifications
        cfg_recipe.model.gradient_accumulation_fusion = False
        cfg_recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        cfg_recipe.ddp.average_in_collective = False
        cfg_recipe.checkpoint.ckpt_format = "fsdp_dtensor"
        cfg_recipe.checkpoint.load = None

        print_config_structure(cfg_recipe, "RECIPE CONFIG (fails)")

        # Check for any special attributes or state
        print("\n\nChecking recipe config for special attributes:")
        for attr in dir(cfg_recipe):
            if not attr.startswith("_") and attr not in [
                "model",
                "optimizer",
                "ddp",
                "dataset",
                "checkpoint",
                "train",
                "scheduler",
                "logger",
                "tokenizer",
                "rng",
                "dist",
                "ft",
                "straggler",
                "nvrx_straggler",
                "profiling",
                "peft",
                "comm_overlap",
                "mixed_precision",
                "tensor_inspect",
                "inprocess_restart",
                "rerun_state_machine",
            ]:
                val = getattr(cfg_recipe, attr, None)
                if not callable(val):
                    print(f"  {attr}: {val}")

        # Check if optimizer or dataset have any cached references to model params
        print("\n\nChecking for cached state in sub-configs:")
        print(f"  optimizer.__dict__ keys: {list(cfg_recipe.optimizer.__dict__.keys())}")
        print(f"  ddp.__dict__ keys: {list(cfg_recipe.ddp.__dict__.keys())}")
        print(f"  dataset.__dict__ keys: {list(cfg_recipe.dataset.__dict__.keys())}")

        print("\n\nDataset type comparison:")
        print(f"  Recipe dataset type: {type(cfg_recipe.dataset)}")
        print(f"  Recipe dataset class name: {cfg_recipe.dataset.__class__.__name__}")
        print(f"  Is it GPTDatasetConfig? {cfg_recipe.dataset.__class__.__name__ == 'GPTDatasetConfig'}")
        print(f"  Is it MockGPTDatasetConfig? {cfg_recipe.dataset.__class__.__name__ == 'MockGPTDatasetConfig'}")

        print("\n\n" + "=" * 80)
        print("KEY FINDING TO INVESTIGATE:")
        print("=" * 80)
        print("The working test uses MockGPTDatasetConfig (mock data)")
        print("The recipe uses GPTDatasetConfig (real data with blend/split)")
        print("Could the dataset type be causing issues with FSDP?")
