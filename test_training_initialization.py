#!/usr/bin/env python3
"""
Sanity test script for training initialization logic.

This script mimics the complete training setup process without actually training,
testing that base model initialization, pretrained checkpoint loading, and PEFT
application work as expected.

Usage:
    python test_training_initialization.py
"""

import os
import sys
import time

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

# # Add the src directory to the path so we can import megatron modules
# sys.path.insert(0, str(Path(__file__).parent / "src"))
from megatron.hub.models import get_base_model, get_distributed_model
from megatron.hub.peft.lora import LoRA
from megatron.hub.training.checkpointing import checkpoint_exists, load_checkpoint
from megatron.hub.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.hub.training.initialize import initialize_megatron
from megatron.hub.training.state import GlobalState
from megatron.hub.utils.common_utils import print_rank_0
from megatron.hub.utils.config_utils import InstantiationMode
from megatron.hub.utils.log_utils import setup_logging


def create_test_config(pretrained_checkpoint_path: str) -> ConfigContainer:
    """Create a test configuration for sanity testing.

    Loads the original configuration from the checkpoint YAML file and then
    overrides the necessary pieces for testing. If YAML is not found, falls
    back to a default Llama3-8B configuration.

    Args:
        pretrained_checkpoint_path: Path to the pretrained checkpoint directory

    Returns:
        ConfigContainer with test configuration
    """
    # Try to load the original configuration from the checkpoint
    config_yaml_path = os.path.join(pretrained_checkpoint_path, "iter_0000000", "run_config.yaml")

    if os.path.exists(config_yaml_path):
        print_rank_0(f"Loading original configuration from: {config_yaml_path}")

        try:
            # Load the configuration using lenient mode to handle any missing fields
            cfg = ConfigContainer.from_yaml(config_yaml_path, mode=InstantiationMode.LENIENT)
            print_rank_0("✅ Configuration loaded successfully from YAML")
        except Exception as e:
            print_rank_0(f"⚠️  Failed to load YAML config: {e}")
            print_rank_0("   Falling back to default Llama3-8B configuration")
            cfg = _create_fallback_config()
    else:
        print_rank_0(f"⚠️  Configuration YAML not found at: {config_yaml_path}")
        print_rank_0("   Using default Llama3-8B configuration")
        cfg = _create_fallback_config()

    print_rank_0(f"Configuration: {cfg}")

    # Override specific settings for testing
    print_rank_0("Overriding configuration for testing...")

    # PEFT configuration with LoRA
    peft_config = LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=8,  # Small rank for testing
        alpha=16,
        dropout=0.1,
    )
    cfg.peft = peft_config

    # Override training configuration for testing
    cfg.train = TrainingConfig(micro_batch_size=1, global_batch_size=1, train_iters=10, eval_iters=1, eval_interval=5)

    # Override checkpoint configuration
    cfg.checkpoint.save = None  # Don't save during testing
    cfg.checkpoint.load = None  # Will be set later if needed
    cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint_path
    cfg.checkpoint.finetune = True  # Important for PEFT
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.load_optim = False

    # Override model configuration for testing (single GPU)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.use_cpu_initialization = True
    cfg.model.cross_entropy_loss_fusion = False

    # Override distributed configuration for single GPU testing
    cfg.dist = DistributedInitConfig()

    # Set logger level for testing
    cfg.logger = LoggerConfig(log_interval=1)

    cfg.optimizer = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        use_distributed_optimizer=False,
        clip_grad=0.2,
        lr=4e-5,
        weight_decay=0.01,
        min_lr=0.1 * 4e-5,
    )
    cfg.scheduler = SchedulerConfig(
        start_weight_decay=0.01,
        end_weight_decay=0.01,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_decay_iters=100,
        lr_warmup_iters=int(0.03 * 100),
        lr_warmup_init=0.0,
        override_opt_param_scheduler=True,
    )
    cfg.ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        use_distributed_optimizer=True,
    )

    # Configure tokenizer appropriately for the checkpoint
    _configure_tokenizer_for_checkpoint(cfg, pretrained_checkpoint_path)

    print_rank_0("Configuration loaded and overridden for testing")

    return cfg


def _configure_tokenizer_for_checkpoint(cfg: ConfigContainer, checkpoint_path: str) -> None:
    """Configure the tokenizer appropriately based on the checkpoint structure.

    Args:
        cfg: Configuration container to modify
        checkpoint_path: Path to the checkpoint directory
    """
    from megatron.hub.training.config import TokenizerConfig

    # Check for HuggingFace assets directory (common in converted checkpoints)
    hf_assets_path = os.path.join(checkpoint_path, "hf_assets")

    # Check for tokenizer files in various locations
    tokenizer_json_path = os.path.join(checkpoint_path, "tokenizer.json")
    tokenizer_model_path = os.path.join(checkpoint_path, "tokenizer.model")
    hf_tokenizer_json = os.path.join(hf_assets_path, "tokenizer.json")

    if os.path.exists(hf_assets_path) and os.path.exists(hf_tokenizer_json):
        # HuggingFace assets directory found (preferred for converted checkpoints)
        print_rank_0(f"Found HuggingFace tokenizer assets at: {hf_assets_path}")
        cfg.tokenizer = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_assets_path,
            vocab_size=128256,
        )
    elif os.path.exists(tokenizer_json_path):
        # HuggingFace tokenizer files found in root
        print_rank_0(f"Found HuggingFace tokenizer at: {tokenizer_json_path}")
        cfg.tokenizer = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=checkpoint_path,
            vocab_size=128256,
        )
    elif os.path.exists(tokenizer_model_path):
        # SentencePiece tokenizer model found
        print_rank_0(f"Found SentencePiece tokenizer at: {tokenizer_model_path}")
        cfg.tokenizer = TokenizerConfig(
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model=tokenizer_model_path,
            vocab_size=128256,
        )
    else:
        # Fall back to HuggingFace model identifier for Llama-3
        print_rank_0("No local tokenizer found, using HuggingFace model identifier")
        cfg.tokenizer = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Meta-Llama-3-8B",
            vocab_size=128256,
        )


def _create_fallback_config() -> ConfigContainer:
    """Create a fallback configuration when YAML loading fails.

    Returns:
        ConfigContainer with default Llama3-8B configuration
    """
    from megatron.hub.models.llama import Llama3Config8B

    # Model configuration for Llama-3-8B
    model_config = Llama3Config8B(
        vocab_size=128256,  # Llama-3 vocab size
        make_vocab_size_divisible_by=64,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        fp16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        params_dtype=torch.float16,
    )

    # Basic training configuration
    train_config = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=1,
        train_iters=10,  # Small for testing
        eval_iters=1,
        eval_interval=5,
    )

    # Checkpoint configuration
    checkpoint_config = CheckpointConfig(
        save=None,  # Don't save during testing
        load=None,  # Will be set later if needed
        pretrained_checkpoint=None,  # Will be set later
        finetune=True,  # Important for PEFT
        save_optim=False,
        load_optim=False,
    )

    # Tokenizer configuration for Llama (HuggingFace)
    # Try to auto-detect tokenizer from checkpoint, fall back to HF model identifier
    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=None,  # Will auto-detect from checkpoint or use HF identifier
        vocab_size=128256,
    )

    # Other required configurations
    rng_config = RNGConfig(seed=1234)
    dist_config = DistributedInitConfig()
    logger_config = LoggerConfig(log_interval=1)
    scheduler_config = SchedulerConfig(lr_decay_style="constant")

    # Dummy optimizer config (required but not used in this test)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-4,
        weight_decay=0.01,
    )

    # Dummy dataset config (required but not used in this test)
    dataset_config = FinetuningDatasetConfig(
        dataset_root="dummy",
        seq_length=2048,
    )

    # Dummy DDP config
    ddp_config = DistributedDataParallelConfig()

    return ConfigContainer(
        model=model_config,
        peft=None,  # Will be set later
        train=train_config,
        checkpoint=checkpoint_config,
        tokenizer=tokenizer_config,
        rng=rng_config,
        dist=dist_config,
        logger=logger_config,
        scheduler=scheduler_config,
        optimizer=optimizer_config,
        dataset=dataset_config,
        ddp=ddp_config,
    )


def test_base_model_creation(cfg: ConfigContainer) -> None:
    """Test base model creation."""
    print_rank_0("\n" + "=" * 60)
    print_rank_0("TESTING: Base Model Creation")
    print_rank_0("=" * 60)

    try:
        start_time = time.time()
        base_model = get_base_model(cfg.model)

        print_rank_0(f"✅ Base model created successfully in {time.time() - start_time:.2f}s")
        print_rank_0(f"   Model type: {type(base_model)}")
        print_rank_0(f"   Number of model chunks: {len(base_model) if isinstance(base_model, list) else 1}")

        # Print model statistics
        if isinstance(base_model, list):
            model_to_analyze = base_model[0]
        else:
            model_to_analyze = base_model

        total_params = sum(p.numel() for p in model_to_analyze.parameters())
        print_rank_0(f"   Total parameters: {total_params:,}")

        return base_model
    except Exception as e:
        print_rank_0(f"❌ Base model creation failed: {e}")
        raise


def test_pretrained_checkpoint_loading(cfg: ConfigContainer, base_model) -> None:
    """Test loading pretrained checkpoint into base model."""
    print_rank_0("\n" + "=" * 60)
    print_rank_0("TESTING: Pretrained Checkpoint Loading")
    print_rank_0("=" * 60)

    if not cfg.checkpoint.pretrained_checkpoint:
        print_rank_0("⚠️  No pretrained checkpoint path provided, skipping test")
        return base_model

    if not checkpoint_exists(cfg.checkpoint.pretrained_checkpoint):
        print_rank_0(f"⚠️  Pretrained checkpoint not found at {cfg.checkpoint.pretrained_checkpoint}")
        print_rank_0("   Skipping pretrained checkpoint loading test")
        return base_model

    try:
        start_time = time.time()

        # Create a minimal state object for checkpoint loading
        state = GlobalState()
        state.cfg = cfg

        # Load the pretrained checkpoint
        load_checkpoint(
            state,
            base_model,
            None,  # No optimizer
            None,  # No scheduler
            strict=True,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )

        print_rank_0(f"✅ Pretrained checkpoint loaded successfully in {time.time() - start_time:.2f}s")
        print_rank_0(f"   Checkpoint path: {cfg.checkpoint.pretrained_checkpoint}")

        return base_model
    except Exception as e:
        print_rank_0(f"❌ Pretrained checkpoint loading failed: {e}")
        raise


def test_peft_application(cfg: ConfigContainer, base_model) -> None:
    """Test PEFT application to the base model."""
    print_rank_0("\n" + "=" * 60)
    print_rank_0("TESTING: PEFT Application")
    print_rank_0("=" * 60)

    if cfg.peft is None:
        print_rank_0("⚠️  No PEFT configuration provided, skipping test")
        return base_model

    try:
        start_time = time.time()

        # Apply PEFT transformation
        peft_model = cfg.peft(base_model, training=True)

        print_rank_0(f"✅ PEFT applied successfully in {time.time() - start_time:.2f}s")
        print_rank_0(f"   PEFT type: {type(cfg.peft).__name__}")

        # Analyze PEFT statistics
        model_to_analyze = peft_model[0] if isinstance(peft_model, list) else peft_model
        total_params = sum(p.numel() for p in model_to_analyze.parameters())
        trainable_params = sum(p.numel() for p in model_to_analyze.parameters() if p.requires_grad)

        print_rank_0(f"   Total parameters: {total_params:,}")
        print_rank_0(f"   Trainable parameters: {trainable_params:,}")
        print_rank_0(f"   Trainable percentage: {100 * trainable_params / total_params:.2f}%")

        # Set up parameters for checkpointing
        cfg.peft.set_params_to_save(peft_model)
        print_rank_0(f"   Parameters marked for saving: {len(cfg.peft.params_to_save)}")

        return peft_model
    except Exception as e:
        print_rank_0(f"❌ PEFT application failed: {e}")
        raise


def test_distributed_model_setup(cfg: ConfigContainer, model) -> None:
    """Test distributed model setup (DDP/FSDP wrapping)."""
    print_rank_0("\n" + "=" * 60)
    print_rank_0("TESTING: Distributed Model Setup")
    print_rank_0("=" * 60)

    try:
        start_time = time.time()

        # Apply distributed wrappers
        distributed_model = get_distributed_model(
            model,
            cfg.model,
            cfg.ddp,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=False,
            wrap_with_ddp=True,
            data_parallel_random_init=cfg.rng.data_parallel_random_init,
        )

        print_rank_0(f"✅ Distributed model setup completed in {time.time() - start_time:.2f}s")
        print_rank_0(f"   Model type after wrapping: {type(distributed_model)}")

        if isinstance(distributed_model, list):
            print_rank_0(f"   Number of model chunks: {len(distributed_model)}")
            for i, chunk in enumerate(distributed_model):
                print_rank_0(f"   Chunk {i} type: {type(chunk)}")

        return distributed_model
    except Exception as e:
        print_rank_0(f"❌ Distributed model setup failed: {e}")
        raise


def test_model_forward_pass(cfg: ConfigContainer, model) -> None:
    """Test a forward pass through the model."""
    print_rank_0("\n" + "=" * 60)
    print_rank_0("TESTING: Model Forward Pass")
    print_rank_0("=" * 60)

    try:
        start_time = time.time()

        # Create dummy input
        batch_size = cfg.train.micro_batch_size
        seq_length = 128  # Short sequence for testing
        vocab_size = cfg.model.vocab_size

        # Get model device
        model_chunks = model if isinstance(model, list) else [model]
        model_device = next(model_chunks[0].parameters()).device

        # Create input tensors in the format expected by Megatron models
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=model_device)
        position_ids = (
            torch.arange(seq_length, dtype=torch.long, device=model_device).unsqueeze(0).expand(batch_size, -1)
        )

        # Create 4D causal attention mask [batch_size, 1, seq_length, seq_length]
        # True values are masked out (don't attend), False values attend
        attention_mask = torch.tril(torch.ones(seq_length, seq_length, device=model_device)) < 0.5
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Prepare forward arguments
        forward_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

        # Set model to eval mode
        for chunk in model_chunks:
            chunk.eval()

        with torch.no_grad():
            # Forward pass through each model chunk
            output = None
            for i, chunk in enumerate(model_chunks):
                output = chunk(**forward_args)
                print_rank_0(f"   Chunk {i} forward pass completed")

                # For pipeline parallel, output of one chunk becomes input to next
                if i < len(model_chunks) - 1 and output is not None:
                    # Update input for next chunk (this is simplified - real PP is more complex)
                    if isinstance(output, tuple):
                        forward_args["input_ids"] = output[0]
                    else:
                        forward_args["input_ids"] = output

        print_rank_0(f"✅ Forward pass completed successfully in {time.time() - start_time:.2f}s")
        print_rank_0(f"   Input shape: {input_ids.shape}")
        print_rank_0(f"   Position IDs shape: {position_ids.shape}")
        print_rank_0(f"   Attention mask shape: {attention_mask.shape}")

        if isinstance(output, tuple):
            print_rank_0(f"   Output is tuple with {len(output)} elements")
            if len(output) > 0 and hasattr(output[0], "shape"):
                print_rank_0(f"   First output shape: {output[0].shape}")
        elif hasattr(output, "shape"):
            print_rank_0(f"   Output shape: {output.shape}")
        else:
            print_rank_0(f"   Output type: {type(output)}")

    except Exception as e:
        print_rank_0(f"❌ Forward pass failed: {e}")
        raise


def main():
    """Main function to run all sanity tests."""
    pretrained_checkpoint_path = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/"

    print_rank_0("=" * 80)
    print_rank_0("MEGATRON-LM TRAINING INITIALIZATION SANITY TEST")
    print_rank_0("=" * 80)
    print_rank_0(f"Pretrained checkpoint path: {pretrained_checkpoint_path}")
    print_rank_0(f"PyTorch version: {torch.__version__}")
    print_rank_0(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print_rank_0(f"CUDA device count: {torch.cuda.device_count()}")
        print_rank_0(f"Current CUDA device: {torch.cuda.current_device()}")

    try:
        # Create test configuration
        cfg = create_test_config(pretrained_checkpoint_path)
        cfg.validate()

        # Set up logging
        setup_logging(
            logging_level=cfg.logger.logging_level,
            filter_warning=cfg.logger.filter_warnings,
        )

        # Initialize Megatron
        print_rank_0("\n" + "=" * 60)
        print_rank_0("TESTING: Megatron Initialization")
        print_rank_0("=" * 60)

        start_time = time.time()
        initialize_megatron(cfg=cfg, allow_no_cuda=False)
        print_rank_0(f"✅ Megatron initialized successfully in {time.time() - start_time:.2f}s")

        # Test 1: Base model creation
        base_model = test_base_model_creation(cfg)

        # Test 2: Pretrained checkpoint loading
        base_model = test_pretrained_checkpoint_loading(cfg, base_model)

        # Test 3: PEFT application
        peft_model = test_peft_application(cfg, base_model)

        # Test 4: Distributed model setup
        distributed_model = test_distributed_model_setup(cfg, peft_model)

        # Test 5: Forward pass
        test_model_forward_pass(cfg, distributed_model)

        # Final success message
        print_rank_0("\n" + "=" * 80)
        print_rank_0("🎉 ALL TESTS PASSED! Training initialization pipeline works correctly.")
        print_rank_0("=" * 80)
        print_rank_0("Summary:")
        print_rank_0("  ✅ Base model creation")
        print_rank_0("  ✅ Pretrained checkpoint loading")
        print_rank_0("  ✅ PEFT application")
        print_rank_0("  ✅ Distributed model setup")
        print_rank_0("  ✅ Model forward pass")
        print_rank_0("\n🚀 Ready to start actual training!")

    except Exception as e:
        print_rank_0(f"\n❌ TEST FAILED: {e}")
        import traceback

        print_rank_0(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
