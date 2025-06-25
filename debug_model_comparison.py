#!/usr/bin/env python3
"""
Debug script to compare NeMo and NeMo-LM models with PEFT applied.
This script will help identify why gradient norms differ (0.8 vs 3.36).
"""

import logging
import os

import torch
import torch.distributed as dist


# Set up environment
os.environ["NEMO_HOME"] = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache"
os.environ["NEMO_MODELS_CACHE"] = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache/models"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard vocabulary size for both models
VOCAB_SIZE = 32000


def create_custom_tokenizer():
    """Create a custom tokenizer with the standardized vocabulary size."""
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

    class CustomTokenizer(TokenizerSpec):
        def __init__(self, vocab_size=VOCAB_SIZE):
            super().__init__()
            self.vocab_size = vocab_size

        def text_to_tokens(self, text):
            # Mock implementation - not used in our case
            return []

        def tokens_to_text(self, tokens):
            # Mock implementation - not used in our case
            return ""

        def text_to_ids(self, text):
            # Mock implementation - not used in our case
            return []

        def ids_to_text(self, ids):
            # Mock implementation - not used in our case
            return ""

        def tokens_to_ids(self, tokens):
            # Mock implementation - not used in our case
            return []

        def ids_to_tokens(self, ids):
            # Mock implementation - not used in our case
            return []

    return CustomTokenizer(vocab_size=VOCAB_SIZE)


def initialize_nemo_lm_model():
    """Initialize NeMo-LM model with LoRA following the proper setup.py process."""
    logger.info("Initializing NeMo-LM model...")

    from megatron.core.distributed import DistributedDataParallelConfig as CoreDDPConfig
    from megatron.core.optimizer import OptimizerConfig

    from megatron.hub.models.llama import Llama3Config8B
    from megatron.hub.peft.lora import LoRA
    from megatron.hub.tokenizers.tokenizer import build_tokenizer
    from megatron.hub.training.config import (
        CheckpointConfig,
        ConfigContainer,
        LoggerConfig,
        MockGPTDatasetConfig,
        RNGConfig,
        SchedulerConfig,
        TokenizerConfig,
        TrainingConfig,
    )
    from megatron.hub.training.initialize import initialize_megatron
    from megatron.hub.training.setup import setup_model
    from megatron.hub.training.state import GlobalState

    # Step 1: Create model configuration matching the comparison script
    model_cfg = Llama3Config8B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=2048,
        bf16=True,
        params_dtype=torch.bfloat16,
        cross_entropy_loss_fusion=False,
        vocab_size=VOCAB_SIZE,  # Use standardized vocab size
    )

    # Step 2: Create LoRA PEFT configuration
    peft_config = LoRA(dim=8, alpha=16)

    # Step 3: Create checkpoint configuration
    pretrained_checkpoint = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B"
    checkpoint_path = pretrained_checkpoint

    checkpoint_cfg = CheckpointConfig(
        pretrained_checkpoint=checkpoint_path,
        save_interval=200,
        save="/tmp/debug_checkpoints",
        load=None,
    )

    # Step 4: Create DDP configuration
    ddp_cfg = CoreDDPConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,  # Simplified for debugging
        overlap_param_gather=False,
        average_in_collective=True,
        use_distributed_optimizer=False,  # Disabled for LoRA
    )

    # Step 4b: Create RNG configuration
    rng_cfg = RNGConfig(seed=1234, data_parallel_random_init=False)

    optimizer = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        lr=3e-3,
        weight_decay=0.01,
        min_lr=1e-6,
    )

    # Step 5: Create minimal config container
    cfg = ConfigContainer(
        model=model_cfg,
        peft=peft_config,
        checkpoint=checkpoint_cfg,
        ddp=ddp_cfg,
        rng=rng_cfg,
        train=TrainingConfig(
            train_iters=1,
            global_batch_size=2,
            micro_batch_size=1,
        ),
        optimizer=optimizer,
        dataset=MockGPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=2048,
            num_dataset_builder_threads=1,
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
        ),
        scheduler=SchedulerConfig(),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=VOCAB_SIZE,  # Use standardized vocab size
        ),
        logger=LoggerConfig(),
    )
    cfg.validate()

    # Step 6: Create global state (simplified)
    state = GlobalState()
    state.cfg = cfg

    # Step 6b: Initialize Megatron parallel state (CRITICAL)
    logger.info("Initializing Megatron parallel state...")
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
    )
    logger.info("Successfully initialized Megatron parallel state")

    # Step 7a: Build tokenizer (required before model creation)
    logger.info("Building tokenizer...")
    tokenizer = build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
    )
    if not cfg.model.vocab_size:
        cfg.model.vocab_size = tokenizer.vocab_size

    # Set tokenizer in dataset config
    if cfg.dataset is not None:
        cfg.dataset.tokenizer = tokenizer

    # Step 7b: Create base model (after tokenizer is built)
    logger.info("Creating base model...")
    model = setup_model(cfg, state)
    logger.info("NeMo-LM model initialization completed")
    return model[0]


def initialize_nemo_model():
    """Initialize NeMo model with LoRA using the official train API."""
    logger.info("Initializing NeMo model...")

    from megatron.core.distributed import DistributedDataParallelConfig
    from nemo import lightning as nl
    from nemo.collections.llm.api import validate
    from nemo.collections.llm.gpt.data.mock import MockDataModule
    from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
    from nemo.collections.llm.peft.lora import LoRA
    from nemo.lightning import AutoResume
    from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision

    # Create model config matching NeMo-LM
    model_config = Llama3Config8B(
        seq_length=2048,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        cross_entropy_loss_fusion=False,  # Required for LoRA compatibility
        vocab_size=VOCAB_SIZE,  # Use standardized vocab size
    )

    bf16_mixed = MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    # Create LoRA PEFT configuration (matching NeMo-LM settings)
    peft_config = LoRA(
        dim=8,
        alpha=16,
    )

    # Create MegatronStrategy for proper distributed setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        gradient_as_bucket_view=True,
        ckpt_async_save=False,
        ckpt_parallel_load=False,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=False,  # Disabled for LoRA
        ),
    )

    # Create Lightning Trainer
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy=strategy,
        plugins=[bf16_mixed],
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    # Create custom tokenizer with standardized vocab size
    custom_tokenizer = create_custom_tokenizer()

    # Create mock data module with custom tokenizer to ensure same vocab size
    data_module = MockDataModule(
        seq_length=2048,
        global_batch_size=2,
        micro_batch_size=1,
        tokenizer=custom_tokenizer,  # Use our custom tokenizer
    )

    # Instantiate the model
    logger.info("Creating NeMo model...")
    model = LlamaModel(model_config)

    # Create AutoResume to load base model weights
    base_model_path = "/lustre/fsw/coreai_dlalgo_genai/ansubramania/nemo_cache/models/meta-llama/Meta-Llama-3-8B"
    logger.info(f"Setting up AutoResume to load base model from: {base_model_path}")
    resume = AutoResume(
        restore_config=nl.RestoreConfig(path=base_model_path),
        resume_if_exists=True,
    )

    # Use the official train API to handle setup and training
    logger.info("Using official train API to initialize model with PEFT...")
    train_result = validate(
        model=model,
        data=data_module,
        trainer=trainer,
        resume=resume,
        model_transform=peft_config,
        tokenizer=custom_tokenizer,  # Use our custom tokenizer
    )

    logger.info(f"Train API completed successfully. Results saved to: {train_result}")
    logger.info("Model is now properly initialized with PEFT")
    # Megatron Parallel wrapped around LlamaModel
    return trainer.model


def setup_lora_only_training(nemo_model, nemo_lm_model):
    """Set up models so that only LoRA adapter parameters are trainable.

    Special handling for NeMo: PEFT callback freezes model after validation,
    so we need to properly unfreeze LoRA adapters and fix version counter issues.
    """
    logger.info("Setting up LoRA-only training...")

    # NeMo model (single module) - special handling for PEFT callback freezing
    nemo_model.train()
    nemo_base_count = 0
    nemo_lora_count = 0
    nemo_unfrozen_count = 0

    for name, param in nemo_model.named_parameters():
        if "adapter" in name.lower() or "lora" in name.lower():
            # This is a LoRA parameter - enable gradients
            nemo_lora_count += 1
            was_frozen = not param.requires_grad

            if was_frozen:
                nemo_unfrozen_count += 1
                logger.debug(f"Unfreezing NeMo LoRA parameter: {name}")

            # Force gradient tracking for LoRA parameters (NeMo PEFT callback fix)
            # The PEFT callback freezes model after validation, causing version counter issues
            param.requires_grad_(True)

            # Critical fix for NeMo's frozen LoRA parameters after validation
            # Clone the parameter data to create a new tensor with proper version tracking
            if was_frozen or not hasattr(param, "_version") or param._version is None:
                original_data = param.data.clone()
                param.data = original_data.detach().requires_grad_(True)
                logger.debug(f"Fixed version tracking for {name}")
        else:
            # This is a base model parameter - ensure it stays frozen
            nemo_base_count += 1
            param.requires_grad = False

    # NeMo-LM model (also single module) - standard handling
    nemo_lm_model.train()
    nemo_lm_base_count = 0
    nemo_lm_lora_count = 0
    nemo_lm_unfrozen_count = 0

    for name, param in nemo_lm_model.named_parameters():
        if "adapter" in name.lower() or "lora" in name.lower():
            # This is a LoRA parameter - enable gradients
            nemo_lm_lora_count += 1
            if not param.requires_grad:
                param.requires_grad = True
                nemo_lm_unfrozen_count += 1
            # Ensure gradient tracking is enabled
            param.requires_grad_(True)
        else:
            # This is a base model parameter - freeze it
            nemo_lm_base_count += 1
            param.requires_grad = False

    logger.info(
        f"NeMo: Froze {nemo_base_count} base parameters, unfroze {nemo_lora_count} LoRA parameters ({nemo_unfrozen_count} were previously frozen)"
    )
    logger.info(
        f"NeMo-LM: Froze {nemo_lm_base_count} base parameters, unfroze {nemo_lm_lora_count} LoRA parameters ({nemo_lm_unfrozen_count} were previously frozen)"
    )
    logger.info("Applied NeMo PEFT callback fix for version counter issue")
    return (nemo_base_count, nemo_lora_count), (nemo_lm_base_count, nemo_lm_lora_count)


def verify_lora_only_training(nemo_model, nemo_lm_model):
    """Verify that only LoRA adapter parameters are trainable."""
    logger.info("Verifying LoRA-only training setup...")

    # Check NeMo model
    nemo_trainable_base = []
    nemo_trainable_lora = []
    nemo_frozen_base = 0
    nemo_frozen_lora = 0

    for name, param in nemo_model.named_parameters():
        is_lora = "adapter" in name.lower() or "lora" in name.lower()
        if param.requires_grad:
            if is_lora:
                nemo_trainable_lora.append(name)
            else:
                nemo_trainable_base.append(name)
        else:
            if is_lora:
                nemo_frozen_lora += 1
            else:
                nemo_frozen_base += 1

    # Check NeMo-LM model
    nemo_lm_trainable_base = []
    nemo_lm_trainable_lora = []
    nemo_lm_frozen_base = 0
    nemo_lm_frozen_lora = 0

    for name, param in nemo_lm_model.named_parameters():
        is_lora = "adapter" in name.lower() or "lora" in name.lower()
        if param.requires_grad:
            if is_lora:
                nemo_lm_trainable_lora.append(name)
            else:
                nemo_lm_trainable_base.append(name)
        else:
            if is_lora:
                nemo_lm_frozen_lora += 1
            else:
                nemo_lm_frozen_base += 1

    logger.info(
        f"NeMo: {len(nemo_trainable_lora)} trainable LoRA, {len(nemo_trainable_base)} trainable base, {nemo_frozen_base} frozen base, {nemo_frozen_lora} frozen LoRA"
    )
    logger.info(
        f"NeMo-LM: {len(nemo_lm_trainable_lora)} trainable LoRA, {len(nemo_lm_trainable_base)} trainable base, {nemo_lm_frozen_base} frozen base, {nemo_lm_frozen_lora} frozen LoRA"
    )

    # Report any issues
    if nemo_trainable_base:
        logger.warning(
            f"NeMo has {len(nemo_trainable_base)} trainable base parameters (should be 0): {nemo_trainable_base[:5]}"
        )
    if nemo_lm_trainable_base:
        logger.warning(
            f"NeMo-LM has {len(nemo_lm_trainable_base)} trainable base parameters (should be 0): {nemo_lm_trainable_base[:5]}"
        )
    if nemo_frozen_lora > 0:
        logger.warning(f"NeMo has {nemo_frozen_lora} frozen LoRA parameters (should be 0)")
    if nemo_lm_frozen_lora > 0:
        logger.warning(f"NeMo-LM has {nemo_lm_frozen_lora} frozen LoRA parameters (should be 0)")

    return {
        "nemo": {"trainable_lora": len(nemo_trainable_lora), "trainable_base": len(nemo_trainable_base)},
        "nemo_lm": {"trainable_lora": len(nemo_lm_trainable_lora), "trainable_base": len(nemo_lm_trainable_base)},
    }


def compare_model_structures(nemo_model, nemo_lm_model):
    """Compare the structure of both models."""
    logger.info("Comparing model structures...")

    def get_module_info(model, prefix=""):
        """Get module information recursively."""
        info = {}
        # Both models are single modules
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            info[full_name] = {
                "type": type(module).__name__,
                "parameters": sum(p.numel() for p in module.parameters()),
                "trainable_parameters": sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            # Recursively get child modules
            child_info = get_module_info(module, full_name)
            info.update(child_info)
        return info

    nemo_info = get_module_info(nemo_model)
    nemo_lm_info = get_module_info(nemo_lm_model)

    print("\n=== MODEL STRUCTURE COMPARISON ===")
    print(f"NeMo model type: {type(nemo_model)}")
    print(f"NeMo-LM model type: {type(nemo_lm_model)}")
    print(f"NeMo model modules: {len(nemo_info)}")
    print(f"NeMo-LM model modules: {len(nemo_lm_info)}")

    # Find differences (simplified comparison for now)
    print(f"\nNeMo model has {len(nemo_info)} named modules")
    print(f"NeMo-LM model has {len(nemo_lm_info)} named modules")

    return nemo_info, nemo_lm_info


def compare_parameters(nemo_model, nemo_lm_model):
    """Compare parameters between models."""
    logger.info("Comparing model parameters...")

    def get_param_info(model, model_name=""):
        """Get parameter information."""
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        lora_params = 0

        param_details = {}

        # Both models are single modules
        logger.info(f"{model_name} is a single model")
        for name, param in model.named_parameters():
            total_params += param.numel()
            param_details[name] = {
                "shape": param.shape,
                "requires_grad": param.requires_grad,
                "dtype": param.dtype,
                "is_lora": "adapter" in name.lower() or "lora" in name.lower(),
            }

            if param.requires_grad:
                trainable_params += param.numel()
                if "adapter" in name.lower() or "lora" in name.lower():
                    lora_params += param.numel()
            else:
                frozen_params += param.numel()

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "lora_params": lora_params,
            "param_details": param_details,
        }

    nemo_params = get_param_info(nemo_model, "NeMo")
    nemo_lm_params = get_param_info(nemo_lm_model, "NeMo-LM")

    print("\n=== PARAMETER COMPARISON ===")
    print(
        f"NeMo - Total: {nemo_params['total_params']:,}, Trainable: {nemo_params['trainable_params']:,}, LoRA: {nemo_params['lora_params']:,}"
    )
    print(
        f"NeMo-LM - Total: {nemo_lm_params['total_params']:,}, Trainable: {nemo_lm_params['trainable_params']:,}, LoRA: {nemo_lm_params['lora_params']:,}"
    )

    # Compare LoRA parameters specifically
    nemo_lora_params = {k: v for k, v in nemo_params["param_details"].items() if v["is_lora"]}
    nemo_lm_lora_params = {k: v for k, v in nemo_lm_params["param_details"].items() if v["is_lora"]}

    print(f"\nNeMo LoRA parameters: {len(nemo_lora_params)}")
    for name, info in list(nemo_lora_params.items())[:5]:  # Show first 5 for brevity
        print(f"  {name}: {info['shape']}, grad={info['requires_grad']}")
    if len(nemo_lora_params) > 5:
        print(f"  ... and {len(nemo_lora_params) - 5} more")

    print(f"\nNeMo-LM LoRA parameters: {len(nemo_lm_lora_params)}")
    for name, info in list(nemo_lm_lora_params.items())[:5]:  # Show first 5 for brevity
        print(f"  {name}: {info['shape']}, grad={info['requires_grad']}")
    if len(nemo_lm_lora_params) > 5:
        print(f"  ... and {len(nemo_lm_lora_params) - 5} more")

    return nemo_params, nemo_lm_params


def compare_weights(nemo_model, nemo_lm_model):
    """Compare specific weight values between models."""
    logger.info("Comparing model weights...")

    def extract_weights(model, model_name=""):
        """Extract weights from model."""
        weights = {}
        devices = set()

        # Both models are single modules
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
            devices.add(param.device)

        logger.info(f"{model_name} weights are on devices: {devices}")
        return weights

    def move_to_same_device(tensor1, tensor2, prefer_gpu=True):
        """Move tensors to the same device, preferring GPU if available."""
        if tensor1.device == tensor2.device:
            return tensor1, tensor2

        # Prefer GPU device if available
        if prefer_gpu:
            if tensor1.device.type == "cuda":
                return tensor1, tensor2.to(tensor1.device)
            elif tensor2.device.type == "cuda":
                return tensor1.to(tensor2.device), tensor2

        # Fall back to CPU
        return tensor1.cpu(), tensor2.cpu()

    nemo_weights = extract_weights(nemo_model, "NeMo")
    nemo_lm_weights = extract_weights(nemo_lm_model, "NeMo-LM")

    print("\n=== WEIGHT COMPARISON ===")
    print(f"NeMo model has {len(nemo_weights)} weight tensors")
    print(f"NeMo-LM model has {len(nemo_lm_weights)} weight tensors")

    # Find embedding layer weights to compare
    nemo_embedding = None
    nemo_lm_embedding = None

    for name, weight in nemo_weights.items():
        if "embedding" in name.lower() and "word" in name.lower():
            nemo_embedding = weight
            print(f"Found NeMo embedding: {name}, shape: {weight.shape}, device: {weight.device}")
            break

    for name, weight in nemo_lm_weights.items():
        if "embedding" in name.lower() and "word" in name.lower():
            nemo_lm_embedding = weight
            print(f"Found NeMo-LM embedding: {name}, shape: {weight.shape}, device: {weight.device}")
            break

    if nemo_embedding is not None and nemo_lm_embedding is not None:
        if nemo_embedding.shape == nemo_lm_embedding.shape:
            # Move to same device before comparison
            nemo_embedding_comp, nemo_lm_embedding_comp = move_to_same_device(nemo_embedding, nemo_lm_embedding)

            diff = torch.abs(nemo_embedding_comp - nemo_lm_embedding_comp)
            max_diff = torch.max(diff)
            mean_diff = torch.mean(diff)
            print(f"Embedding weight difference - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}")

            # Check if they're essentially the same (within floating point precision)
            if max_diff < 1e-6:
                print("✓ Embedding weights are essentially identical")
            else:
                print("✗ Embedding weights have significant differences")
        else:
            print(f"✗ Embedding shape mismatch: NeMo {nemo_embedding.shape} vs NeMo-LM {nemo_lm_embedding.shape}")
    else:
        if nemo_embedding is None:
            print("✗ Could not find NeMo embedding layer")
        if nemo_lm_embedding is None:
            print("✗ Could not find NeMo-LM embedding layer")

    # Separate LoRA and non-LoRA weights
    nemo_lora_weights = {k: v for k, v in nemo_weights.items() if "adapter" in k.lower() or "lora" in k.lower()}
    nemo_lm_lora_weights = {k: v for k, v in nemo_lm_weights.items() if "adapter" in k.lower() or "lora" in k.lower()}

    nemo_base_weights = {k: v for k, v in nemo_weights.items() if not ("adapter" in k.lower() or "lora" in k.lower())}
    nemo_lm_base_weights = {
        k: v for k, v in nemo_lm_weights.items() if not ("adapter" in k.lower() or "lora" in k.lower())
    }

    print("\n=== BASE MODEL WEIGHTS COMPARISON ===")
    print(f"Base weights - NeMo: {len(nemo_base_weights)}, NeMo-LM: {len(nemo_lm_base_weights)}")

    # Compare base model weights - these should be identical (from same checkpoint)
    if len(nemo_base_weights) > 0 and len(nemo_lm_base_weights) > 0:
        print("\nComparing base model weights (should be identical from same checkpoint):")

        # Find matching weight pairs by looking for similar names
        matched_pairs = []
        unmatched_nemo = []
        unmatched_nemo_lm = []

        for nemo_name, nemo_weight in nemo_base_weights.items():
            # Try to find corresponding weight in NeMo-LM
            matched = False
            for nemo_lm_name, nemo_lm_weight in nemo_lm_base_weights.items():
                if nemo_name == nemo_lm_name and nemo_weight.shape == nemo_lm_weight.shape:
                    matched_pairs.append((nemo_name, nemo_lm_name, nemo_weight, nemo_lm_weight))
                    matched = True
                    break

            if not matched:
                unmatched_nemo.append(nemo_name)

        # Find unmatched NeMo-LM weights
        matched_nemo_lm_names = {pair[1] for pair in matched_pairs}
        unmatched_nemo_lm = [name for name in nemo_lm_base_weights.keys() if name not in matched_nemo_lm_names]

        print(f"Matched base weight pairs: {len(matched_pairs)}")
        print(f"Unmatched NeMo weights: {len(unmatched_nemo)}")
        print(f"Unmatched NeMo-LM weights: {len(unmatched_nemo_lm)}")

        if len(matched_pairs) > 0:
            # Sample a few pairs to compare
            sample_pairs = matched_pairs[: min(5, len(matched_pairs))]
            identical_count = 0
            total_max_diff = 0
            total_mean_diff = 0

            for i, (nemo_name, nemo_lm_name, nemo_weight, nemo_lm_weight) in enumerate(sample_pairs):
                # Move to same device
                nemo_comp, nemo_lm_comp = move_to_same_device(nemo_weight, nemo_lm_weight)

                # Compare weights
                diff = torch.abs(nemo_comp - nemo_lm_comp)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()

                total_max_diff += max_diff
                total_mean_diff += mean_diff

                weight_type = "unknown"
                if "attention" in nemo_name.lower() or "attn" in nemo_name.lower():
                    weight_type = "attention"
                elif "mlp" in nemo_name.lower() or "ffn" in nemo_name.lower():
                    weight_type = "MLP"
                elif "norm" in nemo_name.lower() or "layernorm" in nemo_name.lower():
                    weight_type = "normalization"
                elif "embedding" in nemo_name.lower():
                    weight_type = "embedding"

                print(f"  {weight_type} weight - Max diff: {max_diff:.8f}, Mean diff: {mean_diff:.8f}")
                print(f"    NeMo: {nemo_name} {nemo_weight.shape}")
                print(f"    NeMo-LM: {nemo_lm_name} {nemo_lm_weight.shape}")

                if max_diff < 1e-6:
                    identical_count += 1
                    print("    ✓ Weights are essentially identical")
                else:
                    print("    ✗ Weights have significant differences")
                print()

            avg_max_diff = total_max_diff / len(sample_pairs)
            avg_mean_diff = total_mean_diff / len(sample_pairs)

            print(f"Summary of {len(sample_pairs)} sampled base weight comparisons:")
            print(f"  Average max difference: {avg_max_diff:.8f}")
            print(f"  Average mean difference: {avg_mean_diff:.8f}")
            print(f"  Identical weights: {identical_count}/{len(sample_pairs)}")

            if identical_count == len(sample_pairs):
                print("  ✓ All sampled base weights are identical - checkpoint loading works correctly!")
            else:
                print("  ✗ Some base weights differ - checkpoint loading may have issues")

        # Show some unmatched weights for debugging
        if len(unmatched_nemo) > 0:
            print("\nSome unmatched NeMo weights:")
            for name in unmatched_nemo[:3]:
                print(f"  {name}")
            if len(unmatched_nemo) > 3:
                print(f"  ... and {len(unmatched_nemo) - 3} more")

        if len(unmatched_nemo_lm) > 0:
            print("\nSome unmatched NeMo-LM weights:")
            for name in unmatched_nemo_lm[:3]:
                print(f"  {name}")
            if len(unmatched_nemo_lm) > 3:
                print(f"  ... and {len(unmatched_nemo_lm) - 3} more")

    print("\n=== LORA WEIGHTS COMPARISON ===")
    print(f"LoRA weights - NeMo: {len(nemo_lora_weights)}, NeMo-LM: {len(nemo_lm_lora_weights)}")

    if len(nemo_lora_weights) > 0:
        print("NeMo LoRA weight names:")
        for name in list(nemo_lora_weights.keys())[:3]:
            print(f"  {name}")
        if len(nemo_lora_weights) > 3:
            print(f"  ... and {len(nemo_lora_weights) - 3} more")

    if len(nemo_lm_lora_weights) > 0:
        print("NeMo-LM LoRA weight names:")
        for name in list(nemo_lm_lora_weights.keys())[:3]:
            print(f"  {name}")
        if len(nemo_lm_lora_weights) > 3:
            print(f"  ... and {len(nemo_lm_lora_weights) - 3} more")

    if len(nemo_lora_weights) > 0 and len(nemo_lm_lora_weights) > 0:
        # Compare a few LoRA weights (they should be randomly initialized and different)
        nemo_lora_sample = list(nemo_lora_weights.values())[0]
        nemo_lm_lora_sample = list(nemo_lm_lora_weights.values())[0]

        print(f"Sample LoRA weight shapes - NeMo: {nemo_lora_sample.shape}, NeMo-LM: {nemo_lm_lora_sample.shape}")
        print(f"Sample LoRA weight devices - NeMo: {nemo_lora_sample.device}, NeMo-LM: {nemo_lm_lora_sample.device}")

        # Move to same device for norm calculation
        nemo_lora_comp, nemo_lm_lora_comp = move_to_same_device(nemo_lora_sample, nemo_lm_lora_sample)

        # Check if they have similar magnitudes (both should be small random values)
        nemo_lora_norm = torch.norm(nemo_lora_comp)
        nemo_lm_lora_norm = torch.norm(nemo_lm_lora_comp)
        print(f"Sample LoRA weight norms - NeMo: {nemo_lora_norm:.6f}, NeMo-LM: {nemo_lm_lora_norm:.6f}")

        if abs(nemo_lora_norm - nemo_lm_lora_norm) / max(nemo_lora_norm, nemo_lm_lora_norm) < 0.5:
            print("✓ LoRA weights have similar magnitudes (good initialization)")
        else:
            print("✗ LoRA weights have very different magnitudes")
    elif len(nemo_lora_weights) == 0:
        print("✗ No LoRA weights found in NeMo model")
    elif len(nemo_lm_lora_weights) == 0:
        print("✗ No LoRA weights found in NeMo-LM model")

    return nemo_weights, nemo_lm_weights


def compare_loss_computation_simple(nemo_model, nemo_lm_model):
    """Simple loss comparison using identical synthetic data."""
    logger.info("Comparing loss computation with identical data...")

    # Create identical synthetic batch
    batch_size = 2
    seq_length = 64

    # Create simple batch
    tokens = torch.randint(10, VOCAB_SIZE - 10, (batch_size, seq_length))
    labels = torch.cat([tokens[:, 1:], torch.full((batch_size, 1), 1, dtype=tokens.dtype)], dim=1)

    # Create shared inputs (both models should take the same format)
    loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
    position_ids = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)

    # Create base batch format that both can use
    base_batch = {
        "tokens": tokens,  # NeMo format
        "input_ids": tokens,  # Standard format
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    # NeMo batch (uses 'tokens' key and includes loss_mask)
    nemo_batch = {
        "tokens": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_mask": loss_mask,
    }

    # NeMo-LM batch (uses 'input_ids' key)
    nemo_lm_batch = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    cuda_device = torch.device("cuda:0")
    print(f"Using CUDA device: {cuda_device}")

    # Move models to GPU if not already there
    nemo_model = nemo_model.cuda()
    print("Moved NeMo model to GPU")

    nemo_lm_model = nemo_lm_model.cuda()
    print("Moved NeMo-LM model to GPU")

    # Move all batch tensors to GPU
    for k, v in nemo_batch.items():
        if torch.is_tensor(v):
            nemo_batch[k] = v.cuda()
            print(f"Moved NeMo batch[{k}] to {nemo_batch[k].device}")

    for k, v in nemo_lm_batch.items():
        if torch.is_tensor(v):
            nemo_lm_batch[k] = v.cuda()
            print(f"Moved NeMo-LM batch[{k}] to {nemo_lm_batch[k].device}")

    # Move loss_mask to GPU too
    loss_mask = loss_mask.cuda()
    print(f"Moved loss_mask to {loss_mask.device}")

    # Verify everything is on CUDA
    print("\nVerifying all tensors are on CUDA:")
    print(f"NeMo batch devices: {[(k, v.device) for k, v in nemo_batch.items() if torch.is_tensor(v)]}")
    print(f"NeMo-LM batch devices: {[(k, v.device) for k, v in nemo_lm_batch.items() if torch.is_tensor(v)]}")
    print(f"Loss mask device: {loss_mask.device}")

    # Set device references for later use
    nemo_device = cuda_device
    nemo_lm_device = cuda_device

    print("\n=== LOSS COMPARISON WITH IDENTICAL DATA ===")
    print(f"Batch shape: {tokens.shape}")
    print(f"Token range: {tokens.min().item()} to {tokens.max().item()}")

    # Set models to train mode and enable gradients for LoRA
    nemo_model.train()
    nemo_lm_model.train()

    for name, param in nemo_model.named_parameters():
        param.requires_grad = True

    # Don't use no_grad since LoRA needs gradient tracking
    try:
        # Try direct forward with standard parameter names
        nemo_forward_batch = {
            "input_ids": nemo_batch["tokens"],
            "position_ids": nemo_batch["position_ids"],
            "attention_mask": nemo_batch["attention_mask"],
            "labels": nemo_batch["labels"],
        }
        nemo_output = nemo_model(**nemo_forward_batch)

        # Extract loss
        if hasattr(nemo_output, "loss"):
            nemo_loss = nemo_output.loss.item()
        elif isinstance(nemo_output, tuple):
            nemo_loss = nemo_output[0].item() if torch.is_tensor(nemo_output[0]) else nemo_output[1].item()
        else:
            nemo_loss = None
            print(f"NeMo output type: {type(nemo_output)}")
            print(f"NeMo output: {nemo_output}")

    except Exception as e:
        print(f"NeMo forward pass error: {e}")
        import traceback

        traceback.print_exc()
        nemo_loss = None

    try:
        # NeMo-LM forward pass (returns output_tensor, loss computed separately)
        nemo_lm_output_tensor = nemo_lm_model(**nemo_lm_batch)

        # Import the loss function used in NeMo-LM
        from megatron.hub.training.losses import masked_next_token_loss

        # Compute loss using NeMo-LM's loss function (loss_mask already on GPU)
        nemo_lm_loss_tensor, num_tokens, loss_dict = masked_next_token_loss(loss_mask, nemo_lm_output_tensor)
        nemo_lm_loss = nemo_lm_loss_tensor.item()
        nemo_lm_num_tokens = num_tokens.item()

        print("NeMo-LM loss computation:")
        print(f"  Raw loss (sum): {nemo_lm_loss:.6f}")
        print(f"  Number of tokens: {nemo_lm_num_tokens}")
        print(f"  Mean loss: {nemo_lm_loss / nemo_lm_num_tokens:.6f}")

    except Exception as e:
        print(f"NeMo-LM forward pass error: {e}")
        import traceback

        traceback.print_exc()
        nemo_lm_loss = None
        nemo_lm_num_tokens = None

    # Compare results
    if nemo_loss is not None and nemo_lm_loss is not None:
        # NeMo-LM returns sum, so compute mean for fair comparison
        if "nemo_lm_num_tokens" in locals() and nemo_lm_num_tokens is not None:
            nemo_lm_mean_loss = nemo_lm_loss / nemo_lm_num_tokens
            print("\n=== FAIR COMPARISON (Mean Losses) ===")
            print(f"NeMo loss (mean): {nemo_loss:.6f}")
            print(f"NeMo-LM loss (mean): {nemo_lm_mean_loss:.6f}")

            mean_ratio = nemo_lm_mean_loss / nemo_loss
            mean_diff = abs(nemo_lm_mean_loss - nemo_loss)

            print(f"Ratio (NeMo-LM mean/NeMo): {mean_ratio:.4f}")
            print(f"Absolute difference: {mean_diff:.6f}")

            if mean_diff < 1e-6:
                print("✓ Losses are essentially identical (after mean normalization)")
            elif mean_diff < 1e-3:
                print("⚠ Losses are similar but have small differences")
            else:
                print("✗ Losses are still significantly different even after normalization")
        else:
            # Fallback to original comparison
            ratio = nemo_lm_loss / nemo_loss
            diff = abs(nemo_lm_loss - nemo_loss)

            print(f"NeMo loss: {nemo_loss:.6f}")
            print(f"NeMo-LM loss (raw sum): {nemo_lm_loss:.6f}")
            print(f"Ratio (NeMo-LM/NeMo): {ratio:.4f}")
            print(f"Absolute difference: {diff:.6f}")

            # Test if NeMo-LM is returning sum instead of mean
            num_tokens = batch_size * seq_length  # Total tokens in batch
            predicted_nemo_lm_if_sum = nemo_loss * num_tokens
            sum_ratio = nemo_lm_loss / predicted_nemo_lm_if_sum

            print("\n🔍 SUM vs MEAN ANALYSIS:")
            print(f"Batch tokens: {num_tokens}")
            print(f"If NeMo-LM returns sum, expected loss: {predicted_nemo_lm_if_sum:.6f}")
            print(f"Actual NeMo-LM loss: {nemo_lm_loss:.6f}")
            print(f"Sum hypothesis ratio: {sum_ratio:.4f}")

            if 0.95 < sum_ratio < 1.05:
                print("✓ CONFIRMED: NeMo-LM returns SUM, NeMo returns MEAN")
                print("  Solution: Divide NeMo-LM loss by number of tokens for comparison")
            else:
                print("✗ Sum hypothesis doesn't match - different issue")

        if diff < 1e-6:
            print("✓ Losses are essentially identical")
        elif diff < 1e-3:
            print("⚠ Losses are similar but have small differences")
        else:
            print("✗ Losses are significantly different")

        if ratio > 1.05:
            print(f"⚠ NeMo-LM loss is {(ratio - 1) * 100:.1f}% higher than NeMo")
        elif ratio < 0.95:
            print(f"⚠ NeMo-LM loss is {(1 - ratio) * 100:.1f}% lower than NeMo")

        # Most common causes based on the ratio
        if 1.01 < ratio < 1.20:  # 1-20% higher
            print("\n🔍 LIKELY CAUSES (NeMo-LM consistently 1-20% higher):")
            print("1. Different loss reduction: NeMo uses 'mean', NeMo-LM uses 'sum' (or vice versa)")
            print("2. Different gradient accumulation handling")
            print("3. Different mixed precision loss scaling")
        elif ratio > 1.20:  # >20% higher
            print("\n🔍 LIKELY CAUSES (NeMo-LM >20% higher):")
            print("1. Different loss masking - NeMo-LM not masking padding tokens properly")
            print("2. Different label creation/shifting")
            print("3. Different cross-entropy implementation")

        print("\n📋 DEBUGGING ACTIONS:")
        print("1. Check model configs for: cross_entropy_loss_fusion, loss_reduction")
        print("2. Check DDP configs for: grad_reduce_in_fp32, average_in_collective")
        print("3. Check optimizer configs for: loss scaling, gradient accumulation")
        print("4. Compare actual training configs between your NeMo and NeMo-LM scripts")

    else:
        print("✗ Could not compute losses for comparison")

    return nemo_loss, nemo_lm_loss


def main():
    """Main debugging function."""
    logger.info("Starting model comparison debugging...")

    dist.init_process_group(backend="nccl")

    # Initialize models
    logger.info("Initializing models...")
    nemo_lm_model = initialize_nemo_lm_model()
    base_nemo_model = initialize_nemo_model()
    nemo_model = base_nemo_model[0].module

    # Set up LoRA-only training (freeze base model, enable LoRA adapters)
    logger.info("Setting up LoRA-only training...")
    setup_lora_only_training(nemo_model, nemo_lm_model)

    # Verify that only LoRA parameters are trainable
    verify_lora_only_training(nemo_model, nemo_lm_model)

    # Verify both models have the same vocabulary size
    def get_vocab_size(model):
        if hasattr(model, "embedding"):
            return model.embedding.word_embeddings.weight.shape[0]
        # Try to find embedding layer
        for name, module in model.named_modules():
            if "embedding" in name.lower() and hasattr(module, "weight"):
                return module.weight.shape[0]
        return None

    nemo_vocab_size = get_vocab_size(nemo_model)
    nemo_lm_vocab_size = get_vocab_size(nemo_lm_model)

    print("\n=== VOCABULARY SIZE VERIFICATION ===")
    print(f"NeMo model vocabulary size: {nemo_vocab_size}")
    print(f"NeMo-LM model vocabulary size: {nemo_lm_vocab_size}")
    print(f"Target vocabulary size: {VOCAB_SIZE}")

    if nemo_vocab_size == nemo_lm_vocab_size == VOCAB_SIZE:
        print("✓ All vocabulary sizes match!")
    else:
        print("✗ Vocabulary size mismatch detected")

    # Run comparisons
    print("\n=== NEMO-LM MODEL ===")
    print(nemo_lm_model)

    print("\n=== NEMO MODEL ===")
    print(nemo_model)

    compare_model_structures(nemo_model, nemo_lm_model)
    compare_parameters(nemo_model, nemo_lm_model)
    compare_weights(nemo_model, nemo_lm_model)

    # Test loss computation to identify the source of higher NeMo-LM loss
    print("\n" + "=" * 60)
    print("INVESTIGATING LOSS DIFFERENCE")
    print("=" * 60)
    compare_loss_computation_simple(base_nemo_model, nemo_lm_model)

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
