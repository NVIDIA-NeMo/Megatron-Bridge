# Training Initialization Sanity Test

This script (`test_training_initialization.py`) provides a comprehensive sanity test for the training initialization pipeline in NeMo-LM, specifically testing:

1. **Base Model Creation** - Verifies that the model can be instantiated with the correct configuration
2. **Pretrained Checkpoint Loading** - Tests loading weights from a pretrained checkpoint
3. **PEFT Application** - Applies Parameter-Efficient Fine-Tuning (LoRA) to the model
4. **Distributed Model Setup** - Wraps the model with DDP/FSDP for distributed training
5. **Forward Pass** - Performs a test forward pass to ensure the model works end-to-end

## Features

- **Configuration Loading**: Automatically loads the exact configuration used to train the base model from the checkpoint's `run_config.yaml`
- **Smart Overrides**: Only overrides the necessary settings for testing (PEFT config, batch sizes, etc.)
- **Intelligent Tokenizer Detection**: Automatically detects and configures the appropriate tokenizer (HuggingFace vs SentencePiece) based on checkpoint contents
- **Comprehensive Testing**: Tests each component of the training initialization pipeline
- **Detailed Reporting**: Provides clear success/failure messages with timing information
- **PEFT Statistics**: Reports parameter counts and trainable percentage after PEFT application
- **Fallback Configuration**: Provides robust fallback to default Llama3-8B config if YAML loading fails

## Prerequisites

- CUDA-enabled GPU
- NeMo-LM environment properly set up
- Pretrained checkpoint with configuration file

## Usage

### Basic Usage

```bash
python test_training_initialization.py
```

The script is configured by default to use the checkpoint at:
```
/lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/
```

### Expected Output

```
================================================================================
MEGATRON-LM TRAINING INITIALIZATION SANITY TEST
================================================================================
Pretrained checkpoint path: /lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/
PyTorch version: 2.1.0
CUDA available: True
CUDA device count: 1
Current CUDA device: 0

Loading original configuration from: /lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/iter_0000000/run_config.yaml
✅ Configuration loaded successfully from YAML
Overriding configuration for testing...
Found HuggingFace tokenizer assets at: /lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/hf_assets
Configuration loaded and overridden for testing

============================================================
TESTING: Megatron Initialization
============================================================
✅ Megatron initialized successfully in 2.34s

============================================================
TESTING: Base Model Creation
============================================================
✅ Base model created successfully in 15.67s
   Model type: <class 'list'>
   Number of model chunks: 1
   Total parameters: 8,030,261,248

============================================================
TESTING: Pretrained Checkpoint Loading
============================================================
✅ Pretrained checkpoint loaded successfully in 12.45s
   Checkpoint path: /lustre/fsw/coreai_dlalgo_genai/ansubramania/models/Meta-Llama3-8B/

============================================================
TESTING: PEFT Application
============================================================
✅ PEFT applied successfully in 3.21s
   PEFT type: LoRA
   Total parameters: 8,030,261,248
   Trainable parameters: 8,388,608
   Trainable percentage: 0.10%
   Parameters marked for saving: 256

============================================================
TESTING: Distributed Model Setup
============================================================
✅ Distributed model setup completed in 1.89s
   Model type after wrapping: <class 'list'>
   Number of model chunks: 1
   Chunk 0 type: <class 'megatron.core.distributed.distributed_data_parallel.DistributedDataParallel'>

============================================================
TESTING: Model Forward Pass
============================================================
✅ Forward pass completed successfully in 0.45s
   Input shape: torch.Size([128, 1])
   Output is tuple with 1 elements
   First output shape: torch.Size([128, 1, 128256])

================================================================================
🎉 ALL TESTS PASSED! Training initialization pipeline works correctly.
================================================================================
Summary:
  ✅ Base model creation
  ✅ Pretrained checkpoint loading
  ✅ PEFT application
  ✅ Distributed model setup
  ✅ Model forward pass

🚀 Ready to start actual training!
```

## Configuration Details

### Automatic Configuration Loading

The script automatically loads the configuration from:
```
{checkpoint_path}/iter_0000000/run_config.yaml
```

This ensures that the test uses the exact same model configuration, tokenizer settings, and other parameters that were used during the original training.

### Intelligent Tokenizer Configuration

The script automatically detects and configures the appropriate tokenizer based on the checkpoint structure:

1. **HuggingFace Assets Directory**: If `hf_assets/` directory with `tokenizer.json` is found (preferred for converted checkpoints)
2. **HuggingFace Root Directory**: If `tokenizer.json` is found in the checkpoint root directory
3. **SentencePiece Tokenizer**: If `tokenizer.model` is found in the checkpoint directory  
4. **Fallback**: Uses HuggingFace model identifier `meta-llama/Meta-Llama-3-8B` if no local tokenizer files are found

This ensures compatibility with both:
- **Converted checkpoints** (e.g., from HuggingFace to Megatron format) with local tokenizer files
- **Original HuggingFace models** accessed via model identifiers

### Test-Specific Overrides

The following settings are overridden for testing purposes:

#### PEFT Configuration
```python
peft_config = LoRA(
    target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
    dim=16,  # Small rank for testing
    alpha=32,
    dropout=0.1,
)
```

#### Training Settings
- `micro_batch_size = 1`
- `global_batch_size = 1`
- `train_iters = 10`
- `eval_iters = 1`
- `eval_interval = 5`

#### Model Settings
- `tensor_model_parallel_size = 1` (single GPU)
- `pipeline_model_parallel_size = 1`
- `sequence_parallel = False`
- `use_cpu_initialization = True`

#### Checkpoint Settings
- `save = None` (no checkpoint saving during test)
- `finetune = True` (enables PEFT mode)
- `pretrained_checkpoint = {provided_path}`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - The script is configured for single GPU testing with small batch sizes
   - If you still encounter OOM, the model might be too large for your GPU
   - Consider using a smaller model configuration

2. **Configuration File Not Found**
   ```
   FileNotFoundError: Configuration YAML not found at: .../run_config.yaml
   ```
   - Ensure the checkpoint directory contains `iter_0000000/run_config.yaml`
   - Check that the checkpoint path is correct

3. **Import Errors**
   - Ensure the NeMo-LM environment is properly activated
   - Check that all dependencies are installed

4. **Model Loading Failures**
   - Verify that the checkpoint is compatible with the current codebase version
   - Check that the checkpoint contains the expected files

### Debugging

To get more detailed debugging information, you can:

1. Enable more verbose logging by modifying the logger configuration
2. Add breakpoints in the test functions to inspect intermediate states
3. Run individual test functions separately

## Customization

### Changing the Checkpoint Path

Modify the `pretrained_checkpoint_path` variable in the `main()` function:

```python
def main():
    pretrained_checkpoint_path = "/path/to/your/checkpoint/"
    # ... rest of the function
```

### Modifying PEFT Configuration

Edit the PEFT configuration in the `create_test_config()` function:

```python
peft_config = LoRA(
    target_modules=["linear_qkv", "linear_proj"],  # Target fewer modules
    dim=32,  # Larger rank
    alpha=64,
    dropout=0.0,  # No dropout
)
```

### Adding Custom Tests

You can add additional test functions following the pattern:

```python
def test_custom_functionality(cfg: ConfigContainer, model) -> None:
    """Test custom functionality."""
    print_rank_0("\n" + "="*60)
    print_rank_0("TESTING: Custom Functionality")
    print_rank_0("="*60)
    
    try:
        # Your test code here
        print_rank_0("✅ Custom test passed")
    except Exception as e:
        print_rank_0(f"❌ Custom test failed: {e}")
        raise
```

## Next Steps

After the sanity test passes, you can proceed with:

1. **Full PEFT Training**: Use the verified configuration for actual training
2. **Multi-GPU Testing**: Adapt the script for multi-GPU scenarios
3. **Different PEFT Methods**: Test other PEFT techniques beyond LoRA
4. **Performance Benchmarking**: Add timing and memory usage measurements

## Support

If you encounter issues with the test script, please check:

1. The error messages for specific failure points
2. The checkpoint compatibility with your codebase version
3. The GPU memory requirements for your model size
4. The environment setup and dependencies 