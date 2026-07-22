# Training Entry Points

Megatron Bridge provides unified training entry points for pretraining, Supervised Fine-Tuning (SFT), and Parameter-Efficient Fine-Tuning (PEFT). All training modes share the same underlying training loop architecture, differing primarily in their data handling and model configuration.

## Choosing `pretrain()` or `finetune()`

Use `pretrain()` for language-model pretraining jobs that use `GPTDatasetConfig` or `MockGPTDatasetConfig`. This includes training from scratch, continued pretraining on new corpora, and initializing model weights from `checkpoint.pretrained_checkpoint` before starting a new training run.

Use `finetune()` for full SFT and PEFT. The function validates that either `checkpoint.pretrained_checkpoint` or `checkpoint.load` is set, then calls the same underlying training loop used by `pretrain()`. PEFT does not use a separate entry point: set `cfg.peft` to a LoRA or DoRA config, use `GPTSFTDatasetConfig`, `DirectHFSFTDatasetConfig`, or a specialized dataset provider, and launch through `finetune()`.

The public Slurm launcher is `scripts/training/train.sh`, backed by `setup_experiment.py`, which executes
`run_recipe.py` inside the training container. Select exactly one model stem with `--model` or one complete library
recipe function with `--recipe`, and always pass `--mode pretrain`, `sft`, `lora`, or `dora`. Public dataset names
are distinct from data-source selectors. Named presets such as `openmathinstruct2` and `medpix` select specific
datasets, while `megatron-indexed`, `local-jsonl`, and `local-vlm` select local input formats and `mock` selects
generated data. Each choice resolves to a dataset config preset, and the config type selects the existing runtime
builder. Trailing `dataset.*` values override that config directly; `--seq_length` is the only dataset-field
convenience argument. Offline packing is selected independently with `dataset.enable_offline_packing=true`; it is not
encoded in a dataset name. The default `--step-func` is
`llm_step`; pass another registered step explicitly for non-LLM recipes. Common values accept convenience arguments
such as `-gb 8`, `-sl 4096`, and `-tp 2`; `run_recipe.py --help` lists the complete mapping. Every value can also be set
directly with `ConfigContainer` overrides such as `train.global_batch_size=8` and
`model.tensor_model_parallel_size=2`. A trailing override takes precedence over a convenience argument for the same
field.
The launcher inherits only explicitly named `--env NAME` values and forwards `--mount HOST` or
`--mount HOST:CONTAINER` paths. `pretrain` mode runs `pretrain()`; SFT, LoRA, and DoRA run `finetune()`.

## Checkpoint Source by Workflow

| Workflow | Dataset config | How to initialize or resume |
|----------|----------------|-----------------------------|
| From-scratch LLM pretraining | `GPTDatasetConfig` | Leave `checkpoint.pretrained_checkpoint` and `checkpoint.load` unset, or clear the recipe default `checkpoint.load` if an old local checkpoint directory exists. |
| Full native Megatron resume | Any training workflow | Set `checkpoint.load` to the base checkpoint directory and optionally set `checkpoint.ckpt_step` for a specific iteration. `checkpoint.load` should not point to an `iter_N` directory. |
| Initialize from native Megatron weights | Pretraining, SFT, or PEFT | Set `checkpoint.pretrained_checkpoint` to either the base checkpoint directory or a specific `iter_N` directory. |
| Initialize from Hugging Face weights | Pretraining, SFT, or PEFT | Set `checkpoint.pretrained_checkpoint` to a local Hugging Face full-model directory containing `config.json` and weight files. A remote Hugging Face model ID is not a checkpoint path; download it locally or convert it first. |
| Resume PEFT adapter training | PEFT | Keep `checkpoint.pretrained_checkpoint` pointed at the frozen base model and set `checkpoint.load` to the adapter checkpoint directory. `checkpoint.ckpt_step` applies to the adapter `load` path. |

For multi-node jobs and repeatable production runs, converting a Hugging Face model to a native Megatron checkpoint first is usually the most robust option. Use `checkpoint.pretrained_checkpoint` for weight initialization and `checkpoint.load` for training-state resume; using a Hugging Face directory with `checkpoint.load` raises an error because HF format does not contain optimizer, RNG, dataloader, or scheduler state.

## Main Entry Points

The {py:func}`bridge.training.pretrain.pretrain` and {py:func}`bridge.training.finetune.finetune` functions are the primary entry points for pretraining models—either from scratch or through fine-tuning. Each function accepts a {py:class}`bridge.training.config.ConfigContainer` along with a `forward_step_func` that defines how the training loop should be run.


## Forward Step Function

The `forward_step_func` defines how each training step is executed. It should follow this signature:

```python
def forward_step_func(
    global_state: GlobalState,
    data_iterator: Iterable,
    model: MegatronModule,
    return_schedule_plan: bool = False,
) -> tuple[Any, Callable]:
    """Forward step function.
    
    Args:
        global_state: Training state object containing configuration and utilities
        data_iterator: Iterator over training/evaluation data
        model: The model to perform forward step on
        return_schedule_plan: Whether to return schedule plan (for MoE overlap)
        
    Returns:
        tuple containing:
        - output: Forward pass output (tensor or collection of tensors)
        - loss_func: Function to compute loss from the output
    """
```

### Responsibilities

The forward step function has three main responsibilities:

1. **Get a Batch**: Retrieve and process the next batch from the data iterator.
2. **Run Forward Pass**: Execute the model's forward pass on the batch.
3. **Return Loss Function**: Provide a function to compute loss from the output.

### State Access

Megatron Bridge automatically provides the {py:class}`bridge.training.state.GlobalState` object containing:
- **Configuration**: Complete training configuration (`global_state.cfg`).
- **Timers**: Performance monitoring utilities (`global_state.timers`).
- **Training Progress**: Current step, consumed samples (`global_state.train_state`).
- **Loggers**: TensorBoard and WandB loggers for metrics tracking.

All configuration and state information are accessible through the injected `state` object.

For complete implementation examples, see {py:func}`bridge.training.gpt_step.forward_step`.

## Loss Calculation and Reduction

The loss function returned by the forward step can follow different patterns based on your needs:

### Loss Function Patterns

1. **Standard Pattern**: Return `(loss, metadata_dict)`
   - The loss is automatically averaged across microbatches
   - Metadata dict contains named loss components for logging
   - Most common pattern for standard training

2. **Token-aware Pattern**: Return `(loss, num_tokens, metadata_dict)`
   - Loss is averaged across both microbatches and tokens
   - Useful when you want per-token loss averaging
   - Recommended for variable-length sequences

3. **Inference Pattern**: Return arbitrary data structures
   - Used with `collect_non_loss_data=True` and `forward_only=True`
   - Suitable for inference, evaluation metrics, or custom data collection
   - No automatic loss processing applied

### Automatic Loss Processing

The training loop automatically handles:
- **Microbatch Reduction**: Aggregates losses across all microbatches in the global batch.
- **Distributed Reduction**: Performs all-reduce operations across data parallel ranks.
- **Pipeline Coordination**: Only the last pipeline stage computes and reduces losses.
- **Logging Integration**: Automatically logs loss components to TensorBoard/WandB.

For implementation details, see {py:func}`bridge.training.train.train_step` and {py:func}`bridge.training.losses.masked_token_loss`, as an example.

## Customization

### When to Customize

You can customize the forward step function when you need:

- **Custom Loss Functions**: Beyond standard language modeling loss (e.g., adding regularization, multi-objective training).
- **Multi-task Learning**: Training models on multiple tasks simultaneously with different loss components.
- **Custom Data Processing**: Specialized batch preprocessing for domain-specific data formats.
- **Additional Metrics**: Computing extra evaluation metrics during training.
- **Model-specific Logic**: Special handling for custom model architectures or training procedures.
