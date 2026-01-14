# Knowledge Distillation

Megatron Bridge provides a streamlined setup for Knowledge Distillation (KD) training, making it easy to enable and integrate into your workflow. This section explains how to use this feature effectively.

Knowledge Distillation is a technique where a pre-trained model (the "teacher") transfers its learned knowledge to a second model (the "student"), which is typically smaller and faster. This process helps the student model learn more efficiently by mimicking the behavior of the teacher. KD offers two key advantages over traditional training: faster convergence and higher final accuracy.

In Megatron Bridge, KD is enabled by NVIDIA Model Optimizer (ModelOpt) — a library to optimize deep-learning models for inference on GPUs.

## Knowledge Distillation Process

The KD process involves these steps:

1. **Loads Checkpoints**: Loads both the student and teacher model checkpoints.
2. **Replaces Loss Function**: Replaces the standard loss function with the KL-Divergence between the output logits (and potentially additional losses between pairs of intermediate model states).
3. **Trains Models**: Runs forward passes on both models, but executes the backward pass only on the student model.
4. **Saves Checkpoints**: Saves only the student model checkpoint, allowing it to be used later in the same manner as before.

## Limitations

* Only GPT-based checkpoints are currently supported.
* Student and teacher models must support the same parallelism strategy.
* If Pipeline Parallelism is enabled, intermediate-state based KD losses are only supported on the final pipeline stage.

## Configuration

### Knowledge Distillation Config

You can configure the KD process via the `ModelOptDistillConfig` class or a YAML file. The configuration includes:

* `logit_layers`: The layer names of student and teacher model logit layers. These names correspond to the PyTorch submodule attributes of the Megatron Core model. (For GPT-based models, this is `"output_layer"`). Default: `["output_layer", "output_layer"]`
* `intermediate_layer_pairs`: A list of pairs of intermediate layer names. These pairs will by default have a Cosine-Similarity loss between them, and if tensor-parallelism is enabled, these layers must have sequence parallel outputs (i.e. LayerNorms), as Cosine loss cannot have a split hidden dimension. Default: `[["decoder.final_layernorm", "decoder.final_layernorm"]]`
* `skip_lm_loss`: Whether to skip the default language modeling (LM) loss. If `false`, it will be added to the distillation loss. (Note it consumes more memory). Default: `true`
* `kd_loss_scale`: Relative scale factor for the distillation loss. The cumulative logits-and-intermediate loss gets scaled to `kd_loss_scale` times the magnitude of the LM loss. Not used if `skip_lm_loss` is `true`. Default: `1.0`
* `logit_kl_temperature`: Temperature variable for KL Divergence loss calculation. Default: `1.0`

Example YAML configuration:

```yaml
logit_layers: ["output_layer", "output_layer"]
intermediate_layer_pairs:
  - ["decoder.final_layernorm", "decoder.final_layernorm"]
logit_kl_temperature: 2.0
```

## Usage

### Basic Usage with Default Configuration

The simplest way to run knowledge distillation is to use or adapt one of the provided recipe scripts. Here's an example for distilling Llama3.2-3B into Llama3.2-1B:

```bash
torchrun --nproc_per_node=1 examples/recipes/llama/distill_llama32_3b-1b.py
```

### Using a Custom YAML Config File

You can provide a custom YAML configuration file to override default settings:

```bash
torchrun --nproc_per_node=1 examples/recipes/llama/distill_llama32_3b-1b.py \
    --config-file my_custom_config.yaml
```

### Using CLI Overrides

Megatron Bridge supports Hydra-style CLI overrides for flexible configuration:

```bash
torchrun --nproc_per_node=2 examples/recipes/llama/distill_llama32_3b-1b.py \
    model.tensor_model_parallel_size=2 \
    model.teacher.tensor_model_parallel_size=2
```

### Combining YAML and CLI Overrides

CLI overrides take precedence over YAML configuration:

```bash
torchrun --nproc_per_node=2 examples/recipes/llama/distill_llama32_3b-1b.py \
    --config-file conf/my_config.yaml \
    train.global_batch_size=512
```

## Model Support

Currently, distillation is supported for GPT and Mamba-based models

To enable distillation for a model:

1. Use `DistillationProvider` instead of `GPTModelProvider`
2. Set the `teacher` attribute to the teacher model configuration
3. Configure `kd_config` with desired distillation settings

## Checkpointing

During distillation training:

* Only the **student model** checkpoints are saved
* Teacher model remains frozen and is not modified
* Checkpoints can be used for inference or further training like any standard checkpoint

## Best Practices

1. **Match Parallelism**: Ensure student and teacher use compatible parallelism configurations
2. **Monitor Loss**: Track both distillation loss and (if enabled) language modeling loss
3. **Batch Size**: Use larger batch sizes for better stability during distillation
4. **Learning Rate**: Start with a smaller LR than pretraining
5. **Data Quality**: Use high-quality, diverse training data for best distillation results

## Troubleshooting

### Out of Memory Errors

* Reduce `train.micro_batch_size`
* Increase parallelism sizes
* Set `model.kd_config.skip_lm_loss = True` to save memory

## References

For more information on the underlying implementation, see:
* [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
