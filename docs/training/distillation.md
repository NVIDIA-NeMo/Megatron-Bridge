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

* Student and teacher models must support the same parallelism strategy.
* If Pipeline Parallelism is enabled, intermediate-state based KD losses are only supported on the final pipeline stage.
* Virtual pipeline parallelism is not supported.

## Configuration

### Knowledge Distillation Config

You can configure the KD process via the `ModelOptDistillConfig` class or a YAML file. The configuration includes:

* `logit_layers`: The layer names of student and teacher model logit layers. These names correspond to the PyTorch submodule attributes of the Megatron Core model. (For GPT-based models, this is `"output_layer"`). Default: `["output_layer", "output_layer"]`
* `intermediate_layer_pairs`: A list of pairs of intermediate layer names. These pairs will by default have a Cosine-Similarity loss between them, and if tensor-parallelism is enabled, these layers must have sequence parallel outputs (i.e. LayerNorms), as Cosine loss cannot have a split hidden dimension. Default: `[["decoder.final_layernorm", "decoder.final_layernorm"]]`
* `skip_lm_loss`: Whether to skip the default language modeling (LM) loss. If `false`, it will be added to the distillation loss. (Note it consumes more memory). Default: `true`
* `kd_loss_scale`: Relative scale factor for the distillation loss. The cumulative logits-and-intermediate loss gets scaled to `kd_loss_scale` times the magnitude of the LM loss. Not used if `skip_lm_loss` is `true`. Default: `1.0`
* `logit_kl_temperature`: Temperature variable for KL Divergence loss calculation. Default: `1.0`

When using the provided recipe script, put KD settings under `model.kd_config` in the ConfigContainer override file:

```yaml
model:
  kd_config:
    logit_layers: ["output_layer", "output_layer"]
    intermediate_layer_pairs:
      - ["decoder.final_layernorm", "decoder.final_layernorm"]
    skip_lm_loss: true
    kd_loss_scale: 1.0
    logit_kl_temperature: 2.0
```

## Usage

### Model Optimizer Script

NVIDIA Model Optimizer provides a general [Megatron Bridge distillation script](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#distillation). This is the recommended workflow for custom use cases; see the Model Optimizer repository for usage details.

### Megatron Bridge Recipe

You can also quickly try a knowledge distillation example for distilling Llama3.2-3B into Llama3.2-1B with:

```bash
uv run -m torch.distributed.run --nproc_per_node=2 examples/distillation/llama/distill_llama32_3b-1b.py
```

You can customize this by providing a YAML configuration file to override default settings:

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/distillation/llama/distill_llama32_3b-1b.py \
    --config-file my_custom_config.yaml
```

It additionally supports Hydra-style CLI overrides for flexible configuration (CLI overrides take precedence over YAML):

```bash
uv run -m torch.distributed.run --nproc_per_node=2 examples/distillation/llama/distill_llama32_3b-1b.py \
    --config-file my_custom_config.yaml \
    train.global_batch_size=512
```

## Model Support

Currently, distillation is supported for GPT and Mamba-based model providers.

To enable distillation for a model:

1. Set the `teacher` attribute to the teacher model configuration
2. Configure `kd_config` with desired distillation settings (else uses default)
3. Use `convert_to_distillation_provider()` to convert your existing model provider

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
