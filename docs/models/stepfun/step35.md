# Step-3.5-Flash

[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) is a MoE language model from StepFun. Megatron Bridge supports checkpoint conversion, inference, and continued pretraining through a dedicated Step-3.5 bridge, provider, and recipe.

## Supported Variants

- Step-3.5-Flash: https://huggingface.co/stepfun-ai/Step-3.5-Flash

## Architecture Notes

- Hybrid attention pattern with full attention interleaved with sliding attention.
- Fused per-head attention gate (`g_proj`) is merged into Megatron `linear_qkv` weights.
- MoE decoder layers are surrounded by dense layers at the beginning/end and MTP layers.
- The provider carries `layer_types`, full-attention settings, and sliding-attention settings from the Hugging Face config.
- Current examples require Megatron-LM `dev` branch changes until the required MCore support reaches the pinned submodule.

## Examples

For conversion, inference, pretraining, Slurm launch scripts, and MCore branch notes, see the [Step-3.5-Flash examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/stepfun/step35/README.md).

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/stepfun`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/stepfun)
- Recipe: [`src/megatron/bridge/recipes/stepfun/step35.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/stepfun/step35.py)
- Examples: [`examples/models/stepfun/step35`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/stepfun/step35)

