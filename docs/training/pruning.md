# Pruning

Pruning reduces model size by removing redundant parameters (e.g., shrinking hidden dimensions or layers) while preserving accuracy. In Megatron Bridge, pruning is provided by NVIDIA Model Optimizer (ModelOpt) using the Minitron algorithm for GPT and Mamba-based models loaded from HuggingFace.

## Pre-requisites

Running the pruning example requires Megatron-Bridge and Model-Optimizer dependencies. We recommend using the NeMo container (e.g., `nvcr.io/nvidia/nemo:26.02`). To use the latest ModelOpt scripts, mount your Model-Optimizer repo at `/opt/Megatron-Bridge/3rdparty/Model-Optimizer` or pull the latest changes inside the container (`cd /opt/Megatron-Bridge/3rdparty/Model-Optimizer && git checkout main && git pull`).

## Usage

### Prune to a target parameter count (NAS)

Example: prune Qwen3-8B to 6B on 2 GPUs (Pipeline Parallelism = 2), skipping pruning of `num_attention_heads`. Defaults: 1024 samples from [nemotron-post-training-dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration, at most 20% depth (`num_layers`) and 40% width per prunable hyperparameter (`hidden_size`, `ffn_hidden_size`, ...), top-10 candidates evaluated for MMLU (5% sampled data) to select the best model.

```bash
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_target_params 6e9 \
    --hparams_to_skip num_attention_heads \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B
```

### Prune to a specific architecture (manual config)

Example: prune Qwen3-8B to a fixed architecture. Defaults: 1024 samples from [nemotron-post-training-dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) for calibration.

```bash
torchrun --nproc_per_node 2 /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py \
    --hf_model_name_or_path Qwen/Qwen3-8B \
    --prune_export_config '{"hidden_size": 3584, "ffn_hidden_size": 9216}' \
    --output_hf_path /tmp/Qwen3-8B-Pruned-6B-manual
```

To see the full list of options for advanced configurations, run:

```bash
torchrun --nproc_per_node 1 /opt/Megatron-Bridge/3rdparty/Model-Optimizer/examples/megatron_bridge/prune_minitron.py --help
```

### Uneven pipeline parallelism

If the number of layers is not divisible by the number of GPUs (pipeline parallel size), set `--num_layers_in_first_pipeline_stage` and `--num_layers_in_last_pipeline_stage`. For example, Qwen3-8B with 36 layers on 8 GPUs: set both to 3 to get 3-5-5-5-5-5-5-3 layers per GPU.

## More information

For more details, see the [ModelOpt pruning README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/megatron_bridge#readme).

## Next steps: Knowledge Distillation

Knowledge Distillation is required to recover the performance of the pruned model. See the [Knowledge Distillation](distillation.md) guide for more details.