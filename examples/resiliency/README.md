# Resiliency

Examples demonstrating Megatron-Bridge resiliency features powered by [nvidia-resiliency-ext](https://github.com/NVIDIA/nvidia-resiliency-ext).

## Documentation

- [Megatron-Bridge Resiliency Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/resiliency.html)
- [nvidia-resiliency-ext Documentation](https://nvidia.github.io/nvidia-resiliency-ext/)

## Prerequisites

- [NeMo Framework Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- 2+ GPUs for distributed examples
- HuggingFace token with Llama access (set `HF_TOKEN` env var)
  - Accept license at https://huggingface.co/meta-llama/Llama-3.2-1B

## Examples

### Straggler Detection

Detects slow-performing GPUs during training using NVRx straggler detection.

```bash
uv run python -m torch.distributed.run --nproc_per_node=2 examples/resiliency/straggler_detection/straggler_detection_example.py
```

Or use the launch script:

```bash
./examples/resiliency/straggler_detection/run_straggler_detection.sh
```

### Fault Tolerance

Enables automatic hang detection and job restart using the `ft_launcher`.

```bash
./examples/resiliency/fault_tolerance/run_fault_tolerance.sh
```

To test fault recovery with simulated failures:

```bash
./examples/resiliency/fault_tolerance/run_fault_tolerance.sh --simulate-fault
```

Note: Fault simulation requires careful timing - the fault must trigger after a checkpoint is saved but before training completes. See the `--simulate-fault` option in `fault_tolerance_example.py` for details.
