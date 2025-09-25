from typing import Any


def get_perf_matrix_overrides(yaml_root: Any, args: Any) -> Any:
    """Get the performance matrix overrides from the YAML file."""
    perf = yaml_root.get("perf_matrix") if hasattr(yaml_root, "get") else None
    if not perf:
        return
    if args.gpu not in perf:
        return
    num_gpus_value = args.num_gpus or args.gpus_per_node
    num_gpus_yaml_key = f"num_gpus_{num_gpus_value}"
    gpu_block = perf.get(args.gpu) or {}
    preset = gpu_block.get(num_gpus_yaml_key) or {}

    return preset
