# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import json

import torch
import torch.distributed as dist


def _get_rank_and_world_size():
    """Get the current rank and world size for distributed training.

    Returns:
        Tuple[int, int]: (rank, world_size) where rank=0 and world_size=1 for non-distributed setups.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def tensor_fingerprint(tensor):
    """Create a fingerprint for a tensor that includes basic properties and statistical summaries.

    Args:
        tensor: PyTorch tensor to analyze.

    Returns:
        Dict containing tensor shape, dtype, device, and statistical summaries (min, max, mean, abs_sum).
    """
    cpu_tensor = tensor.detach().float().cpu()
    numel = cpu_tensor.numel()

    # Compute statistics (if there is at least one element).
    if numel > 0:
        stats = {
            "min": float(cpu_tensor.min()),
            "max": float(cpu_tensor.max()),
            "mean": float(cpu_tensor.mean()),
            "abs_sum": float(cpu_tensor.abs().sum()),
        }
    else:
        stats = {"min": None, "max": None, "mean": None, "abs_sum": None}

    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "device": str(tensor.device),
        **stats,
    }


def safe_convert(obj):
    """Recursively convert objects into JSON-serializable representations.

    Args:
        obj: Object to convert (tensor, list, dict, etc.)

    Returns:
        JSON-serializable representation where tensors are replaced by their fingerprint.
    """
    if isinstance(obj, torch.Tensor):
        return tensor_fingerprint(obj)
    elif isinstance(obj, (list, tuple)):
        return [safe_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return f"<non-serializable: {type(obj).__name__}>"


def get_forward_arg_names(module, inputs):
    """Extract the forward() method's argument names (excluding 'self').

    Args:
        module: PyTorch module to inspect.
        inputs: Tuple of input arguments to the forward method.

    Returns:
        List of argument names. Falls back to default names if inspection fails.
    """
    try:
        sig = inspect.signature(module.forward)
        arg_names = list(sig.parameters.keys())
        if arg_names and arg_names[0] == "self":
            arg_names = arg_names[1:]
        if len(arg_names) != len(inputs):
            return [f"input{i}" for i in range(len(inputs))]
        return arg_names
    except Exception:
        return [f"input{i}" for i in range(len(inputs))]


def create_forward_hook(module_names, file_prefix="debug_"):
    """Create a forward hook that logs module inputs, outputs, and weights.

    Args:
        module_names: Dict mapping module IDs to their hierarchical names.
        file_prefix: Prefix for log files.

    Returns:
        Forward hook function that logs to JSONL files.
    """

    def forward_hook(module, inputs, output):
        global_name = module_names.get(id(module), module.__class__.__name__)
        input_names = get_forward_arg_names(module, inputs)
        input_summary = {name: safe_convert(inp) for name, inp in zip(input_names, inputs)}
        output_summary = safe_convert(output)

        try:
            weight_summary = safe_convert(module.weight)
        except Exception:
            weight_summary = "NONE"

        log_entry = {
            "hook": "forward",
            "module": global_name,
            "inputs": input_summary,
            "output": output_summary,
            "weight": weight_summary,
        }

        # Determine the current GPU rank if in a distributed setup.
        rank, world_size = _get_rank_and_world_size()
        log_filename = f"{file_prefix}fwd_log_{world_size}_rank_{rank}.jsonl"  # JSON Lines format

        try:
            with open(log_filename, "a") as f:
                f.write(json.dumps(log_entry, indent=2) + "\n")
        except Exception as e:
            with open(log_filename, "a") as f:
                error_entry = {"module": global_name, "error": f"Serialization error: {str(e)}"}
                f.write(json.dumps(error_entry) + "\n")

    return forward_hook


def create_backward_hook(module_names, file_prefix="debug_"):
    """Create a backward hook that logs gradient information.

    Args:
        module_names: Dict mapping module IDs to their hierarchical names.
        file_prefix: Prefix for log files.

    Returns:
        Backward hook function that logs to JSONL files.
    """

    def backward_hook(module, grad_input, grad_output):
        global_name = module_names.get(id(module), module.__class__.__name__)
        grad_input_summary = safe_convert(grad_input)
        grad_output_summary = safe_convert(grad_output)

        log_entry = {
            "hook": "backward",
            "module": global_name,
            "grad_input": grad_input_summary,
            "grad_output": grad_output_summary,
        }

        rank, world_size = _get_rank_and_world_size()
        log_filename = f"{file_prefix}bwd_log_{world_size}_rank_{rank}.jsonl"

        try:
            with open(log_filename, "a") as f:
                f.write(json.dumps(log_entry, indent=2) + "\n")
        except Exception as e:
            with open(log_filename, "a") as f:
                error_entry = {"module": global_name, "error": f"Serialization error: {str(e)}"}
                f.write(json.dumps(error_entry) + "\n")

    return backward_hook


def register_hooks(model, file_prefix="debug_"):
    """Register both forward and backward hooks on every submodule of the model.

    A mapping from module id to hierarchical name is built so that each log entry
    contains a global name (e.g., "layer1.block.MLP") instead of just the class name.

    Args:
        model: The PyTorch model to register hooks on.
        file_prefix: Prefix for the log files.
    """
    # Build mapping: module id -> hierarchical name.
    module_names = {
        id(module): name if name != "" else module.__class__.__name__ for name, module in model.named_modules()
    }

    # Create hook functions that share the module names mapping.
    forward_hook_fn = create_forward_hook(module_names, file_prefix)
    backward_hook_fn = create_backward_hook(module_names, file_prefix)

    # Register both hooks on each module.
    for module in model.modules():
        module.register_forward_hook(forward_hook_fn)
        module.register_full_backward_hook(backward_hook_fn)
