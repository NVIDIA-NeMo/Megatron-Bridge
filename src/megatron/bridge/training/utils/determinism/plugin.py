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

"""
Determinism Debug Plugin for Megatron-Bridge

Compares LOCAL tensors across repeated runs to detect non-determinism.
- Run forward+backward twice with same input
- Compare tensors between run 1 and run 2 (bitwise equality)
- Gather results at step end (safe sync point, no PP deadlock)
- Tracks forward activations, backward gradients, and parameter gradients

Usage:
    plugin = DeterminismDebugPlugin(config)
    plugin.register_hooks(model)
    
    # Run 1
    plugin.start_new_run()  # Increments run_id to 1
    loss = model(batch)
    loss.backward()
    model.zero_grad()  # Keep weights same
    
    # Run 2 (same batch)
    plugin.start_new_run()  # Increments run_id to 2
    loss = model(batch)
    loss.backward()
    
    # Gather and analyze - this compares run 1 vs run 2
    results = plugin.gather_all_results()
    print(analyze_results(results))
    
    # Reset for next step (important to avoid cross-step contamination!)
    plugin.reset_for_next_step()
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.bridge.utils.common_utils import print_rank_last


@dataclass
class DeterminismConfig:
    """Configuration for determinism verification."""
    
    enabled: bool = True
    """Enable determinism checking."""
    
    check_layers: Optional[List[str]] = None
    """Layer name patterns (regex) to check. If None, checks all layers."""
    
    layers_to_skip: List[str] = field(default_factory=lambda: [
        r".*dropout.*",
        r".*Dropout.*",
    ])
    """Regex patterns for layers to skip."""
    
    verbose: bool = False
    """Print detailed logs."""
    
    check_backward: bool = True
    """Enable backward activation tracking."""
    
    check_param_grad: bool = True
    """Enable parameter gradient tracking."""
    
    memory_efficient: bool = True
    """Use memory-efficient mode (store hashes instead of full tensors).
    When True: Uses SHA256 hashes (~32 bytes per tensor) instead of storing full tensors.
    When False: Stores full tensors on CPU (can use ~10GB+ for large models)."""
    
    output_dir: str = "./determinism_logs"
    """Directory to save analysis results."""


class DeterminismDebugPlugin:
    """
    Plugin for determinism verification via forward/backward hooks.
    
    Compares local tensors across repeated runs. PP-safe because it does
    NO cross-rank communication in hooks - only at step end via gather_all_results().
    """
    
    def __init__(self, config: Optional[DeterminismConfig] = None, **kwargs):
        if config is None:
            config = DeterminismConfig(**kwargs)
        self.config = config
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if self.config.memory_efficient:
            # Memory-efficient mode: store hashes (32 bytes) instead of tensors
            self._tensor_hashes: Dict[Tuple[str, int, str, int], bytes] = {}
        else:
            # Original mode: store full tensors on CPU
            self._tensors: Dict[Tuple[str, int, str, int], torch.Tensor] = {}
        
        self._call_counts: Dict[Tuple[str, int, str], int] = {}  # Track call count per (name, run_id, pass_type)
        self._run_id = 0
        self._results: Dict[str, bool] = {}  # "name:pass_type:call_idx" -> match/mismatch
        self._first_mismatch: Optional[str] = None
        
        # Storage for gradient hook handles to prevent garbage collection
        self._grad_hook_handles: List = []
        
        # Compile patterns
        self._skip_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.layers_to_skip]
        self._check_patterns = None
        if self.config.check_layers:
            self._check_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.check_layers]
    
    def _should_check(self, name: str) -> bool:
        """Check if a layer should be monitored."""
        if not self.config.enabled:
            return False
        for p in self._skip_patterns:
            if p.search(name):
                return False
        if self._check_patterns:
            return any(p.search(name) for p in self._check_patterns)
        return True
    
    def _get_output_tensor(self, output) -> Optional[torch.Tensor]:
        """Extract the main output tensor from hook output."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            if isinstance(output[0], torch.Tensor):
                return output[0]
        return None
    
    def _compute_hash(self, tensor: torch.Tensor) -> bytes:
        """Compute SHA256 hash of a tensor for memory-efficient comparison."""
        import hashlib
        # Ensure tensor is contiguous and on CPU for consistent hashing
        tensor_cpu = tensor.detach().cpu().contiguous()
        
        # Convert to float32 if bfloat16 (NumPy doesn't support bfloat16)
        # This preserves determinism since bfloat16 -> float32 is exact
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_cpu = tensor_cpu.to(torch.float32)
        
        return hashlib.sha256(tensor_cpu.numpy().tobytes()).digest()
    
    def _record(self, name: str, tensor: torch.Tensor, pass_type: str):
        """
        Record a tensor for later comparison.
        
        NO cross-rank communication here - safe for PP.
        Comparison happens in gather_all_results() after both runs complete.
        
        Handles multiple calls to the same layer (e.g., activation checkpointing)
        by tracking call count and storing each call separately.
        
        In memory-efficient mode, stores only SHA256 hash (~32 bytes) instead of full tensor.
        
        Args:
            name: Layer name or parameter name
            tensor: Activation or gradient tensor
            pass_type: "forward", "backward", "output_grad", or "param_grad"
        """
        # Track call count for this (name, run_id, pass_type) combination
        count_key = (name, self._run_id, pass_type)
        call_idx = self._call_counts.get(count_key, 0)
        self._call_counts[count_key] = call_idx + 1
        
        # Store with call_idx to handle multiple calls to same layer
        key = (name, self._run_id, pass_type, call_idx)
        
        if self.config.memory_efficient:
            # Store hash instead of full tensor (~32 bytes vs potentially GBs)
            self._tensor_hashes[key] = self._compute_hash(tensor)
        else:
            # Original mode: store full tensor on CPU
            self._tensors[key] = tensor.detach().clone().cpu()
    
    def _create_forward_hook(self, layer_name: str):
        """Create forward hook to record output tensor."""
        def hook(module, inputs, output):
            if not self.config.enabled:
                return
            tensor = self._get_output_tensor(output)
            if tensor is not None:
                self._record(layer_name, tensor, "forward")
                
                # Also register a hook to check the gradient of this tensor after backward
                # This will catch non-deterministic backward passes in autograd Functions
                if self.config.check_backward and tensor.requires_grad:
                    def grad_hook(grad):
                        if grad is not None:
                            self._record(layer_name, grad, "output_grad")
                        return None  # Don't modify the gradient
                    
                    # Store the handle to prevent garbage collection
                    handle = tensor.register_hook(grad_hook)
                    self._grad_hook_handles.append(handle)
        return hook
    
    def _create_backward_hook(self, layer_name: str):
        """Create backward hook to record gradient tensor."""
        def hook(module, grad_input, grad_output):
            if not self.config.enabled:
                return
            tensor = self._get_output_tensor(grad_output)
            if tensor is not None:
                self._record(layer_name, tensor, "backward")
        return hook
    
    def _create_param_grad_hook(self, param_name: str):
        """Create parameter gradient hook."""
        def hook(param):
            if not self.config.enabled or param.grad is None:
                return
            self._record(param_name, param.grad, "param_grad")
        return hook
    
    def register_hooks(self, model: torch.nn.Module):
        """
        Register hooks on model layers for cross-run comparison.
        
        Args:
            model: The model to register hooks on
        """
        count = 0
        attention_count = 0
        attention_keywords = ['attention', 'attn', 'self_attention', 'core_attention', 'qkv', 'dot_product']
        
        for name, module in model.named_modules():
            if name and self._should_check(name):
                module.register_forward_hook(self._create_forward_hook(name))
                
                if self.config.check_backward:
                    module.register_full_backward_hook(self._create_backward_hook(name))
                
                count += 1
                
                # Track attention layers
                if any(kw in name.lower() for kw in attention_keywords):
                    attention_count += 1
        
        # Parameter gradient hooks
        if self.config.check_param_grad:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._create_param_grad_hook(name))
        
        mode_str = "memory-efficient (hash-based)" if self.config.memory_efficient else "full-tensor"
        print_rank_last(f"[Rank {self.rank}] Registered determinism hooks on {count} layers ({mode_str})")
        if attention_count > 0:
            print_rank_last(f"[Rank {self.rank}]   Including {attention_count} attention-related layers")
    
    def _compare_runs(self, run1: int, run2: int):
        """
        Compare tensors/hashes from two runs after both are complete.
        Uses bitwise equality (torch.equal for tensors, == for hashes).
        
        Compares each call separately - e.g., if a layer is called twice
        (activation checkpointing), call 0 from run1 is compared with call 0 from run2.
        """
        if self.config.memory_efficient:
            # Compare hashes
            run2_keys = [(name, pass_type, call_idx) 
                         for (name, rid, pass_type, call_idx) in self._tensor_hashes.keys() 
                         if rid == run2]
            
            for name, pass_type, call_idx in run2_keys:
                key1 = (name, run1, pass_type, call_idx)
                key2 = (name, run2, pass_type, call_idx)
                
                if key1 not in self._tensor_hashes:
                    continue
                
                hash1 = self._tensor_hashes[key1]
                hash2 = self._tensor_hashes[key2]
                
                # Hash equality means tensors are bitwise identical
                is_match = (hash1 == hash2)
                
                # Include call_idx in result key if there are multiple calls
                if call_idx > 0:
                    result_key = f"{name}:{pass_type}:{call_idx}"
                else:
                    result_key = f"{name}:{pass_type}"
                self._results[result_key] = is_match
                
                if not is_match and self._first_mismatch is None:
                    self._first_mismatch = result_key
                    if self.config.verbose:
                        print_rank_last(f"[Rank {self.rank}] MISMATCH: {result_key}")
        else:
            # Original mode: compare full tensors
            run2_keys = [(name, pass_type, call_idx) 
                         for (name, rid, pass_type, call_idx) in self._tensors.keys() 
                         if rid == run2]
            
            for name, pass_type, call_idx in run2_keys:
                key1 = (name, run1, pass_type, call_idx)
                key2 = (name, run2, pass_type, call_idx)
                
                if key1 not in self._tensors:
                    continue
                
                t1 = self._tensors[key1]
                t2 = self._tensors[key2]
                
                # Bitwise equality - determinism means identical results
                is_match = torch.equal(t1, t2)
                
                # Include call_idx in result key if there are multiple calls
                if call_idx > 0:
                    result_key = f"{name}:{pass_type}:{call_idx}"
                else:
                    result_key = f"{name}:{pass_type}"
                self._results[result_key] = is_match
                
                if not is_match and self._first_mismatch is None:
                    self._first_mismatch = result_key
                    if self.config.verbose:
                        print_rank_last(f"[Rank {self.rank}] MISMATCH: {result_key}")
    
    def start_new_run(self):
        """
        Call BEFORE each run to signal a new run is starting.
        
        This increments the run counter so tensors are stored with the correct run_id.
        The actual comparison happens in gather_all_results() after both runs complete.
        """
        self._run_id += 1
    
    def reset_for_next_step(self):
        """
        Clear all state for the next step. Call after gather_all_results().
        
        This resets the run counter and clears all tensors/hashes to prevent
        cross-step contamination where data from a previous step could be
        incorrectly compared with data from the current step.
        """
        if self.config.memory_efficient:
            self._tensor_hashes.clear()
        else:
            self._tensors.clear()
        self._call_counts.clear()
        self._results.clear()
        self._first_mismatch = None
        self._grad_hook_handles.clear()
        self._run_id = 0
    
    def gather_all_results(self) -> List[Dict]:
        """
        Gather results from all ranks. ONLY call at step end after backward completes.
        This is the ONLY cross-rank communication in the plugin.
        
        This method first compares the two runs, then gathers results from all ranks.
        
        Returns:
            List of result dicts from all ranks
        """
        # Compare run 1 vs run 2
        if self._run_id >= 2:
            self._compare_runs(self._run_id - 1, self._run_id)
        
        local_result = {
            'rank': self.rank,
            'pp_rank': self._get_pp_rank(),
            'tp_rank': self._get_tp_rank(),
            'dp_rank': self._get_dp_rank(),
            'results': self._results.copy(),
            'first_mismatch': self._first_mismatch,
            'num_mismatches': sum(1 for v in self._results.values() if not v),
        }
        
        if not dist.is_initialized():
            return [local_result]
        
        all_results = [None] * dist.get_world_size()
        dist.all_gather_object(all_results, local_result)
        return all_results
    
    def _get_pp_rank(self) -> int:
        """Get pipeline parallel rank for logging."""
        try:
            from megatron.core import parallel_state
            return parallel_state.get_pipeline_model_parallel_rank()
        except:
            return 0
    
    def _get_tp_rank(self) -> int:
        """Get tensor parallel rank for logging."""
        try:
            from megatron.core import parallel_state
            return parallel_state.get_tensor_model_parallel_rank()
        except:
            return 0
    
    def _get_dp_rank(self) -> int:
        """Get data parallel rank for logging."""
        try:
            from megatron.core import parallel_state
            return parallel_state.get_data_parallel_rank()
        except:
            return 0
    
    def reset(self):
        """Reset all state."""
        if self.config.memory_efficient:
            self._tensor_hashes.clear()
        else:
            self._tensors.clear()
        self._call_counts.clear()
        self._results.clear()
        self._first_mismatch = None
        self._grad_hook_handles.clear()
        self._run_id = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of determinism check results."""
        return {
            "rank": self.rank,
            "run_id": self._run_id,
            "first_mismatch": self._first_mismatch,
            "num_checked": len(self._results),
            "num_mismatches": sum(1 for v in self._results.values() if not v),
            "results": self._results,
        }
    
    def print_summary(self):
        """Print summary of determinism check results."""
        s = self.get_summary()
        print_rank_last("\n" + "=" * 50)
        print_rank_last(f"Determinism Summary (Rank {self.rank})")
        print_rank_last("=" * 50)
        print_rank_last(f"Runs compared: {s['run_id']}")
        print_rank_last(f"Layers checked: {s['num_checked']}")
        print_rank_last(f"Mismatches: {s['num_mismatches']}")
        if s['first_mismatch']:
            print_rank_last(f"First mismatch: {s['first_mismatch']}")
        print_rank_last("=" * 50 + "\n")
    
    # ========== MODEL PROVIDER HOOKS ==========
    
    def pre_wrap_hook(self, model: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """Hook for model provider pre-wrap."""
        for m in model:
            self.register_hooks(m)
        return model
    
    def post_wrap_hook(self, model: List[torch.nn.Module]) -> List[torch.nn.Module]:
        """Hook for model provider post-wrap."""
        for m in model:
            self.register_hooks(m.module if hasattr(m, 'module') else m)
        return model


def analyze_results(all_results: List[Dict]) -> str:
    """
    Analyze gathered results from all ranks to identify patterns.
    
    Args:
        all_results: List of result dicts from gather_all_results()
        
    Returns:
        Analysis report string
    """
    mismatches = [(r['rank'], r['pp_rank'], r['first_mismatch']) 
                  for r in all_results if r['first_mismatch']]
    
    if not mismatches:
        return "All ranks deterministic across runs"
    
    # Collect ALL mismatches from all ranks (not just first)
    all_mismatches_by_type = {'forward': set(), 'backward': set(), 'output_grad': set(), 'param_grad': set()}
    for r in all_results:
        for result_key, is_match in r['results'].items():
            if not is_match:  # mismatch
                # result_key format: "layer_name:pass_type"
                if ':forward' in result_key:
                    all_mismatches_by_type['forward'].add(result_key)
                elif ':backward' in result_key:
                    all_mismatches_by_type['backward'].add(result_key)
                elif ':output_grad' in result_key:
                    all_mismatches_by_type['output_grad'].add(result_key)
                elif ':param_grad' in result_key:
                    all_mismatches_by_type['param_grad'].add(result_key)
    
    # Group by layer (including pass type) for first mismatch analysis
    by_layer = {}
    for rank, pp, layer_and_pass in mismatches:
        by_layer.setdefault(layer_and_pass, []).append((rank, pp))
    
    lines = ["Non-determinism detected:"]
    
    # Show statistics
    total_checks = sum(len(r['results']) for r in all_results) // len(all_results)  # avg per rank
    lines.append(f"\n  Checks performed: ~{total_checks} per rank")
    lines.append(f"  Total unique mismatches: {sum(len(v) for v in all_mismatches_by_type.values())}")
    
    # Separate forward, backward, and param_grad mismatches
    if all_mismatches_by_type['forward']:
        lines.append(f"\n  Forward pass mismatches ({len(all_mismatches_by_type['forward'])} unique):")
        for layer_pass in sorted(all_mismatches_by_type['forward'])[:5]:  # show first 5
            layer_name = layer_pass.replace(':forward', '')
            lines.append(f"    {layer_name}")
        if len(all_mismatches_by_type['forward']) > 5:
            lines.append(f"    ... and {len(all_mismatches_by_type['forward']) - 5} more")
    
    if all_mismatches_by_type['backward']:
        lines.append(f"\n  Backward/Module gradient mismatches ({len(all_mismatches_by_type['backward'])} unique):")
        for layer_pass in sorted(all_mismatches_by_type['backward'])[:5]:
            layer_name = layer_pass.replace(':backward', '')
            lines.append(f"    {layer_name}")
        if len(all_mismatches_by_type['backward']) > 5:
            lines.append(f"    ... and {len(all_mismatches_by_type['backward']) - 5} more")
    
    if all_mismatches_by_type['output_grad']:
        lines.append(f"\n  Output tensor gradient mismatches ({len(all_mismatches_by_type['output_grad'])} unique):")
        lines.append(f"    (These capture autograd.Function backward passes like Flash Attention)")
        for layer_pass in sorted(all_mismatches_by_type['output_grad'])[:5]:
            layer_name = layer_pass.replace(':output_grad', '')
            lines.append(f"    {layer_name}")
        if len(all_mismatches_by_type['output_grad']) > 5:
            lines.append(f"    ... and {len(all_mismatches_by_type['output_grad']) - 5} more")
    
    if all_mismatches_by_type['param_grad']:
        lines.append(f"\n  Parameter gradient mismatches ({len(all_mismatches_by_type['param_grad'])} unique):")
        for layer_pass in sorted(all_mismatches_by_type['param_grad'])[:5]:
            layer_name = layer_pass.replace(':param_grad', '')
            lines.append(f"    {layer_name}")
        if len(all_mismatches_by_type['param_grad']) > 5:
            lines.append(f"    ... and {len(all_mismatches_by_type['param_grad']) - 5} more")
    
    # Pattern detection with attention-specific logic
    first_mismatch = mismatches[0][2]
    lines.append("\n  First mismatch detected:")
    lines.append(f"    {first_mismatch}")
    
    # Determine pass type of first mismatch
    first_pass_type = None
    if ':forward' in first_mismatch:
        first_pass_type = 'forward'
    elif ':backward' in first_mismatch:
        first_pass_type = 'backward'
    elif ':output_grad' in first_mismatch:
        first_pass_type = 'output_grad'
    elif ':param_grad' in first_mismatch:
        first_pass_type = 'param_grad'
    
    lines.append("\n  Analysis:")
    
    # Add propagation explanation if both forward and backward have many mismatches
    if len(all_mismatches_by_type['forward']) > 10 and len(all_mismatches_by_type['backward']) > 10:
        lines.append("    NOTE: Non-determinism propagates through the network")
        lines.append("    → First mismatch is the ROOT CAUSE")
        lines.append("    → All subsequent layers diverge in both forward and backward")
        lines.append("")
    
    # Check if attention-related
    attention_keywords = ['attention', 'attn', 'self_attention', 'core_attention', 'qkv', 'dot_product']
    is_attention_related = any(kw in first_mismatch.lower() for kw in attention_keywords)
    
    if len(by_layer) == 1 and len(mismatches) == len(all_results):
        lines.append(f"    ALL ranks fail at same point")
        
        if is_attention_related:
            if first_pass_type == 'backward' or first_pass_type == 'output_grad':
                lines.append("    → Attention backward non-determinism detected")
                lines.append("    → Flash Attention backward is known to be non-deterministic")
                lines.append("    → Solution: Use fused attention or disable flash attention")
            else:
                lines.append("    → Attention layer non-determinism detected")
                lines.append("    → Check attention backend settings")
        elif first_pass_type == 'forward' and any(kw in first_mismatch.lower() for kw in ['layernorm', 'norm']):
            if all_mismatches_by_type['backward'] or all_mismatches_by_type['output_grad']:
                lines.append("    → Gradient mismatches also detected (likely root cause)")
                # Find which attention layers have gradient mismatches
                attn_grad_mismatches = [m for m in (all_mismatches_by_type['backward'] | all_mismatches_by_type['output_grad']) 
                                       if any(kw in m.lower() for kw in attention_keywords)]
                if attn_grad_mismatches:
                    lines.append(f"    → Attention gradient mismatches found: {len(attn_grad_mismatches)}")
                    lines.append("    → Flash Attention backward is likely the root cause")
            else:
                lines.append("    → Forward-only mismatch at normalization layer")
                lines.append("    → Could be caused by:")
                lines.append("      • Non-deterministic operations in previous layers")
                lines.append("      • Residual connections accumulating small differences")
                lines.append("      • Check if attention layers have any mismatches")
        elif first_pass_type == 'backward' or first_pass_type == 'output_grad':
            lines.append("    → Backward pass non-determinism detected")
            lines.append("    → Check for non-deterministic gradient computations")
        else:
            lines.append("    → Global non-determinism detected")
            lines.append("    → Check: CUBLAS_WORKSPACE_CONFIG, NVTE_ALLOW_NONDETERMINISTIC_ALGO")
    elif len(set(pp for _, pp, _ in mismatches)) == 1:
        pp_stage = mismatches[0][1]
        lines.append(f"    Only PP stage {pp_stage} affected")
        lines.append("    → Pipeline-specific issue in that stage")
    elif len(by_layer) > 1:
        lines.append(f"    Multiple points affected ({len(by_layer)} unique)")
        lines.append("    → First mismatch is likely the root cause, rest is propagation")
    
    return "\n".join(lines)
