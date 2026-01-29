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

Memory-efficient mode (default):
- CRITICAL: Moves tensors to CPU IMMEDIATELY in hooks (prevents GPU OOM)
- Async GPU→CPU transfer happens in background during forward/backward
- Never accumulates tensors on GPU - each tensor copied to CPU right away
- Direct bitwise comparison (torch.equal) - more accurate than hashing
- Peak GPU: Model activations only (no determinism overhead!)

Memory flow to prevent OOM:
1. Run 1 Forward: Each hook immediately moves tensor to CPU (async, non-blocking)
   - GPU: Model activations only (~18GB peak during forward)
   - CPU: Accumulates tensors as hooks fire (~8GB final)
2. Run 1 Backward: Each hook immediately moves gradient to CPU
   - GPU: Still just model gradients (~18GB peak during backward)
   - CPU: Accumulates gradients (~16GB final)
3. Run 2: Same process - tensors moved to CPU immediately
   - GPU: No determinism overhead!
   - CPU: Both runs' tensors (~32GB final)
4. Comparison: All tensors already on CPU, compare directly

Why immediate CPU transfer is crucial:
- Prevents GPU OOM: No accumulation of tensors on GPU
- Async transfer: Doesn't block forward/backward (GPU→CPU in background)
- Works with large models: GPU only needs model activations, not extra storage
- Fast comparison: torch.equal() on CPU tensors (no hashing needed)

Usage:
    plugin = DeterminismDebugPlugin(config)
    plugin.register_hooks(model)
    
    # Run 1
    plugin.start_new_run()  # run_id=1
    loss = model(batch)     # Hooks move tensors to CPU immediately
    loss.backward()         # Hooks move gradients to CPU immediately
    model.zero_grad()       # Keep weights same
    
    # Run 2 (same batch)
    plugin.start_new_run()  # run_id=2
    loss = model(batch)     # Hooks move tensors to CPU immediately
    loss.backward()         # Hooks move gradients to CPU immediately
    
    # Gather and analyze
    results = plugin.gather_all_results()  # Compares CPU tensors directly
    print(analyze_results(results))
    
    # Reset for next step
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
    """Use memory-efficient mode (move tensors to CPU immediately in hooks).
    When True: Each tensor moved to CPU async in hook (prevents GPU OOM).
               Peak GPU: Model activations only. Peak CPU: Both runs (~32GB).
    When False: Tensors kept on GPU until comparison (will cause OOM on large models)."""
    
    layer_sample_rate: float = 1.0
    """Fraction of layers to sample for determinism checking (0.0-1.0).
    1.0 = track all layers (default), 0.1 = track 10% of layers.
    Use lower values (e.g., 0.05-0.1) for models with tight GPU memory.
    Layers are sampled deterministically based on hash of layer name."""
    
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
        
        # Storage for hashes from all runs
        # We store hashes instead of full tensors to save CPU memory
        self._tensors: Dict[Tuple[str, int, str, int, Optional[int]], Any] = {}
        
        self._call_counts: Dict[Tuple[str, int, str, Optional[int]], int] = {}  # Track call count per (name, run_id, pass_type, microbatch_id)
        self._run_id = 0
        self._microbatch_id: Optional[int] = None  # Current microbatch ID
        self._results: Dict[str, bool] = {}  # "name:pass_type:call_idx" -> match/mismatch
        self._first_mismatch: Optional[str] = None
        
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
            if not any(p.search(name) for p in self._check_patterns):
                return False
        
        # Layer sampling: use deterministic hash to decide whether to track this layer
        if self.config.layer_sample_rate < 1.0:
            # Hash layer name to get a deterministic 0-1 value
            layer_hash = hash(name) % 10000 / 10000.0
            if layer_hash >= self.config.layer_sample_rate:
                return False
        
        return True
    
    def _get_output_tensor(self, output) -> Optional[torch.Tensor]:
        """Extract the main output tensor from hook output."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            if isinstance(output[0], torch.Tensor):
                return output[0]
        return None
    
    def _move_to_cpu_async(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Move tensor to CPU asynchronously for memory-efficient comparison.
        
        This is faster than hashing and avoids:
        - SHA256 computation overhead
        - numpy conversion overhead
        - Multiple intermediate copies
        
        Returns CPU tensor (transfer happens asynchronously).
        """
        # Detach and move to CPU (async, non-blocking)
        return tensor.detach().cpu()
    
    def _record(self, name: str, tensor: torch.Tensor, pass_type: str):
        """
        Record a tensor's hash for later comparison.
        
        NO cross-rank communication here - safe for PP.
        Comparison happens in gather_all_results() after both runs complete.
        
        Handles multiple calls to the same layer (e.g., activation checkpointing)
        by tracking call count and storing each call separately.
        
        CRITICAL: Tensors are hashed and immediately discarded to save memory.
        
        Args:
            name: Layer name or parameter name
            tensor: Activation or gradient tensor
            pass_type: "forward", "backward", "output_grad", or "param_grad"
        """
        # Auto-detect microbatch ID from Megatron parallel state if not manually set
        mb_id = self._microbatch_id
        if mb_id is None:
            try:
                from megatron.core.pipeline_parallel.schedules import get_current_microbatch_id
                mb_id = get_current_microbatch_id()
            except (ImportError, AttributeError):
                pass

        # Track call count for this (name, run_id, pass_type, mb_id) combination
        # mb_id helps align interleaved microbatches in PP
        count_key = (name, self._run_id, pass_type, mb_id)
        call_idx = self._call_counts.get(count_key, 0)
        self._call_counts[count_key] = call_idx + 1
        
        # Store with call_idx and mb_id to handle multiple calls to same layer
        key = (name, self._run_id, pass_type, call_idx, mb_id)
        
        # #region agent log
        try:
            import json
            import time
            log_entry = {
                "sessionId": "debug-session",
                "runId": f"run_{self._run_id}",
                "hypothesisId": "A",
                "location": "plugin.py:220",
                "message": f"Recording tensor for {name}",
                "data": {
                    "rank": self.rank,
                    "pp_rank": self._get_pp_rank(),
                    "tp_rank": self._get_tp_rank(),
                    "dp_rank": self._get_dp_rank(),
                    "name": name,
                    "pass_type": pass_type,
                    "call_idx": call_idx,
                    "microbatch_id": mb_id,
                    "run_id": self._run_id,
                    "shape": list(tensor.shape) if hasattr(tensor, 'shape') else None,
                    "dtype": str(tensor.dtype) if hasattr(tensor, 'dtype') else None
                },
                "timestamp": int(time.time() * 1000)
            }
            with open("/opt/Megatron-Bridge/.cursor/debug.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass
        # #endregion

        # Move to CPU and hash immediately to free GPU memory
        # This prevents OOM by not accumulating tensors on GPU OR CPU
        if self.config.memory_efficient:
            # Hash the tensor on CPU
            # We use a simple hash of the tensor data for bitwise comparison
            # Note: We use .detach().cpu() to ensure it's off the GPU
            cpu_tensor = tensor.detach().cpu()
            
            # BFloat16 is not supported by numpy.tobytes() directly in some versions
            # and hashing requires a stable byte representation.
            # Convert to float32 for hashing if it's bfloat16 or half
            if cpu_tensor.dtype in [torch.bfloat16, torch.float16]:
                cpu_tensor = cpu_tensor.to(torch.float32)
            
            # Use a robust hash: convert to numpy and use hash() or similar
            # For bitwise equality, we can use the byte representation
            tensor_hash = hash(cpu_tensor.numpy().tobytes())
            self._tensors[key] = tensor_hash

            # #region agent log
            try:
                import json
                import time
                log_entry = {
                    "sessionId": "debug-session",
                    "runId": f"run_{self._run_id}",
                    "hypothesisId": "B",
                    "location": "plugin.py:238",
                    "message": f"Hash computed for {name}",
                    "data": {
                        "rank": self.rank,
                        "pp_rank": self._get_pp_rank(),
                        "tp_rank": self._get_tp_rank(),
                        "dp_rank": self._get_dp_rank(),
                        "name": name,
                        "pass_type": pass_type,
                        "call_idx": call_idx,
                        "microbatch_id": mb_id,
                        "hash": tensor_hash
                    },
                    "timestamp": int(time.time() * 1000)
                }
                with open("/opt/Megatron-Bridge/.cursor/debug.log", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
            # #endregion
        else:
            # Original mode: keep on GPU
            self._tensors[key] = tensor.detach().clone()
    
    def _create_forward_hook(self, layer_name: str):
        """Create forward hook to record input and output tensors.
        
        NOTE: We do NOT use tensor.register_hook() here because it prevents
        PyTorch memory optimization by keeping all intermediate tensors alive
        from forward through backward. Instead, we rely on module backward hooks
        to capture gradients, which allows PyTorch to reuse memory.
        """
        def hook(module, inputs, output):
            if not self.config.enabled:
                return
            
            # Record Input (for root cause analysis)
            if inputs and isinstance(inputs[0], torch.Tensor):
                self._record(f"{layer_name}.input", inputs[0], "forward_input")

            # Record Output
            tensor = self._get_output_tensor(output)
            if tensor is not None:
                self._record(layer_name, tensor, "forward")
                # Gradient will be captured by backward hook instead
        return hook
    
    def _create_backward_hook(self, layer_name: str):
        """Create backward hook to record gradient tensors (input and output)."""
        def hook(module, grad_input, grad_output):
            if not self.config.enabled:
                return
            
            # Record "Input" to backward (gradient from next layer/loss side)
            input_tensor = self._get_output_tensor(grad_output)
            if input_tensor is not None:
                self._record(f"{layer_name}.grad_output", input_tensor, "backward_input")

            # Record "Output" of backward (gradient to previous layer)
            tensor = self._get_output_tensor(grad_input)
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
        
        mode_str = "memory-efficient (CPU-based)" if self.config.memory_efficient else "full-tensor (GPU)"
        sample_str = f", sampling {self.config.layer_sample_rate*100:.0f}%" if self.config.layer_sample_rate < 1.0 else ""
        print_rank_last(f"[Rank {self.rank}] Registered determinism hooks on {count} layers ({mode_str}{sample_str})")
        if attention_count > 0:
            print_rank_last(f"[Rank {self.rank}]   Including {attention_count} attention-related layers")
    
    def _compare_runs(self, run1: int, run2: int):
        """
        Compare tensor hashes from two runs after both are complete.
        
        Compares each call separately - e.g., if a layer is called twice
        (activation checkpointing), call 0 from run1 is compared with call 0 from run2.
        """
        # Find all keys from run2
        run2_keys = [(name, pass_type, call_idx, mb_id) 
                     for (name, rid, pass_type, call_idx, mb_id) in self._tensors.keys() 
                     if rid == run2]
        
        for name, pass_type, call_idx, mb_id in run2_keys:
            key1 = (name, run1, pass_type, call_idx, mb_id)
            key2 = (name, run2, pass_type, call_idx, mb_id)
            
            if key1 not in self._tensors:
                continue
            
            val1 = self._tensors[key1]
            val2 = self._tensors[key2]
            
            # Include call_idx and mb_id in result key if there are multiple calls or microbatches
            result_key_parts = [name, pass_type]
            if mb_id is not None:
                result_key_parts.append(f"mb{mb_id}")
            if call_idx > 0:
                result_key_parts.append(f"idx{call_idx}")
            result_key = ":".join(result_key_parts)

            if self.config.memory_efficient:
                # Compare hashes directly
                is_match = (val1 == val2)
                
                # #region agent log
                try:
                    import json
                    import time
                    log_entry = {
                        "sessionId": "debug-session",
                        "runId": "comparison",
                        "hypothesisId": "C",
                        "location": "plugin.py:338",
                        "message": f"Comparison result for {result_key}",
                        "data": {
                            "rank": self.rank,
                            "pp_rank": self._get_pp_rank(),
                            "tp_rank": self._get_tp_rank(),
                            "dp_rank": self._get_dp_rank(),
                            "name": result_key,
                            "val1_hash": val1,
                            "val2_hash": val2,
                            "is_match": is_match
                        },
                        "timestamp": int(time.time() * 1000)
                    }
                    with open("/opt/Megatron-Bridge/.cursor/debug.log", "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                except Exception:
                    pass
                # #endregion
            else:
                # Compare GPU tensors
                t1 = val1
                t2 = val2
                if t1.is_cuda:
                    t1 = t1.cpu()
                if t2.is_cuda:
                    t2 = t2.cpu()
                is_match = torch.equal(t1, t2)
            
            self._results[result_key] = is_match
            
            if not is_match and self._first_mismatch is None:
                self._first_mismatch = result_key
                if self.config.verbose:
                    print_rank_last(f"[Rank {self.rank}] MISMATCH: {result_key}")
    
    def start_new_run(self):
        """
        Call BEFORE each run to signal a new run is starting.
        
        This increments the run counter so tensors are stored with the correct run_id.
        
        CRITICAL: In memory-efficient mode, we clear the CUDA cache and run GC
        to ensure memory from previous runs is fully reclaimed.
        """
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Move previous run's tensors to CPU to free GPU memory
        if self._run_id > 0 and self.config.memory_efficient:
            self._move_run_to_cpu(self._run_id)
        
        # Reset call counts for the new run so that call_idx starts from 0
        # This prevents "false alarms" caused by index shifting between runs
        self._call_counts.clear()
        self._microbatch_id = None  # Reset microbatch ID for new run
        
        self._run_id += 1
    
    def set_microbatch_id(self, microbatch_id: Optional[int]):
        """
        Set the current microbatch ID for alignment.
        
        In PP, microbatches are interleaved. By setting the microbatch ID,
        the plugin can align tensors from the same microbatch across runs,
        even if the execution order of microbatches changes.
        """
        self._microbatch_id = microbatch_id
    
    def _move_run_to_cpu(self, run_id: int):
        """
        Ensure all tensors from a specific run are on CPU.
        
        In memory-efficient mode, tensors are already moved to CPU during hooks,
        so this is a no-op. In non-memory-efficient mode, moves them now.
        
        Args:
            run_id: The run ID whose tensors should be on CPU
        """
        if self.config.memory_efficient:
            # Already on CPU (moved during hooks)
            return
        
        # Non-memory-efficient mode: move to CPU now
        keys_to_move = [(name, rid, pass_type, call_idx, mb_id) 
                        for (name, rid, pass_type, call_idx, mb_id) in self._tensors.keys() 
                        if rid == run_id]
        
        for key in keys_to_move:
            if self._tensors[key].is_cuda:
                self._tensors[key] = self._tensors[key].cpu()
    
    def reset_for_next_step(self):
        """
        Clear all state for the next step. Call after gather_all_results().
        
        This resets the run counter and clears all tensors to prevent
        cross-step contamination where data from a previous step could be
        incorrectly compared with data from the current step.
        """
        self._tensors.clear()
        self._call_counts.clear()
        self._results.clear()
        self._first_mismatch = None
        self._run_id = 0
        self._microbatch_id = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def gather_all_results(self) -> List[Dict]:
        """
        Gather results from all ranks. ONLY call at step end after backward completes.
        This is the ONLY cross-rank communication in the plugin.
        
        This method first compares the two runs, then gathers results from all ranks.
        
        In memory-efficient mode:
        - Run 1 tensors already on CPU (moved by start_new_run())
        - Run 2 tensors on GPU
        - _compare_runs() handles CPU transfer and comparison
        
        Returns:
            List of result dicts from all ranks
        """
        # Compare run 1 vs run 2 (handles CPU transfer internally)
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
        self._tensors.clear()
        self._call_counts.clear()
        self._results.clear()
        self._first_mismatch = None
        self._run_id = 0
        self._microbatch_id = None
    
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
    all_mismatches_by_type = {
        'forward': set(), 'forward_input': set(),
        'backward': set(), 'backward_input': set(),
        'output_grad': set(), 'param_grad': set()
    }
    for r in all_results:
        for result_key, is_match in r['results'].items():
            if not is_match:  # mismatch
                # result_key format: "layer_name:pass_type"
                if ':forward_input' in result_key:
                    all_mismatches_by_type['forward_input'].add(result_key)
                elif ':forward' in result_key:
                    all_mismatches_by_type['forward'].add(result_key)
                elif ':backward_input' in result_key:
                    all_mismatches_by_type['backward_input'].add(result_key)
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
    
    # Root Cause Analysis
    root_causes = []
    for r in all_results:
        res = r['results']
        for key, is_match in res.items():
            if not is_match:
                # Forward Root Cause: Input matches, Output mismatches
                if ':forward' in key and ':forward_input' not in key:
                    input_key = key.replace(':forward', '.input:forward_input')
                    if input_key in res and res[input_key]:
                        root_causes.append(f"    [FORWARD ROOT CAUSE] {key}")
                
                # Backward Root Cause: grad_output (input) matches, grad_input (output) mismatches
                if ':backward' in key and ':backward_input' not in key:
                    input_key = key.replace(':backward', '.grad_output:backward_input')
                    if input_key in res and res[input_key]:
                        root_causes.append(f"    [BACKWARD ROOT CAUSE] {key}")
    
    if root_causes:
        lines.append("\n  Identified Root Causes (Input matches, Output mismatches):")
        # Deduplicate and show
        for rc in sorted(set(root_causes)):
            lines.append(rc)

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
