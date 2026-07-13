# DeepSeek-V4-Pro GB300 Performance Recipe

This example reproduces the full 61-layer DeepSeek-V4-Pro MXFP8 performance recipe on 256 GB300 GPUs. The configuration uses TP1/PP4/VPP4/EP64, global batch size 4096, full-iteration CUDA graphs, fused DSA, a cuteDSL fused grouped MLP, and fine-grained activation offloading.

## Prerequisites

- Megatron-Bridge `r0.5.0` with the DeepSeek-V4-Pro performance recipe.
- NeMo container `nvcr.io/nvidia/nemo:26.06.01`.
- Megatron-Core commit `9d46c924dce3818f2b5f894f7380712c780d1801` with the compatibility change below.
- A host environment containing `nemo_run`.

The NeMo container includes the required cuDNN frontend 1.24, cuteDSL 4.5.0, `fast_hadamard_transform`, and FlashMLA dependencies.

## Megatron-Core Compatibility Change

The container reports TE 2.16 but includes the clamped SwiGLU backport. Replace the TE version guard in `megatron/core/transformer/moe/experts.py` with a capability check:

```diff
 if self.config.activation_func == F.silu:
     if self.config.activation_func_clamp_value is not None:
-        if not is_te_min_version("2.17.0.dev0"):
-            return _unsupported("clamped SwiGLU needs TE >= 2.17.0.dev0")
         if not hasattr(te_ops, "ScaledClampedQGeGLU"):
             return _unsupported("clamped SwiGLU needs ScaledClampedQGeGLU")
+        if "glu_linear_offset" not in inspect.signature(
+            te_ops.ScaledClampedQGeGLU
+        ).parameters:
+            return _unsupported("clamped SwiGLU needs configurable glu_linear_offset")
```

## Launch

Set the site-specific paths and Slurm values:

```bash
export WORKSPACE=/shared/path/dsv4-pro-validation
export MBRIDGE_VENV=/shared/path/venvs/mbridge
export MCORE_DEV=/shared/path/Megatron-LM-9d46c924d
export SLURM_ACCOUNT=<account>
export SLURM_PARTITION=<gb300-partition>

DRY=1 bash examples/models/deepseek_v4_perf/launch_pro_perf_gb300.sh
bash examples/models/deepseek_v4_perf/launch_pro_perf_gb300.sh
```

Set `HF_TOKEN` if the DeepSeek-V4-Pro Hugging Face configuration is not already cached. The launcher submits 20 iterations by default and detaches after submission.

The validated run completed all 20 iterations with no skipped or NaN iterations and stabilized at approximately 21.6 seconds per iteration.
