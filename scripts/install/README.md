# Install MBridge 0.7.0 in the vLLM base container

This experiment installs MBridge 0.7.0 and its pinned Megatron-LM submodule on
Python 3.14.4, with Transformers downgraded to 5.12.1. Package installation and
imports passed with the overrides below. GPU execution has not been tested.

## Container and expected dependencies

Use **`dl/dgx/vllm:main-py3.83861681-base`**, resolved through your organization's
container registry. The registry hostname is intentionally omitted. The tested
image is Linux x86_64; the script downloads x86_64 CUDA profiler headers.

| Component | Expected image version |
| --- | --- |
| Python | 3.14.4, `/usr/bin/python3` |
| CUDA toolkit | 13.5 |
| cuDNN backend | 9.29.0.29 |
| PyTorch | `2.15.0a0+fd23299d96.nvinternal.main.83205032` |
| Torchvision | `0.30.0a0+6da25ff8.nvinternal.main.83205032` |
| Triton | `3.9.0+git20469696.nv26.11` |
| vLLM | `0.34.0+e51c829d.nvinternal.main.83861681` |
| NumPy | 2.5.5 |
| FlashInfer Python, cubin and JIT caches | `0.7.5+24ac7794.nvinternal.main.cu.83753500` |
| cuDNN frontend | 1.33.0 |

| Installed component | Tested version or source |
| --- | --- |
| MBridge | 0.7.0, with `requires-python = ">=3.12"` |
| Megatron-Core | `0.20.0+503867317`, submodule commit `5038673171e463b1486aca4e85dc36fcf6ffc712` |
| Transformers | 5.12.1 |
| Tokenizers | 0.22.2 |
| Hugging Face Hub | 1.33.0 |
| Transformer Engine | `2.20.2+44f31bbe`, commit `44f31bbeaf99bd30a67b928000d4b980026debd2` |
| NVIDIA Resiliency Extension | 0.7.0 |
| Multi-Storage Client | 0.51.0, built from source commit `6c82181c1b32cfbdc5cf0a40a0dd4ff035927d66` |

The image must provide `git`, `uv`, `curl`, `tar`, `sha256sum`, and CUDA/C++ build
tools. The script installs Python build tools into `.venv`. If Cargo is absent,
it installs Cargo and Rust with `apt-get`, which requires root. Preinstall those
tools to run the script as a non-root user.

## Run

Inside the image, enter this patched checkout and run:

```bash
NVTE_CUDA_ARCHS=90 NVTE_BUILD_MAX_JOBS=12 MAX_JOBS=12 \
  bash scripts/install/vllm.sh .
```

`90` builds Transformer Engine for Hopper/SM90, the architecture used for the
installation probe. Set `NVTE_CUDA_ARCHS` for your target GPU, or omit it to use
TE's default architectures. Compilation may take tens of minutes.

The script creates `.venv` with system site packages and uses the image's Python.
It installs both projects as editable packages with `uv pip install`. Use
`source .venv/bin/activate` for the resulting environment.

An optional `TRANSFORMER_ENGINE_WHEEL=/path/to/wheel.whl` reuses an already built
TE wheel. It must match the TE commit above, Python 3.14, this image's PyTorch/CUDA
stack, and the intended GPU architecture. Validation reused the SM90 wheel
compiled earlier in the same image with the build settings in this script.

`MBRIDGE_TRANSFORMERS_VERSION` defaults to `5.12.1`. That setting also pins
Tokenizers 0.22.2 and HF Hub 1.33.0. Other values let those dependencies resolve
normally. A successful import alone does not validate all model implementations.

## Overrides and build adjustments

- Preserve the image's PyTorch, Torchvision, Triton, vLLM, NumPy, FlashInfer family,
  and cuDNN frontend. Exclude them from uv resolution and verify their versions
  after installation. uv does not resolve against inherited system packages.
- Override Transformers to 5.12.1. This is below MBridge's `>=5.15,<5.18` and
  vLLM's `>=5.16.1,<5.23.0` requirements; the script deliberately bypasses those
  bounds. It does not establish a dependency-consistent vLLM training environment.
- Install Multi-Storage Client 0.51.0 from its source subdirectory because that
  release has no CPython 3.14 wheel.
- Include ONNX on Python 3.14. The release's Python marker excludes it, but TE
  imports ONNX through ONNX Script.
- Set `TMS_CUDA_MAJOR=13` for the Torch Memory Saver source build.
- Set TE's `CUDNN_SKIP_PRECOMPILED_LINK=ON`: the image lacks cuDNN's precompiled
  engines library. Add NVTX 3.4.0 headers and checksum-verified CUDA profiler API
  13.0.85 headers; provide them to both TE's CMake and PyTorch extension builds.
- Retain a persistent TE CMake build directory under `.venv/mbridge-build`.

These overrides live in temporary configuration files. They do not rewrite
`pyproject.toml` or `uv.lock`. Normal project locking remains scoped to Python
3.12; this Python 3.14 experiment uses the separate pip installation path.

## Observed validation and limitations

The installation probe passed imports of `torch`, `transformers`, `flashinfer`,
`transformer_engine.pytorch`, `megatron.core`, `megatron.bridge`, and `vllm`.
The preserved image package versions matched after installation.

MBridge warns that Muse Glimmer classes are unavailable with Transformers 5.12.1.
Transformers also logs model-output docstring errors during registration, without
failing the import. The base image itself lacks `vllm._C`; its top-level import
passing does not verify native vLLM inference. No GPU training, inference, or
model-conversion execution has been validated by this experiment.
