# docker

This directory contains Dockerfiles and supporting scripts for building the Megatron-Bridge
container and the NeMo Framework (NeMo-FW) image stack.

| Image | Dockerfile | Purpose |
|---|---|---|
| megatron-bridge | `Dockerfile.ci` | Megatron-Bridge development and CI container |
| fw-base | `Dockerfile.fw_base` | CUDA / TRT-LLM / vLLM / DeepEP base layer |
| fw-final | `Dockerfile.fw_final` | NeMo, Export-Deploy, Evaluator, NeMo-Run on top of megatron-bridge |

The full NeMo-FW stack is built in order: **fw-base → megatron-bridge → fw-final**.

---

## Megatron-Bridge container

`Dockerfile.ci` builds the Megatron-Bridge development and CI container. It installs the
package and all dependencies using [uv](https://github.com/astral-sh/uv).
Run from the repository root:

```bash
docker build \
  -f docker/Dockerfile.ci \
  --target megatron_bridge \
  -t megatron-bridge:latest \
  .
```

| Argument | Description |
|---|---|
| `BASE_IMAGE` | Base container |
| `MCORE_TRIGGERED_TESTING` | When `true`, skips the uv lockfile check to allow testing against a different Megatron-LM version than the one pinned in the lockfile |
| `UV_CACHE_PRUNE_ARGS` | Extra arguments forwarded to `uv cache prune` after install |

---

## NeMo Framework image stack

### Step 1 — fw-base (`Dockerfile.fw_base`)

Builds the CUDA / TRT-LLM / vLLM / DeepEP base layer.
**Must be run from the repository root** — the build context must include `docker/common/` and `docker/patches/`.

Two modes are controlled by `FW_DEP_BUILDER` and `FW_BASE_FINAL`:

**With TRT-LLM (default):**

```bash
docker buildx build \
  -f docker/Dockerfile.fw_base \
  --target nemo_fw_base_final \
  --build-arg FW_DEP_BUILDER=trtllm_builder \
  --build-arg FW_BASE_FINAL=trtllm_install \
  --build-arg NEMO_FW_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3 \
  --build-arg TRT_LLM_COMMIT=v1.3.0rc4 \
  --build-arg VLLM_VERSION=v0.14.1 \
  -t fw-base:latest \
  .
```

**Without TRT-LLM (faster, for development):**

```bash
docker buildx build \
  -f docker/Dockerfile.fw_base \
  --target nemo_fw_base_final \
  --build-arg FW_DEP_BUILDER=base \
  --build-arg FW_BASE_FINAL=fw_toolkit_builder \
  --build-arg NEMO_FW_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3 \
  -t fw-base:latest \
  .
```

### Step 2 — megatron-bridge (`Dockerfile.ci`)

Built with the fw-base image as `BASE_IMAGE` (see [Megatron-Bridge container](#megatron-bridge-container)):

```bash
docker build \
  -f docker/Dockerfile.ci \
  --target megatron_bridge \
  --build-arg BASE_IMAGE=fw-base:latest \
  -t megatron-bridge:latest \
  .
```

### Step 3 — fw-final (`Dockerfile.fw_final`)

Installs NeMo, Export-Deploy, Evaluator, and NeMo-Run on top of the megatron-bridge image.

```bash
docker build \
  -f docker/Dockerfile.fw_final \
  --target nemo_fw_final \
  --build-arg NEMO_FW_FINAL_BASE_IMAGE=megatron-bridge:latest \
  --build-arg NEMO_COMMIT=<commit-sha> \
  --build-arg NEMO_EXPORT_DEPLOY_COMMIT=<commit-sha> \
  --build-arg NEMO_EVAL_COMMIT=<commit-sha> \
  --build-arg NEMO_RUN_COMMIT=<commit-sha> \
  -t fw-final:latest \
  .
```

---

## Build arguments reference

### `Dockerfile.fw_base`

| Argument | Description |
|---|---|
| `NEMO_FW_BASE_IMAGE` | Base PyTorch container |
| `FW_DEP_BUILDER` | Stage used as the `fw_dep_builder` base. `trtllm_builder` to include TRT-LLM, `base` to skip it |
| `FW_BASE_FINAL` | Output stage. `trtllm_install` (with TRT-LLM) or `fw_toolkit_builder` (without) |
| `FW_NETWORK_LAYER` | Stage used as the `fw_toolkit_builder` base. `aws_ofi_builder` (default) to reinstall EFA and build AWS-OFI-NCCL from source, `fw_dep_builder` to skip it |
| `UV_VERSION` | uv version to install |
| `VLLM_VERSION` | vLLM git tag to build |
| `TRT_LLM_COMMIT` | TensorRT-LLM git commit or tag |
| `TRT_LLM_VERSION` | TensorRT-LLM version string embedded as an image environment variable |
| `TRT_VER` | TensorRT version for the TRT-LLM install scripts |
| `CUDA_VER` | CUDA version for the TRT-LLM install scripts |
| `CUDNN_VER` | cuDNN version for the TRT-LLM install scripts |
| `NCCL_VER` | NCCL version for the TRT-LLM install scripts |
| `CUBLAS_VER` | cuBLAS version for the TRT-LLM install scripts |
| `NVRTC_VER` | NVRTC version for the TRT-LLM install scripts |
| `INSTALL_DEEPEP` | Set to `True` to build and install DeepEP and nvshmem |
| `DEEPEP_COMMIT` | DeepEP git commit SHA |
| `REINSTALL_NSYS` | Set to `True` to reinstall Nsight Systems from the NVIDIA apt repo |
| `NSYS_VERSION` | Nsight Systems version (e.g. `2026.1.0.1085`) |
| `REINSTALL_CUDNN` | Set to `True` to reinstall cuDNN from the NVIDIA apt repo |
| `CUDNN_VERSION` | cuDNN apt version (e.g. `9.18.1.3-1`) |
| `REINSTALL_NCCL` | Set to `True` to reinstall NCCL from the NVIDIA apt repo |
| `NCCL_VERSION` | NCCL apt version (e.g. `2.28.9-1+cuda13.0`) |
| `REINSTALL_CUBLAS` | Set to `True` to reinstall cuBLAS and cuBLASLt from the NVIDIA apt repo |
| `CUBLAS_VERSION` | cuBLAS apt version (e.g. `13.2.1.1-1`) |

### `Dockerfile.ci`

| Argument | Description |
|---|---|
| `BASE_IMAGE` | Base container; set to the fw-base image when building the full stack |
| `MCORE_TRIGGERED_TESTING` | Skip uv lockfile check for cross-version Megatron-LM testing |
| `UV_CACHE_PRUNE_ARGS` | Extra arguments for `uv cache prune` |

### `Dockerfile.fw_final`

| Argument | Description |
|---|---|
| `NEMO_FW_FINAL_BASE_IMAGE` | Base image; must be a megatron-bridge image |
| `NEMO_COMMIT` | NeMo git commit SHA |
| `NEMO_EXPORT_DEPLOY_COMMIT` | NeMo Export-Deploy git commit SHA |
| `NEMO_EVAL_COMMIT` | NeMo Evaluator git commit SHA |
| `NEMO_RUN_COMMIT` | NeMo Run git commit SHA |

---

## Supporting files

| File | Description |
|---|---|
| `common/fw_pyproject.toml` | uv project config for the NeMo-FW virtual environment (copied into the fw-final container as `pyproject.toml`) |
| `common/install_cublas.sh` | Reinstall cuBLAS and cuBLASLt from the public NVIDIA CUDA apt repo |
| `common/install_nccl.sh` | Reinstall NCCL from the public NVIDIA CUDA apt repo |
| `common/install_cudnn.sh` | Reinstall cuDNN from the public NVIDIA CUDA apt repo |
| `common/install_nsys.sh` | Reinstall Nsight Systems from the public NVIDIA CUDA apt repo |
| `patches/deepep.patch` | Patch applied to DeepEP during fw-base build |
| `patches/vllm.patch` | Patch applied to vLLM after install in fw-base |
