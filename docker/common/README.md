# NeMo Framework Docker Container

## Repository Overview

- Megatron-Bridge (`/opt/Megatron-Bridge/`)
- Megatron-LM (`/opt/Megatron-Bridge/3rdparty/Megatron-LM/`)
- Evaluator (`/opt/Evaluator/`)
- Export-Deploy (`/opt/Export-Deploy/`)
- NeMo (`/opt/NeMo/`)
- Run (`/opt/Run/`)

---

## Installed Packages

Pip installed packages:

- DeepEP
- vLLM
- TRT-LLM

Execute `pip list` to see full list of installed packages via pip.

Uv virtualenv installed packages (/opt/venv/)

- TransformerEngine
- nvidia-resiliency
- Model-Optimizer

Execute `uv pip list` to see full list of installed packages. Packages installed in /opt/venv take precedence over pip installed packages.

See `/opt/NeMo-FW/pyproject.toml` for additonal uv configurations.

---

## Development

### Mounting and syncing local repository into the container

Local working directories can be mounted via docker run:

```bash
docker run -v <local-folder-path>:<container-folder-path> <container-image>
```

#### 1 Overwrite existing directory

1. Define the `<container-folder-path>` to correspond to the path defined in Repository Overview. (ie. `/my-path/Megatron-Bridge/:/opt/Megatron-Bridge/`)

2. `cd /opt/NeMo-FW` and run `uv sync --no-cache-dir --all-groups --inexact`

3. Local development directory is synced to run in the container

#### 2 Mount directory to a different path

1. Modify `/opt/NeMo-FW/pyproject.toml` sections `tool.uv.sources` and `tool.uv; override-dependencies` to reflect new path. (ie. `"megatron-bridge" = { path = <container-folder-path>, editable = true }`)

2. `cd /opt/NeMo-FW` and run `uv sync --no-cache-dir --all-groups --inexact`

3. Local development directory is synced to run in the container

### Installing packages inside the container

All packages share a single uv virtualenv (`/opt/venv/`). The *location* you install
from determines which resolution rules apply — use the guide below.

#### Which code needs this package?

**Megatron-Bridge/Megatron-LM** (training, model architecture):

```bash
cd /opt/Megatron-Bridge
uv pip install <package>
```

Examples of Megatron-Bridge managed packages are TransformerEngine, nvidia-resiliency-ext, nvidia-modelopt. For a complete list, please visit `/opt/Megatron-Bridge/pyproject.toml`. Any of these dependencies can be re-installed via this directory.  

It is unsafe to install packages that are not part of Megatron-Bridge from within this directory. For changing those dependencies, please visit the next section.  

**NeMo toolkit, Export-Deploy, Run, or Evaluator**:

```bash
cd /opt/NeMo-FW
uv pip install <package>
```

This is safe to use for general packages. This directory also prevents accidental re-install of heavy dependencies like trt-llm or vllm. MBridge-managed packages
(TransformerEngine, nvidia-resiliency-ext, nvidia-modelopt, etc.) are protected
and will not be overwritten. For reinstalling those, please visit the next section.

**vllm, tensorrt-llm, or tensorrt**:

These are built from source and baked into the container at build time (see
`docker/Dockerfile.fw_base`). They cannot be managed via `uv pip install`.
To change them, rebuild the container.

**general note:**

Running uv pip install outside any of the two directories above might lead to a re-install of torch, thus breaking all dependencies that have been compiled against the original torch version. By running uv pip install inside any of the two directories, we can avoid this unwanted side-effect.

### Reinstalling video / image decoding packages (`av`, `decord`, `opencv-python-headless`) at runtime

Starting with the 26.04 (r0.4.0) release, the following Python packages are **not installed** in the container:

| Package | Why it was removed |
|---------|---------------------|
| [`av`](https://pypi.org/project/av/) (PyAV) | Vendored FFmpeg binaries carried an unfixed CVE. |
| [`decord`](https://pypi.org/project/decord/) | Unmaintained; vendored FFmpeg binaries carried an unfixed CVE. |
| [`opencv-python-headless`](https://pypi.org/project/opencv-python-headless/) | Bundled native libs carried an unfixed CVE. The Dockerfile also explicitly runs `pip uninstall -y opencv-python-headless` after the base-container install to scrub any pre-existing copy. |

These packages are suppressed via `sys_platform == 'never'` overrides in `/opt/Megatron-Bridge/pyproject.toml` (for `av`) and `/opt/NeMo-FW/pyproject.toml` (for all three). The override propagates to transitive consumers such as `qwen-vl-utils` and `decord[av-decode]`, so `uv sync` and `uv pip install <pkg>` will silently skip them.

If your workflow needs any of these at runtime (for example, video decoding in multimodal data pipelines), install them directly with `pip`, which does not consult uv's override list:

```bash
# install any combination you need
pip install --no-deps av
pip install --no-deps decord
pip install --no-deps opencv-python-headless
```

Notes:

- `--no-deps` keeps the install from re-resolving torch or other framework packages, preserving the container's pinned versions.
- You accept the CVE risk in each package's vendored native libraries by reinstalling it. Restrict this to workloads where you control the input media.
- The install is not persistent — rebuild it into your own image (or your job's startup script) if you need it across container restarts.
