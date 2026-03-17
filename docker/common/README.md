# NeMo Framework Docker Container

## Repository Overview

- Megatron-Bridge (/opt/Megatron-Bridge/)
- Megatron-LM (/opt/Megatron-Bridge/3rdparty/Megatron-LM/)
- Evaluator (/opt/Evaluator/)
- Export-Deploy (/opt/Export-Deploy/)
- NeMo (/opt/NeMo/)
- Run (/opt/Run/)

---

## Installed Packged

Pip installed packages:

- DeepEP
- vLLM
- TRT-LLM

Execute `pip list` to see full list of installed packages via pip.

Uv virtualenv installed packages (/opt/venv/)

- TransformerEngine
- nvidia-resiliency
- TensorRT-Model-Optimizer

Execute `uv pip list` to see full list of installed packages. Packages installed in /opt/venv take precedence over pip installed packages.

See /opt/NeMo-FW/pyproject.toml for additonal uv configurations.

---

## Development

### Mounting and syncing local repository into the container

Local working directories can be mounted via docker run:

```bash
docker run -v <local-folder-path>:<container-folder-path> <container-image>
```

#### 1 Overwrite existing directory

1. Define the `<container-folder-path>` to correspond to the path defined in Repository Overview. (ie. /my-path/Megatron-Bridge/:/opt/Megatron-Bridge/)

2. `cd /opt/NeMo-FW` and run `uv sync --no-cache-dir --all-groups --inexact`

3. Local development directory is synced to run in the container

#### 2 Mount directory to a different path

1. Modify `/opt/NeMo-FW/pyproject.toml` sections `tool.uv.sources` and `tool.uv; override-dependencies` to reflect new path. (ie. "megatron-bridge" = { path = `<container-folder-path>`, editable = true })

2. `cd /opt/NeMo-FW` and run `uv sync --no-cache-dir --all-groups --inexact`

3. Local development directory is synced to run in the container

### Installing packages inside the container

Packages can be installed into the uv virtualenv (`/opt/venv/`) from two locations:

#### From `/opt/NeMo-FW` (default)

Use this when unsure, or when the dependency is not specific to Megatron:

```bash
cd /opt/NeMo-FW
uv pip install <package>
```

#### From `/opt/Megatron-Bridge`

Use this only for Megatron-specific dependencies:

```bash
cd /opt/Megatron-Bridge
uv pip install <package>
```
