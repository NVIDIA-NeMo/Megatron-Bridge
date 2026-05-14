#!/bin/bash
# Shared HSG helpers for DeepSeek-V4 smoke scripts.

ensure_transformers_site() {
  local site=${1:?transformers site path required}
  local python_bin=${PYTHON_BIN:-/opt/venv/bin/python}
  local lock="${site}.lock"

  mkdir -p "$(dirname "$site")"

  (
    flock 9

    if [ -d "$site/transformers" ] && PYTHONPATH="$site:${PYTHONPATH:-}" "$python_bin" - <<'PY'
import transformers
from transformers import DeepseekV4Config

assert transformers.__version__ == '5.8.1', transformers.__version__
print('transformers:', transformers.__version__, transformers.__file__)
print('deepseek_v4_config:', DeepseekV4Config.__name__)
PY
    then
      exit 0
    fi

    local tmp="${site}.tmp.${SLURM_JOB_ID:-nojob}.${SLURM_PROCID:-0}.$$"
    rm -rf "$tmp"
    "$python_bin" -m pip install --target "$tmp" --no-deps 'transformers==5.8.1'

    PYTHONPATH="$tmp:${PYTHONPATH:-}" "$python_bin" - <<'PY'
import transformers
from transformers import DeepseekV4Config

assert transformers.__version__ == '5.8.1', transformers.__version__
print('transformers:', transformers.__version__, transformers.__file__)
print('deepseek_v4_config:', DeepseekV4Config.__name__)
PY

    rm -rf "$site"
    mv "$tmp" "$site"
  ) 9>"$lock"
}

ensure_emerging_optimizers_site() {
  local site=${1:?emerging optimizers site path required}
  local python_bin=${PYTHON_BIN:-/opt/venv/bin/python}
  local lock="${site}.lock"

  mkdir -p "$(dirname "$site")"

  (
    flock 9

    if [ -d "$site/emerging_optimizers" ] && PYTHONPATH="$site:${PYTHONPATH:-}" "$python_bin" - <<'PY'
import importlib.metadata as md

version = md.version('emerging-optimizers')
assert tuple(int(x) for x in version.split('.')[:2]) >= (0, 2), version
import absl
import emerging_optimizers

print('absl:', absl.__file__)
print('emerging-optimizers:', version, emerging_optimizers.__file__)
PY
    then
      exit 0
    fi

    local tmp="${site}.tmp.${SLURM_JOB_ID:-nojob}.${SLURM_PROCID:-0}.$$"
    rm -rf "$tmp"
    "$python_bin" -m pip install --target "$tmp" --no-deps \
      'absl-py==2.4.0' \
      'git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.2.0'

    PYTHONPATH="$tmp:${PYTHONPATH:-}" "$python_bin" - <<'PY'
import importlib.metadata as md

version = md.version('emerging-optimizers')
assert tuple(int(x) for x in version.split('.')[:2]) >= (0, 2), version
import absl
import emerging_optimizers

print('absl:', absl.__file__)
print('emerging-optimizers:', version, emerging_optimizers.__file__)
PY

    rm -rf "$site"
    mv "$tmp" "$site"
  ) 9>"$lock"
}
