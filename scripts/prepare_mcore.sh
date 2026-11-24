#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Keep the standalone repository as the only provider of megatron.bridge.
# An optional checkout path also supports Docker's separate baseline clone.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mcore="${1:-$repo_root/3rdparty/Megatron-LM}"
fail() { echo "[prepare_mcore] $*" >&2; exit 1; }

if [[ ! -e "$mcore/.git" ]]; then
    [[ $# -eq 0 ]] || fail "Not a Git checkout: $mcore"
    git -C "$repo_root" submodule update --init 3rdparty/Megatron-LM
fi
[[ -z "$(git -C "$mcore" status --porcelain --untracked-files=no)" ]] \
    || fail "MCore has tracked changes; save them before preparing the checkout."

# Never discard local source files, including ignored files in the excluded tree.
for options in '--others --exclude-standard' '--others --ignored --exclude-standard'; do
    # Intentional word splitting: these are fixed Git options, not user input.
    [[ -z "$(git -C "$mcore" ls-files $options -- megatron/bridge)" ]] \
        || fail "Local files remain in megatron/bridge; move them out before preparing MCore."
done
for output in build megatron_core.egg-info; do
    [[ ! -L "$mcore/$output" ]] || fail "Refusing to clean symlink: $mcore/$output"
    [[ -z "$(git -C "$mcore" ls-files -- "$output")" ]] \
        || fail "Refusing to clean tracked build output: $output"
done

# init + set also works on Git 2.25. Older Git needs this directory to exist.
info_dir="$(git -C "$mcore" rev-parse --git-path info)"
[[ "$info_dir" = /* ]] || info_dir="$mcore/$info_dir"
mkdir -p "$info_dir"
git -C "$mcore" sparse-checkout init --no-cone
git -C "$mcore" sparse-checkout set '/*' '!/megatron/bridge/'

# Setuptools can otherwise reuse Bridge files from an earlier unfiltered wheel.
# Only ignored build output is removed; unexpected non-ignored files cause failure.
git -C "$mcore" clean -fdX -- build/ megatron_core.egg-info/
[[ ! -e "$mcore/build" && ! -e "$mcore/megatron_core.egg-info" ]] \
    || fail "Move remaining build/ or megatron_core.egg-info/ files out before installing MCore."
[[ ! -e "$mcore/megatron/bridge" ]] \
    || fail "Upstream megatron/bridge is still present; do not install this checkout."
echo "[prepare_mcore] Ready: $(git -C "$mcore" rev-parse --short HEAD) (upstream Bridge excluded)"
