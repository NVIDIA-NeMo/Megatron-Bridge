#!/bin/bash
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

# Switch the 3rdparty/Megatron-LM submodule between pinned main and dev commits.
#
# Usage:
#   ./scripts/switch_mcore.sh dev      # Switch to the pinned dev commit
#   ./scripts/switch_mcore.sh main     # Switch to the pinned main commit
#   ./scripts/switch_mcore.sh status   # Show current submodule status

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Pinned commit hashes — update these when bumping Megatron-LM versions
# ──────────────────────────────────────────────────────────────────────────────
MAIN_COMMIT="3d1a4ba71ecc49f1a0c9480c90f819d2b00f9915"   # main branch
DEV_COMMIT="cd1c215b956e09fad153e1034d2ea5ee70345234"    # dev branch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."
SUBMODULE_PATH="$REPO_ROOT/3rdparty/Megatron-LM"

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  dev       Switch 3rdparty/Megatron-LM to the pinned dev commit  (${DEV_COMMIT:0:12})
  main      Switch 3rdparty/Megatron-LM to the pinned main commit (${MAIN_COMMIT:0:12})
  status    Show the current submodule commit info

Examples:
  $(basename "$0") dev
  $(basename "$0") main
  $(basename "$0") status
EOF
    exit 1
}

log() { echo "[switch_mcore] $*"; }
err() { echo "[switch_mcore] ERROR: $*" >&2; exit 1; }

check_submodule() {
    if [ ! -d "$SUBMODULE_PATH/.git" ] && [ ! -f "$SUBMODULE_PATH/.git" ]; then
        log "Submodule not initialized. Running 'git submodule update --init'..."
        git -C "$REPO_ROOT" submodule update --init 3rdparty/Megatron-LM
    fi
}

get_current_commit() {
    git -C "$SUBMODULE_PATH" rev-parse HEAD 2>/dev/null || echo "unknown"
}

do_status() {
    check_submodule
    local current
    current=$(get_current_commit)
    log "Current commit: ${current:0:12}"
    log "Pinned main:    ${MAIN_COMMIT:0:12}"
    log "Pinned dev:     ${DEV_COMMIT:0:12}"

    if [ "$current" = "$MAIN_COMMIT" ]; then
        log "State:          main"
    elif [ "$current" = "$DEV_COMMIT" ]; then
        log "State:          dev"
    else
        log "State:          custom (neither pinned main nor dev)"
    fi
}

do_switch() {
    local target_name="$1"
    local target_commit="$2"

    check_submodule

    local current
    current=$(get_current_commit)

    if [ "$current" = "$target_commit" ]; then
        log "Already at $target_name commit (${target_commit:0:12}). Nothing to do."
        return 0
    fi

    log "Before: ${current:0:12}"
    log "Switching to $target_name commit: ${target_commit:0:12}"

    git -C "$SUBMODULE_PATH" checkout "$target_commit" || err "Failed to checkout $target_commit"

    log "After:  $(get_current_commit | cut -c1-12)"
    log "Done."
}

# --- Main ---

if [ $# -lt 1 ]; then
    usage
fi

case "$1" in
    dev)    do_switch "dev"  "$DEV_COMMIT" ;;
    main)   do_switch "main" "$MAIN_COMMIT" ;;
    status) do_status ;;
    -h|--help) usage ;;
    *) err "Unknown command: $1. Use 'dev', 'main', or 'status'." ;;
esac
