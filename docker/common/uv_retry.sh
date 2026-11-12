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

# Retry only uv downloads rejected by a transient HTTP response. Preserve the
# original command, diagnostics and failure after the bounded retry budget.
set -uo pipefail

log=$(mktemp) || exit 1
trap 'rm -f "$log"' EXIT
for attempt in 1 2 3; do
    "$@" 2>&1 | tee "$log"
    statuses=("${PIPESTATUS[@]}")
    status=${statuses[0]}
    if ((statuses[1] != 0)); then
        exit "${statuses[1]}"
    fi
    if ((status == 0)); then
        exit 0
    fi
    if ((attempt == 3)) || ! grep -q 'Failed to download' "$log" ||
        ! grep -Eq 'HTTP status (server error \(5[0-9][0-9][^0-9]|client error \(429[^0-9])' "$log"; then
        exit "$status"
    fi
    echo "uv download received a transient HTTP response; retrying ($attempt/3) in 10s" >&2
    sleep 10 || exit "$?"
done
