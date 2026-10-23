# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Run the launcher's Megatron-Bridge checkout in NVCRE pods instead of the image's copy.

On Slurm the launcher's checkout is bind-mounted over the image's ``/opt/Megatron-Bridge``.
Pods have no such mount, so an init container clones the same Megatron-Bridge commit into
a volume that the training container shares."""

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit


BRIDGE_POD_ROOT = "/mnt/workspace/megatron-bridge"
WORKSPACE_VOLUME: dict[str, Any] = {"name": "workspace", "emptyDir": {}}
WORKSPACE_MOUNT: dict[str, Any] = {"name": "workspace", "mountPath": "/mnt/workspace"}


class LocalBridge(NamedTuple):
    """Where the launcher's checkout can be fetched from, and at which commit."""

    url: str
    commit: str


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def pod_clone_url(remote_url: str) -> str:
    """HTTPS URL a pod can clone anonymously: SSH remotes are rewritten and any credentials dropped."""
    # scp-style ``git@host:path``: the colon separates host from path.
    scp = re.match(r"^git@([^:/]+):(.+)$", remote_url)
    if scp:
        return f"https://{scp.group(1)}/{scp.group(2)}"
    parts = urlsplit(remote_url)
    if parts.scheme == "ssh" and parts.hostname:
        # The port of an ssh:// URL is the SSH port, which does not apply over HTTPS.
        return urlunsplit(("https", parts.hostname, parts.path, "", ""))
    if parts.scheme in ("http", "https") and parts.hostname:
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    raise ValueError(f"Cannot derive a clone URL a pod can reach from remote {remote_url!r}")


def local_bridge_source(repo_root: Path) -> LocalBridge:
    """The origin URL and checked-out commit of the Megatron-Bridge checkout at ``repo_root``."""
    try:
        url = pod_clone_url(_git(repo_root, "config", "--get", "remote.origin.url"))
        commit = _git(repo_root, "rev-parse", "HEAD")
        # Submodule pointers are ignored: only this repository's own commit is fetched.
        dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"--nvcre_use_mbridge_commit needs {repo_root} to be a git checkout with an 'origin' remote: {exc.stderr}"
        ) from exc
    if dirty:
        raise RuntimeError(
            f"--nvcre_use_mbridge_commit runs commit {commit} in the pods, but {repo_root} has uncommitted changes "
            "to tracked files. Commit and push them, or stash them, and resubmit."
        )
    return LocalBridge(url, commit)


def bridge_init_container(source: LocalBridge, image: str) -> dict[str, Any]:
    """Init container that clones ``source`` into ``BRIDGE_POD_ROOT``; ``image`` is the training image (has git)."""
    steps = [
        "set -ex",
        f"git init -q {BRIDGE_POD_ROOT}",
        f"cd {BRIDGE_POD_ROOT}",
        f"git remote add origin {shlex.quote(source.url)}",
        f"git fetch -q --depth 1 origin {shlex.quote(source.commit)}",
        "git checkout -q FETCH_HEAD",
    ]
    return {
        "name": "megatron-bridge-clone",
        "image": image,
        "command": ["/bin/bash", "-c"],
        "args": ["\n".join(steps)],
        "volumeMounts": [dict(WORKSPACE_MOUNT)],  # a copy: shared objects would be dumped as YAML anchors
    }
