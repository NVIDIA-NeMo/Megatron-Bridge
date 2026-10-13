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

On Slurm the launcher's checkout is bind-mounted over the image's ``/opt/Megatron-Bridge``. Pods have no
such mount, so an init container clones the same commits (Megatron-Bridge and its Megatron-LM submodule)
into a volume that the training container shares.
"""

import logging
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)

BRIDGE_POD_ROOT = "/mnt/workspace/megatron-bridge"
WORKSPACE_VOLUME: dict[str, Any] = {"name": "workspace", "emptyDir": {}}
WORKSPACE_MOUNT: dict[str, Any] = {"name": "workspace", "mountPath": "/mnt/workspace"}
MCORE_SUBMODULE = "3rdparty/Megatron-LM"


class LocalBridge(NamedTuple):
    """Where the launcher's checkout can be fetched from, and at which commits."""

    url: str
    commit: str
    mcore_commit: str | None


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def pod_clone_url(remote_url: str) -> str:
    """HTTPS URL a pod can clone anonymously: SSH remotes are rewritten and any credentials dropped."""
    ssh = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+)$", remote_url)
    if ssh:
        return f"https://{ssh.group(1)}/{ssh.group(2)}"
    parts = urlsplit(remote_url)
    if parts.scheme in ("http", "https") and parts.hostname:
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    raise ValueError(f"Cannot derive a clone URL a pod can reach from remote {remote_url!r}")


def local_bridge_source(repo_root: Path) -> LocalBridge:
    """The origin URL and checked-out commits of the Megatron-Bridge checkout at ``repo_root``."""
    try:
        url = pod_clone_url(_git(repo_root, "config", "--get", "remote.origin.url"))
        commit = _git(repo_root, "rev-parse", "HEAD")
        dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
        # A submodule is a gitlink entry ("160000 commit <sha>\t<path>") in the tree, checked out or not.
        entry = _git(repo_root, "ls-tree", "HEAD", MCORE_SUBMODULE)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"--nvcre_use_mbridge_commit needs {repo_root} to be a git checkout with an 'origin' remote: {exc.stderr}"
        ) from exc
    if dirty:
        logger.warning("Local Megatron-Bridge has uncommitted changes; pods will run the committed %s only", commit)
    return LocalBridge(url, commit, entry.split()[2] if entry.startswith("160000") else None)


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
    if source.mcore_commit:
        steps.append(f"git submodule update --init --depth 1 {MCORE_SUBMODULE}")
    return {
        "name": "megatron-bridge-clone",
        "image": image,
        "command": ["/bin/bash", "-c"],
        "args": ["\n".join(steps)],
        "volumeMounts": [dict(WORKSPACE_MOUNT)],  # a copy: shared objects would be dumped as YAML anchors
    }
