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

"""Exercise Python Ray workers after removing the optional Java runtime."""

import logging
from pathlib import Path

import ray


@ray.remote
def double(value: int) -> int:
    """Run a small task in a Python worker."""
    return value * 2


@ray.remote
class Counter:
    """Keep state in a Python actor."""

    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> int:
        """Increment and return the actor's value."""
        self.value += 1
        return self.value


def main() -> None:
    """Check startup, task execution, and actor state without Java or GPUs."""
    assert not (Path(ray.__file__).parent / "jars").exists(), "Remove Ray's optional Java runtime first"
    try:
        ray.init(num_cpus=2, include_dashboard=False, object_store_memory=80 * 1024**2)
        assert ray.get(double.remote(21), timeout=30) == 42
        counter = Counter.remote()
        assert ray.get(counter.increment.remote(), timeout=30) == 1
        assert ray.get(counter.increment.remote(), timeout=30) == 2
    finally:
        ray.shutdown()
    logging.getLogger(__name__).info("Python Ray startup, task, and actor checks passed without Java")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
