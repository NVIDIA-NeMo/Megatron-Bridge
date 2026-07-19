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

import argparse
import logging
import re
from pathlib import Path

from megatron.bridge.data.energon.prepare import prepare_webdataset


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare one BAGEL T2I tar for Energon")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Prepare one BAGEL T2I tar as an Energon CrudeWebdataset."""
    args = parse_args()
    tar_paths = sorted(args.dataset_dir.rglob("*.tar"))
    if len(tar_paths) != 1:
        raise ValueError(f"expected exactly one .tar file, found {len(tar_paths)}")

    relative_tar = tar_paths[0].relative_to(args.dataset_dir).as_posix()
    prepare_webdataset(
        args.dataset_dir,
        {"train": rf"^{re.escape(relative_tar)}$"},
        num_workers=args.num_workers,
    )
    dataset_yaml = args.dataset_dir / ".nv-meta" / "dataset.yaml"
    dataset_yaml.write_text(
        "__module__: megatron.energon\n__class__: CrudeWebdataset\n",
        encoding="utf-8",
    )
    logger.info("Prepared %s as an Energon CrudeWebdataset", relative_tar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
