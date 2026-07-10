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

"""Step3.7 Flickr8k SFT dataset pipeline.

Self-contained inside Megatron-Bridge with no external runtime dependency.
``trust_remote_code`` is never set; the tokenizer is loaded directly from
the local HF snapshot via ``transformers.AutoTokenizer.from_pretrained``.

See :class:`Step37Flickr8kSFTDataProvider` for the mbridge integration
entry-point.
"""

from megatron.bridge.models.stepfun.data.flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
    PATCH_ITEM_TYPE,
    ImageForInsert,
    build_image_for_insert,
    compute_rope_args,
)
from megatron.bridge.models.stepfun.data.flickr8k.provider import (
    Step37Flickr8kSFTDataProvider,
)


__all__ = [
    "IMAGE_ITEM_TYPE",
    "PATCH_ITEM_TYPE",
    "ImageForInsert",
    "build_image_for_insert",
    "compute_rope_args",
    "Step37Flickr8kSFTDataProvider",
]
