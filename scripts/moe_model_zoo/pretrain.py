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

from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


from utils import get_model_recipe


def main():
    """Main function to run the pretraining/finetuning script."""
    recipe = get_model_recipe()
    pretrain(config=recipe, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
