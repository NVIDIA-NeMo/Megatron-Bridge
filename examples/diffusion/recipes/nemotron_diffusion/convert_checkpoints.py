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
#!/usr/bin/env python3
"""
NemotronDiffusion HF <-> Megatron-Bridge checkpoint conversion.

Uses NemotronDiffusionAutoBridge, a thin AutoBridge subclass that bypasses
architecture name validation (MinistralDiffEncoderModel doesn't end in
ForCausalLM/ForConditionalGeneration) and routes directly to NemotronDiffusionBridge.

Usage:
  # HF -> Megatron-Bridge
  python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py import \
    --hf-model /path/to/hf_model \
    --megatron-path /path/to/mb_checkpoint

  # Megatron-Bridge -> HF
  python examples/diffusion/recipes/nemotron_diffusion/convert_checkpoints.py export \
    --hf-model /path/to/hf_model \
    --megatron-path /path/to/mb_checkpoint \
    --hf-path /path/to/output_hf
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from megatron.bridge.diffusion.conversion.nemotron_diffusion.nemotron_diffusion_bridge import (
    NemotronDiffusionBridge,
)
from megatron.bridge.models.conversion.auto_bridge import AutoBridge


class NemotronDiffusionAutoBridge(AutoBridge):
    """AutoBridge subclass for MinistralDiffEncoderModel.

    AutoBridge rejects architectures not ending in ForCausalLM/ForConditionalGeneration
    in three places: _validate_config, _model_bridge (_causal_lm_architecture), and
    save_hf_weights (_causal_lm_architecture). We override all three to route directly
    to NemotronDiffusionBridge.
    """

    def __init__(self, hf_pretrained):
        super().__init__(hf_pretrained)
        self._nemotron_bridge = NemotronDiffusionBridge()

    @classmethod
    def _validate_config(cls, config, path):
        pass

    @property
    def _model_bridge(self):
        return self._nemotron_bridge

    def save_hf_weights(
        self,
        model,
        path,
        show_progress=True,
        strict=True,
        merge_adapter_weights=True,
        distributed_save=False,
        **kwargs,
    ):
        """Override to avoid _causal_lm_architecture lookup in dispatch."""
        generator = self._nemotron_bridge.stream_weights_megatron_to_hf(
            model,
            self.hf_pretrained,
            cpu=True,
            show_progress=show_progress,
            merge_adapter_weights=merge_adapter_weights,
        )
        state_dict = {name: tensor.contiguous().cpu() for name, tensor in generator}
        plan = split_torch_state_dict_into_shards(state_dict)
        safe_dir = Path(path)
        safe_dir.mkdir(parents=True, exist_ok=True)
        for filename, tensors in plan.filename_to_tensors.items():
            shard = {k: state_dict[k] for k in tensors}
            save_file(shard, safe_dir / filename)
        if plan.is_sharded:
            index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
            with open(safe_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)


def main():
    """Entry point for HF<->Megatron checkpoint conversion."""
    parser = argparse.ArgumentParser(description="NemotronDiffusion checkpoint conversion")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser("import", help="HF -> Megatron-Bridge")
    import_parser.add_argument("--hf-model", required=True)
    import_parser.add_argument("--megatron-path", required=True)
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    import_parser.add_argument("--device", default=None)

    export_parser = subparsers.add_parser("export", help="Megatron-Bridge -> HF")
    export_parser.add_argument("--hf-model", required=True)
    export_parser.add_argument("--megatron-path", required=True)
    export_parser.add_argument("--hf-path", required=True)
    export_parser.add_argument("--no-progress", action="store_true")
    export_parser.add_argument("--not-strict", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    if args.command == "import":
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = dtype_map[args.torch_dtype]
        print(f"Importing {args.hf_model} -> {args.megatron_path}")
        NemotronDiffusionAutoBridge.import_ckpt(
            hf_model_id=args.hf_model,
            megatron_path=args.megatron_path,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print(f"Done. Checkpoint saved to {args.megatron_path}")

    elif args.command == "export":
        print(f"Exporting {args.megatron_path} -> {args.hf_path}")
        bridge = NemotronDiffusionAutoBridge.from_hf_pretrained(args.hf_model, trust_remote_code=True)
        bridge.export_ckpt(
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
        )

        # Ensure rope_scaling is present for backward compatibility with older transformers.
        # Transformers 5.x writes rope_parameters but older versions need rope_scaling.
        config_path = Path(args.hf_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if "rope_parameters" in config and "rope_scaling" not in config:
                config["rope_scaling"] = config["rope_parameters"]
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

        print(f"Done. HF model saved to {args.hf_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
