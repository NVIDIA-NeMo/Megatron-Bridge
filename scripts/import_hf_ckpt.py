#!/usr/bin/env python3
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

"""
Script to import HuggingFace models into Megatron checkpoint format using the AutoBridge.

Usage:
    python import_hf_ckpt.py model_id output_path [--trust-remote-code]
"""

import argparse
import sys

from megatron.bridge.models.conversion.auto_bridge import AutoBridge


def main():
    """Main function to import HuggingFace model to Megatron checkpoint."""
    parser = argparse.ArgumentParser(description="Import HuggingFace model to Megatron checkpoint")
    parser.add_argument("model_id", help="HuggingFace model ID or path")
    parser.add_argument("output_path", help="Output directory for Megatron checkpoint")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()

    # Validate model compatibility
    if not AutoBridge.can_handle(args.model_id, trust_remote_code=args.trust_remote_code):
        print("ERROR: Model is not supported by AutoBridge")
        sys.exit(1)

    try:
        AutoBridge.import_ckpt(args.model_id, args.output_path, trust_remote_code=args.trust_remote_code)
        print(f"Import completed successfully! Checkpoint saved to: {args.output_path}")
    except Exception as e:
        print(f"Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
