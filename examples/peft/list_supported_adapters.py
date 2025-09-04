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

from megatron.bridge.peft import AutoPEFTBridge


def main() -> None:
    """List all PEFT adapter types supported by the AutoPEFTBridge."""
    supported_adapters = AutoPEFTBridge.list_supported_adapters()

    print("🚀 Megatron-Bridge AutoPEFTBridge - Supported PEFT Adapters")
    print("=" * 55)
    print()

    if not supported_adapters:
        print("❌ No supported adapters found.")
        print("   This might indicate that no PEFT bridge implementations are registered.")
        return

    print(f"✅ Found {len(supported_adapters)} supported PEFT adapter type(s):")
    print()

    for i, adapter_type in enumerate(supported_adapters, 1):
        print(f"  {i:2d}. {adapter_type}")

    print()
    print("💡 Usage:")
    print("   To use any of these adapter types, you can load them with:")
    print("   >>> base_bridge = AutoBridge.from_hf_pretrained('base_model')")
    print("   >>> peft_bridge = AutoPEFTBridge.from_hf_pretrained('adapter_model')")
    print("   >>> peft_model = peft_bridge.to_megatron_model(base_bridge)")
    print()
    print("🔍 PEFT Bridge Details:")
    print("   Each adapter type has specific implementation details and configurations.")
    print("   Check the src/megatron/bridge/peft/ directory for:")
    print("   • Adapter-specific bridge implementations")
    print("   • Configuration examples and mapping details")
    print("   • Parameter conversion logic")
    print("   • Distributed training optimizations")
    print()
    print("📚 For more examples, see the examples/peft/ directory.")


if __name__ == "__main__":
    main()