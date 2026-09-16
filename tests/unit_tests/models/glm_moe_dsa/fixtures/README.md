# GLM-5.2 / GLM-5.3 config fixtures

Unmodified publisher `config.json` files, pinned on 2026-09-15:

- `glm52-config.json`: [zai-org/GLM-5.2 at cf457fa734ab149ffef225f80893eb38c6ff5cdc](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json).
- `glm53-config.json`: [zai-org/GLM-5.3 at aca966e4e02791568aa6a4ced368624b3d897f42](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json).

The regression loads both configs through AutoBridge without downloading weights
or mocking provider construction. Architecture fields match; checkpoint
quantization metadata and the recorded Transformers version differ. These tests
do not establish full-model numerical parity or training qualification.
