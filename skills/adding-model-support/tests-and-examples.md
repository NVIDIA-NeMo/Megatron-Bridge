# Test and Example Patterns

## Unit Tests

Location: `tests/unit_tests/models/<model>/`

### Bridge Unit Test

Build a small HF config, then verify `hf_config_to_model_config()` and `mapping_registry()`. Use the
model's HF config class, or a `SimpleNamespace` when the class is not importable. Do not use a
`Mock()` config: every attribute you do not set is itself a `Mock`, so the shared mapping picks up
values such as `hidden_act` and the conversion fails.

```python
from dataclasses import fields
from unittest.mock import Mock, patch

import pytest
import torch.nn.functional as F

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def _make_config(**overrides):
    """Create a small HF config with model-specific attributes."""
    values = dict(
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32000,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    return MyModelConfig(architectures=["MyModelForCausalLM"], **values)


class TestMyModelBridgeModelConfig:
    def test_provider_bridge_is_inherited_compatibility_only(self):
        assert "provider_bridge" not in MyModelBridge.__dict__

    def test_conversion_uses_builder_config(self):
        assert MyModelBridge.USE_MODEL_CONFIG_FOR_CONVERSION is True

    def test_config_mapping(self):
        bridge = MyModelBridge()
        # The builder path must not route through the legacy provider.
        with (
            patch.object(bridge, "provider_bridge", side_effect=AssertionError("provider path used")),
            patch.object(bridge, "hf_config_to_provider_kwargs", side_effect=AssertionError("provider kwargs used")),
        ):
            model_config = bridge.hf_config_to_model_config(_make_config())

        assert isinstance(model_config, BridgeGPTModelConfig)
        assert model_config.num_layers == 4
        assert model_config.hidden_size == 256
        assert model_config.activation_func is F.silu
        assert model_config.position_embedding_type == "rope"
        assert model_config.share_embeddings_and_output_weights is False

    def test_model_config_matches_provider(self):
        config = _make_config()
        bridge = MyModelBridge()
        model_config = bridge.hf_config_to_model_config(config)
        pretrained = Mock(spec=PreTrainedCausalLM)
        pretrained.config = config
        with pytest.warns(FutureWarning):
            provider = bridge.provider_bridge(pretrained)

        model_fields = {f.name for f in fields(model_config)} | {f.name for f in fields(model_config.transformer)}
        shared = ({f.name for f in fields(provider)} & model_fields) - {"transformer_layer_spec"}
        for name in sorted(shared):
            assert getattr(model_config, name) == getattr(provider, name), name


class TestMyModelBridgeMappingRegistry:
    @pytest.fixture
    def bridge(self):
        return MyModelBridge()

    def test_has_embedding_mapping(self, bridge):
        registry = bridge.mapping_registry()
        mapping = registry.megatron_to_hf_lookup("embedding.word_embeddings.weight")
        assert mapping.hf_param == "model.embed_tokens.weight"

    def test_has_output_layer_mapping(self, bridge):
        registry = bridge.mapping_registry()
        megatron_params = {m.megatron_param for m in registry.mappings}
        assert any("output_layer" in p for p in megatron_params)
```

For a VLM, nest `text_config` and `vision_config` in the HF config and assert the fields read from the
top-level config, such as `tie_word_embeddings` and `image_token_id`.

### Config and Builder Unit Test (only if a custom config or builder exists)

Skip this if the bridge uses `BridgeGPTModelConfig` (most LLM bridges). For a custom `ModelConfig`,
test its defaults, serialization roundtrip, and the builder's model construction; see
`tests/unit_tests/models/muse_glimmer/`.

### Provider Unit Test (legacy provider subclasses only)

Only needed when changing a family that still has a provider subclass with custom fields or
`provide()` logic.

```python
class TestMyModelProvider:
    def test_defaults(self):
        provider = MyModelProvider(
            num_layers=32, hidden_size=4096,
            num_attention_heads=32, num_query_groups=8,
        )
        assert provider.normalization == "RMSNorm"

    def test_tp_validation(self):
        with pytest.raises(ValueError):
            provider = MyModelProvider(
                num_query_groups=2,
                tensor_model_parallel_size=4,
            )
            provider.validate_parallelism()
```

### Skip conditions

```python
# Module-level skip for optional dependencies
pytestmark = pytest.mark.skipif(
    not _HAS_MODEL_CLASS,
    reason="transformers version does not support MyModel"
)

# Class-level skip
@pytest.mark.skipif(not _HAS_MOE_CLASS, reason="MoE class not available")
class TestMyMoEBridge:
    ...
```

## Functional Tests

Location: `tests/functional_tests/test_groups/models/<model>/`

### Conversion Functional Test

Tests HF ↔ Megatron roundtrip on GPU with a toy model.

```python
import subprocess
import pytest

# Toy model config (reduced sizes for fast testing)
HF_TOY_MODEL_CONFIG = {
    "model_type": "my_model",
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 2048,
    "max_position_embeddings": 512,
    # ... model-specific fields
}

@pytest.fixture(scope="class")
def toy_model_path(tmp_path_factory):
    """Create a small HF model for testing."""
    from transformers import AutoConfig
    model_dir = tmp_path_factory.mktemp("toy_model")
    config = AutoConfig.for_model(**HF_TOY_MODEL_CONFIG)
    model = MyModelForCausalLM(config)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    return str(model_dir)

@pytest.mark.run_only_on("GPU")
class TestMyModelConversion:
    @pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1)])
    def test_roundtrip(self, toy_model_path, tp, pp, tmp_path):
        result = subprocess.run(
            [
                "uv", "run", "python", "-m", "torch.distributed.run",
                f"--nproc_per_node={tp * pp}",
                "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
                f"--hf-model-id={toy_model_path}",
                f"--output-dir={tmp_path}",
                f"--tp={tp}", f"--pp={pp}",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"
```

### VLM toy model creation

VLM toy models need both text and vision configs:

```python
HF_VLM_TOY_CONFIG = {
    "model_type": "my_vlm",
    "text_config": {
        "num_hidden_layers": 4,
        "hidden_size": 256,
        # ...
    },
    "vision_config": {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        # ...
    },
    "image_token_id": 151655,
    "video_token_id": 151656,
    "tie_word_embeddings": False,
}
```

### MoE toy model: fuse expert weights

Some MoE models store experts in fused format. After creating the model, fuse:

```python
def _fuse_moe_expert_weights(model_dir):
    """Convert per-expert weights to fused gate_up_proj/down_proj layout."""
    # Load safetensors, reshape per-expert into combined tensors, save back
    ...
```

### Test marks

```python
@pytest.mark.run_only_on("GPU")       # Requires GPU
@pytest.mark.parametrize("tp,pp", [(2, 1)])  # Parallelism variants
@pytest.mark.skipif(...)               # Conditional skip
```

## Example Scripts

Example scripts target **real published models** (e.g. `Qwen/Qwen3-8B`), not toy configs.
The inference script must produce reasonable output — a coherent text completion for LLMs,
a plausible image description for VLMs. This is the acceptance bar for the deliverable.

### Conversion example (`examples/models/<family>/<model>/README.md`)

Document real-model import and export with the stable conversion CLI. Use the GPU backend when
the model requires distributed conversion; omit the execution and device flags for local CPU
conversion. Add a model-specific shell wrapper only when the shared CLI cannot express required
model preparation or verification.

```bash
WORKSPACE=${WORKSPACE:-/workspace}
MODEL_NAME=<default-model-name>
HF_MODEL=<org>/${MODEL_NAME}
TP=1; PP=8; EP=1  # Adjust per model

# Import HF → Megatron
./scripts/conversion/convert.sh import \
    --executor local --device gpu --gpus-per-node 8 \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME} \
    --tp ${TP} --pp ${PP} --ep ${EP} \
    --torch-dtype bfloat16

# Compare logits
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/compare_hf_and_megatron/compare.py \
    --hf_model_path ${HF_MODEL} \
    --megatron_model_path ${WORKSPACE}/${MODEL_NAME} \
    --prompt "Hello, how are you?" \
    --tp ${TP} --pp ${PP} --ep ${EP}

# Export Megatron → HF
./scripts/conversion/convert.sh export \
    --executor local --device gpu --gpus-per-node 8 \
    --hf-model ${HF_MODEL} \
    --megatron-path ${WORKSPACE}/${MODEL_NAME}/iter_0000000 \
    --hf-path ${WORKSPACE}/${MODEL_NAME}-hf-export \
    --tp ${TP} --pp ${PP} --ep ${EP}

# Roundtrip validation
uv run python -m torch.distributed.run --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id ${HF_MODEL} --tp ${TP} --pp ${PP} --ep ${EP}
```

### Inference example (`examples/models/<family>/<model>/inference.sh`)

For LLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
    --hf_model_path ${HF_MODEL} --prompt "Hello"
```

For VLMs:
```bash
uv run python examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${HF_MODEL} \
    --image_path "https://example.com/image.jpeg" \
    --prompt "Describe this image."
```

### VLM inference adds `--model_class` for non-default HF classes:
```bash
--model_class "MyModelForConditionalGeneration"
```

## Optional External NeMo-RL E2E

After Bridge unit and conversion tests pass for a new model/provider, optionally run a small
external-loop smoke test through NeMo-RL when downstream RL compatibility matters or the PR claims
NeMo-RL compatibility. This is not required for every model-support change. Start with the
Megatron policy GRPO smoke (`tests/functional/grpo_megatron.sh`) to prove NeMo-RL can import the
local Bridge checkout, build the Megatron policy, initialize optimizer/scheduler state, and
complete a short RL training loop.

Add the non-colocated vLLM refit variant when the change touches HF export, parameter mapping,
policy-to-generation weight transfer, delta compression, or vLLM loading. Add PEFT/checkpoint,
Megatron generation, parallelism stress, learning-signal, or architecture-specific variants when
the change requires that coverage.

Read @skills/nemo-rl-e2e-testing/SKILL.md for the full workflow, environment setup, metric checks,
failure triage, and reporting format.

## Optional External verl E2E

After Bridge unit and conversion tests pass for a new model/provider, optionally run a small
external-loop smoke test through verl when downstream RL compatibility matters or the PR claims verl
compatibility. This is not required for every model-support change. Start with the non-vanilla
Bridge path, LoRA enabled, and Megatron DDP selected, then add save/resume, parallelism stress,
Megatron-FSDP, or architecture-specific variants when the change requires that coverage.

Read @skills/verl-e2e-testing/SKILL.md for the full workflow and reporting format.

## Documentation Page

Create `docs/models/<type>/<model>.md`:

```markdown
# <Model Name>

## Supported Variants

| Variant | Parameters | HF Path |
|---------|-----------|---------|
| <Model>-7B | 7B | <org>/<model>-7B |
| <Model>-70B | 70B | <org>/<model>-70B |

## Conversion

\`\`\`bash
# HF → Megatron
./scripts/conversion/convert.sh import \
    --hf-model <org>/<model> --megatron-path /workspace/<model>
\`\`\`

## Training

See `examples/models/<family>/<model>/slurm_sft.sh` and `slurm_peft.sh` for full Slurm scripts.
Single-node quick-start:

### SFT
\`\`\`bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <model>_<size>_sft_config \
    checkpoint.pretrained_checkpoint=/workspace/models/<model> \
    model.tensor_model_parallel_size=<TP> \
    model.pipeline_model_parallel_size=<PP> \
    train.train_iters=1000 \
    train.global_batch_size=<GBS>
\`\`\`

### PEFT (LoRA)
\`\`\`bash
uv run python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe <model>_<size>_peft_config \
    checkpoint.pretrained_checkpoint=/workspace/models/<model> \
    model.tensor_model_parallel_size=<TP> \
    model.pipeline_model_parallel_size=<PP> \
    train.train_iters=1000 \
    train.global_batch_size=<GBS>
\`\`\`

## Known Limitations
- [List any known issues]
```
