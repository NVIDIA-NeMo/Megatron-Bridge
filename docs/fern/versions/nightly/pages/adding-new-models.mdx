# Contribute a New Model to Megatron Bridge

This guide explains how to add support for a new 🤗 Hugging Face model (or family) to Megatron Bridge so to convert between HF ↔ Megatron-Core formats and participate in training recipes.

Use this checklist-style flow: scaffold → model config/builder mapping → parameter mappings → tests → validation.


## Prerequisites

- Familiarity with the Megatron Bridge repository structure.
- A working Python 3.10+ environment with Megatron Bridge installed (see [installation instructions](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/README.md#installation)), a container is recommended.
- Familiarity with Megatron-Core GPT-style modules and 🤗 Transformers config objects.
- Access to a small HF checkpoint for local testing.
- Read first:
  - [Bridge user guide](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
  - [Technical details](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
  - [Model bridges overview](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/README.md)


## 1) Decide the integration strategy

 Most GPT-style models (such as the Qwen and Llama families) can reuse the Megatron-Core GPT model by mapping their configuration. If the model requires custom building blocks (e.g., an attention variant, RoPE variant, or VLM modules), add a lightweight specialization similar to how 🤗 HuggingFace implements `modeling_xxx.py`.

- **Standard GPT-style models**: Implement a `Bridge` that maps the HF config into `BridgeGPTModelConfig`; reuse the stock GPT builder. See the [Llama bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py).
- **Models with a custom layer spec or builder**: Keep `BridgeGPTModelConfig`, inject a named `transformer_layer_spec` from `hf_config_to_model_config_kwargs`, and use the bridge's `MODEL_BUILDER_CLASS` when stock GPT construction needs a different builder.
- **Models with additional build data or non-GPT components**: Add a serializable model-config dataclass and standalone `ModelBuilder` in the model family directory. Keep custom modeling code local to the family. Existing provider classes are compatibility-only and should not be used by new integrations.


## 2) Scaffold the model folder

Create a folder under `src/megatron/bridge/models/<your_model>/` and add:

- `<your_model>_bridge.py`: architecture-specific bridge that maps HF config → builder-backed model config and defines parameter mappings. For example: [Llama bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/llama_bridge.py), [Qwen3 bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen3_bridge.py), or [Qwen2 bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/qwen/qwen2_bridge.py).
- Optional `<your_model>_model_config.py` or `model_config.py`: pure serializable config plus standalone builder when stock GPT construction is insufficient.
- Optional: `README.md` with any model quirks. For example: [Llama README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/llama/README.md).

## 3) Implement the Model Config and Builder

The bridge maps Hugging Face fields into an outer `ModelConfig` containing the exact Megatron-Core transformer config. Construction belongs to a `ModelBuilder`:

- Parallelism: `tensor_model_parallel_size`, `pipeline_model_parallel_size`, optional VPP/EP settings.
- Numerics: `fp16`, `bf16`, `params_dtype`, activation recomputation.
- Architecture quirks: RoPE base/scale, QK layernorm, tied embeddings, KV groups, max sequence length, etc.
- Optional custom modules: select custom attention/MLP implementations in the builder or layer spec.

For standard GPT-style models, reuse `BridgeGPTModelConfig` and its stock builder. A custom layer spec does not require a config subclass:
```python
def your_model_layer_spec(config, vp_stage=None):
    return get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )


class YourModelBridge(MegatronModelBridge):
    MODEL_BUILDER_CLASS = "megatron.bridge.models.your_model.model_config.YourModelBuilder"

    def hf_config_to_model_config_kwargs(self, hf_config):
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        kwargs["transformer_layer_spec"] = your_model_layer_spec
        return kwargs
```

Use a named, importable factory so config serialization can restore it. Set `MODEL_BUILDER_CLASS` only when a custom builder is needed; omit it to use the stock GPT builder.

Define a custom outer config only when construction needs additional serializable fields that are absent from MCore `TransformerConfig`. Give its builder a stable import path:
```python
@dataclass(kw_only=True)
class YourModelConfig(BridgeGPTModelConfig):
    builder: ClassVar[str] = "megatron.bridge.models.your_model.model_config.YourModelBuilder"
    custom_rope_parameter: float


class YourModelBuilder(GPTModelBuilder):
    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        return YourMegatronModel(
            config=self._model_config.transformer,
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
        )
```

Subclassing `GPTModelBuilder` reuses its distributed wrapping behavior. A builder for a different model shape must also implement `build_distributed_models`. Configs must remain serializable data. Do not add a config subclass merely to select a spec or builder. Do not inherit `ModelProviderMixin`, attach undeclared fields, or put process-group state on the config. Expose nested transformer fields through the flat config API so users can set `cfg.model.tensor_model_parallel_size` without knowing field ownership.


## 4) Define Config and Parameter Mappings

Use `hf_config_to_model_config_kwargs` for standard GPT-style config mapping, or override `model_config_bridge` for custom/multimodal config construction. Use `MegatronMappingRegistry` to map Megatron parameter names to Hugging Face parameter names. Start with the essentials (embeddings, final norm, QKV, MLP), then add extras (biases, rotary embeddings, experts, and vision blocks).

- `model_config_bridge`: see [model_bridge.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/model_bridge.py)
- `MegatronMappingRegistry`: see [mapping_registry.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/mapping_registry.py)
- Mapping implementations: see [param_mapping.py](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/conversion/param_mapping.py)
- Background: see [Bridge technical details](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)

Example registration skeleton:

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import LlamaForCausalLM  # replace with your HF class
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class YourModelBridge(MegatronModelBridge):
    def hf_config_to_model_config_kwargs(self, hf_config):
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs["your_model_field"] = hf_config.your_model_field
        return config_kwargs

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            ...
        )
```

Notes:
- Use `*` wildcards for per-layer patterns; the number of wildcards must match between `megatron_param` and the HF pattern(s).
- `*` typically captures layer indices; `**` can match across dots. For example, to map both `.weight` and `.bias` together:
  ```python
  AutoMapping(
      megatron_param="output_layer.**",
      hf_param="lm_head.**",
  ),
  ```
- In some cases, the same module can have different Megatron parameter names depending on whether you use the Transformer Engine backend or the PyTorch backend. In that case, list both mappings, e.g., `[AutoMapping(megatron_param="te_backend_name", hf_param="hf_name"), AutoMapping(megatron_param="pytorch_backend_name", hf_param="hf_name")]`. Multiple Megatron parameters can map to the same Hugging Face parameter because, during conversion, the registry only queries the current model's module names.
- Prefer `AutoMapping` when the Megatron layer type implies the TP split automatically.
- Use `QKVMapping` for fused QKV and `GatedMLPMapping` for gate/up concatenation.

### Suggested Cursor prompt (Bridge) [Expermental]
```text
You are working in the Megatron Bridge repo. Create `src/megatron/bridge/models/<your_model>/<your_model>_bridge.py`.

Goal: Implement a bridge class that connects an HF model class to a Megatron model using `MegatronModelBridge`.

Tasks:
- Add `@MegatronModelBridge.register_bridge(source=<HFClass>, target=GPTModel)`.
- Implement `hf_config_to_model_config_kwargs(self, hf_config)` for stock GPT construction, or `model_config_bridge(self, hf_pretrained)` plus a standalone model config/builder for custom construction.
- Keep model configs pure and serializable; do not inherit or instantiate `ModelProviderMixin`.
- If any of your config fields aren't bijective through `CONFIG_MAPPING` (e.g. MoE topology, sliding window, custom RoPE), override `megatron_to_hf_config` to reconstruct them for auto-config export.
- Implement `mapping_registry(self)` returning `MegatronMappingRegistry(...)` with:
  - `AutoMapping` for embeddings, final norm, output layer, 1:1 mapped weights.
  - `QKVMapping` for fused QKV if applicable.
  - `GatedMLPMapping` for gate/up if applicable.
- Use `*` wildcards consistently between Megatron and HF patterns.

References:
- `src/megatron/bridge/models/conversion/model_bridge.py`
- `src/megatron/bridge/models/conversion/mapping_registry.py`
- `src/megatron/bridge/models/conversion/param_mapping.py`
- `src/megatron/bridge/models/qwen/qwen2_bridge.py`

Acceptance:
- HF → Megatron load completes with no missing parameters (for a tiny model).
- Megatron → HF export returns tensors with expected shapes/dtypes for several keys.
```

## 5) Minimal smoke test (local)

A minimal bidirectional end-to-end check:
```python
from megatron.bridge import AutoBridge

# HF → Megatron
bridge = AutoBridge.from_hf_pretrained("<org>/<model-id>", trust_remote_code=True)
model_config = bridge.get_model_config()
model_config.tensor_model_parallel_size = 1
model_config.pipeline_model_parallel_size = 1
model_config.finalize()
model = bridge.get_megatron_model(model_config, wrap_with_ddp=False)

# Megatron → HF (stream a few tensors)
for i, (name, tensor) in enumerate(bridge.export_hf_weights(model, cpu=True)):
    print(name, tuple(tensor.shape))
    if i > 10:
        break
```


## 6) Validate with examples
Use the examples in `examples/conversion/` to verify bidirectional conversion and basic generation with more complex model parallel setups. 

- Generate from HF directly with the bridge
- Convert checkpoints back and forth
- Multi-GPU HF load to Megatron

```sh
uv run python examples/conversion/hf_to_megatron_generate_text.py --hf_model_path <org>/<model-id> --prompt "Hello"
uv run python examples/conversion/convert_checkpoints.py import --hf-model <org>/<model-id> --megatron-path ./checkpoints/<model-dir>
```
## 7) Add tests

Add or extend tests under `tests/functional_tests/test_groups/models/<your_model>/` and `tests/unit_tests/models/`:

Tests are organized in model-specific subdirectories that mirror the source structure in `src/megatron/bridge/models/`.

- Conversion coverage:
  - HF → Megatron load succeeds without missing params
  - Megatron → HF export round-trips shapes and dtypes
- Model-config coverage:
  - Config fields align with HF config (heads, groups, FFN size, RoPE)
  - Config serialization restores the same builder and exact nested MCore config type
- Optional numeric checks:
  - Forward parity on a handful of tokens comparing HF vs Megatron outputs

Examples to reference:
- `tests/unit_tests/models/qwen/test_qwen3_bridge.py`
- `tests/functional_tests/test_groups/models/qwen/test_qwen3_conversion.py`

Run fast tests locally:
```sh
uv run python -m pytest -q tests/unit_tests/models/<your_model>/ -k your_model | cat
uv run python -m pytest -q tests/functional_tests/test_groups/models/<your_model>/test_<your_model>_conversion.py -k your_model | cat
```

### 7.1) Model not found in CI Cache

Megatron Bridge functional tests run with `HF_HUB_OFFLINE=1`. This means that contributions including a new bridge and tests
for a HuggingFace model that is not cached in our CI's `$HF_HOME` directory will fail with an error similar to:

```
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled.
```

If such an error is encountered in the CI, please request a repo maintainer to launch the 'Cache HuggingFace model' workflow for the model(s)
you are adding support for in your PR.

### Suggested Cursor prompt (Tests) [Expermental]
```text
You are working in the Megatron Bridge repo. Add tests for a new model `<your_model>`.

Create unit coverage under `tests/unit_tests/models/<your_model>/` and functional conversion coverage under `tests/functional_tests/test_groups/models/<your_model>/`:
1) `test_<your_model>_bridge.py` and `test_<your_model>_model_config.py`
   - Build a tiny HF model/config (or use `<org>/<tiny-model-id>` if available).
   - Use the bridge to derive a builder-backed model config and construct the model with TP=PP=1.
   - Assert config fields match HF config and serialization preserves the builder and exact nested MCore config type.

2) `test_<your_model>_conversion.py`
   - HF → Megatron: load HF weights into the Megatron model via the bridge; assert no missing/extra params.
   - Megatron → HF: export a subset of tensors; assert shape/dtype parity with HF.
   - Optionally run a short generation on CPU and compare logits numerically within tolerance.

Use `tests/unit_tests/models/qwen/test_qwen3_bridge.py` and
`tests/functional_tests/test_groups/models/qwen/test_qwen3_conversion.py` as templates.

Provide `-k your_model` selectors and guard long tests with `pytest.skip` if external weights are unavailable.
```


## 8) Troubleshooting

- Shape mismatches: double-check TP/PP splits and model configs.
- Missing weights: ensure every Megatron param has a mapping; print unresolved names.
- Dtype issues: cast HF weights to destination dtype inside mappings when needed.
- EP/MoE layers: see EP-specific gather/scatter helpers in `param_mapping.py`.

Enable verbose logs:
```python
import logging
logging.getLogger("megatron.bridge").setLevel(logging.DEBUG)
```


## 9) PR checklist

- Provide details in the PR description
- The bridge maps all required fields into a serializable model config
- All parameters are covered by mappings
- Generation results after conversion from HF to Megatron match Megatron, including multi-GPU runs
- Unit/functional tests added and green
- Add your model to the Supported Models table in the repo `README.md` if applicable


## 10) Useful links

- User guide: [docs/bridge-guide.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-guide.md)
- Technical deep-dive: [docs/bridge-tech-details.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md)
- Code examples: [examples/conversion/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/conversion)
- Providers and bridges: [src/megatron/bridge/models/](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models)
- GitHub source tree: [Megatron Bridge src/megatron/bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge)
