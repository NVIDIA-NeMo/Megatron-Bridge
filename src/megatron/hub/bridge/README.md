# Bridging between HuggingFace Transformers & Megatron


### Loading Llama from HuggingFace Hub
```python
from transformers import LlamaForCausalLM
from megatron.hub import CausalLMBridge, GPTModelProvider

bridge: CausalLMBridge[LlamaForCausalLM] = CausalLMBridge.from_pretrained("meta-llama/Llama-3.2-1B")

model_provider: GPTModelProvider = bridge.to_megatron()
model_provider_no_weights = bridge.to_megatron(load_weights=False)

# A model-provider is lazy, so we can do overwrites
model_provider.tensor_model_parallel_size = 8
```


### Loading a PEFT model
> TODO: Implement this
```python
from megatron.hub import CausalLMBridge
from megatron.hub.peft import Llora, get_peft_model_provider


bridge: CausalLMBridge[LlamaForCausalLM] = CausalLMBridge.from_pretrained("meta/llama3-8b")
model_provider: GPTModelProvider = bridge.to_megatron()

peft = Lora(r=8, lora_alpha=32, lora_dropout=0.1)
peft_model_provider = get_peft_model_provider(model_provider, peft)
```