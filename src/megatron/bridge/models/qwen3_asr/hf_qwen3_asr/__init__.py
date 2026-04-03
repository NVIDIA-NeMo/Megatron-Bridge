# Why we vendor the HF model code here instead of `pip install qwen3-asr`:
#
# 1. Dependency conflict – the official qwen3-asr package pins its own
#    transformers / torch versions in pyproject.toml
#    (https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml#L23),
#    which conflicts with the versions required by Megatron-LM.
#
# 2. Missing source on HF Hub – the Qwen3-ASR HuggingFace model repo only
#    ships weights and config; it does not include the modeling / processing
#    source code, so `trust_remote_code=True` alone is not enough to load
#    the model without installing the package.
#
# To work around both issues we keep a local copy of the HF model code and
# register the Auto classes ourselves below.

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig, Qwen3ASRThinkerConfig
from .modeling_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
