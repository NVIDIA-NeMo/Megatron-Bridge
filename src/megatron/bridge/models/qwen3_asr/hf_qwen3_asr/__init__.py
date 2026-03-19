# Register with transformers Auto classes (replaces qwen_asr.inference registration)
from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig, Qwen3ASRThinkerConfig
from .modeling_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
