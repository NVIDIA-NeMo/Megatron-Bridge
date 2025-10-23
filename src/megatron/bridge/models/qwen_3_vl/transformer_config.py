from dataclasses import dataclass, field
from typing import List
from copy import deepcopy
from functools import partial
from typing import Callable
import torch
import torch.nn.functional as F
import functools
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class Qwen3VLTransformerConfig(TransformerConfig):

    vocab_size: int = 64000
    language_max_sequence_length: int  = 2048# Language model maximum sequence length. This is used for positional embedding.


    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 10000
    
    # Multimodal rope section for [temporal, height, width] dimensions
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
