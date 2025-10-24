from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig 
from transformers import Qwen3VLMoeVisionModel, Qwen3VLMoeConfig


class Qwen3VLVisionModel(MegatronModule):

    def __init__(
        self, 
        config: TransformerConfig,
        pretrained_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ):
        """Initialize Qwen3VL Vision Model with random weights.
        
        Args:
            config: Transformer configuration
            pretrained_model_name: HuggingFace model name or local path (used only for config)
        """
        super().__init__(config)
        
        # Load the full model config to extract vision config
        full_config = Qwen3VLMoeConfig.from_pretrained(pretrained_model_name)
        
        # Extract vision config from the full model config
        vision_config = full_config.vision_config
        
        # Initialize vision model with random weights from config
        self.vision_model = Qwen3VLMoeVisionModel._from_config(vision_config)
        
        # Move to device if available
        if torch.cuda.is_available():
            self.vision_model = self.vision_model.to('cuda')

    def forward(self,  hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            

        """
        image_embeds, deepstack_feature_lists = self.vision_model(hidden_states, grid_thw, **kwargs)
        # split_sizes = (grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        # image_embeds = torch.split(image_embeds, split_sizes)
        # image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds, deepstack_feature_lists

