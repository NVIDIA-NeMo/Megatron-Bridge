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
Unit tests for Qwen3VL Model implementation.

Run with: torchrun --nproc_per_node=8 -m pytest tests/unit_tests/models/qwen_3_vl/test_model.py
Or for single GPU: pytest tests/unit_tests/models/qwen_3_vl/test_model.py
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_3_vl.transformer_block import Qwen3VLTransformerBlock as Qwen3VLTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoProcessor
from transformers import Qwen3VLMoeConfig


@pytest.fixture(scope="module")
def processor():
    """Load HuggingFace processor once for all tests."""
    return AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


@pytest.fixture(scope="module")
def hf_config():
    """Load HuggingFace config once for all tests."""
    return Qwen3VLMoeConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")


@pytest.fixture
def random_image():
    """Generate a random PIL image."""
    random_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(random_array)


class TestQwen3VLModel:
    """Test suite for Qwen3VL Model."""

    @classmethod
    def setup_class(cls):
        """Setup distributed process group once for all tests in this class."""
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        
        torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

    @classmethod
    def teardown_class(cls):
        """Teardown distributed process group once after all tests in this class."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _setup_parallel_state(self, tp_size=1, ep_size=1, pp_size=1, cp_size=1):
        """Setup Megatron parallel state with specified parallelism configuration.
        
        Args:
            tp_size: Tensor model parallel size
            ep_size: Expert model parallel size  
            pp_size: Pipeline model parallel size
            cp_size: Context parallel size
        """
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=1,
        )
        
        model_parallel_cuda_manual_seed(123)
    
    def teardown_method(self):
        """Teardown Megatron parallel state after each test method."""
        parallel_state.destroy_model_parallel()

    @staticmethod
    def get_vision_transformer_config():
        """Create a vision transformer config for testing.
        
        Returns:
            TransformerConfig: Configuration for the vision model.
        """
        return TransformerConfig(
            num_layers=2,  # not used, use HF model
            hidden_size=128,  # not used, use HF model
            num_attention_heads=8,  # not used, use HF model
            bf16=False,
            use_cpu_initialization=True,  # not used, use HF model
        )

    @staticmethod
    def get_language_transformer_config(hf_config):
        """Create a language transformer config for testing.
        
        Uses actual Qwen3-VL-30B-A3B model sizes to ensure compatibility
        with the vision model output (2048 hidden size).
        
        Args:
            hf_config: HuggingFace config object.
        
        Returns:
            Qwen3VLTransformerConfig: Configuration for the language model.
        """
        return Qwen3VLTransformerConfig(
            # Use actual model dimensions from HF config
            num_layers=4,  # Reduced for testing (actual: hf_config.text_config.num_hidden_layers)
            hidden_size=hf_config.text_config.hidden_size,  # Must match vision output: 2048
            num_attention_heads=hf_config.text_config.num_attention_heads,
            num_query_groups=hf_config.text_config.num_key_value_heads,
            kv_channels=hf_config.text_config.hidden_size // hf_config.text_config.num_attention_heads,
            ffn_hidden_size=hf_config.text_config.intermediate_size,
            
            # Qwen3-VL specific
            vocab_size=hf_config.text_config.vocab_size,
            language_max_sequence_length=hf_config.text_config.max_position_embeddings,
            
            # Vision parameters
            patch_size=hf_config.vision_config.patch_size,
            temporal_patch_size=hf_config.vision_config.temporal_patch_size,
            in_channels=hf_config.vision_config.in_channels,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
            out_hidden_size=hf_config.text_config.hidden_size,  # Vision output = language input
            
            # RoPE settings
            rotary_base=hf_config.text_config.rope_theta,
            rotary_percent=1.0,
            mrope_section=hf_config.text_config.rope_scaling.get("mrope_section", [16, 24, 24]),
            
            # Training settings
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=True,
            layernorm_epsilon=hf_config.text_config.rms_norm_eps,
            bf16=False,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=hf_config.text_config.attention_dropout,
        )

    @staticmethod
    def get_language_model_layer_spec():
        """Create a GPT layer spec for the language model.
        
        Returns:
            ModuleSpec: Layer specification for transformer layers.
        """
        language_model_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,  # No MoE for basic test
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        )
        return language_model_layer_spec

    @staticmethod
    def get_data_batch(processor, random_image):
        """Generate a batch of data for model forward pass.
        
        Args:
            processor: HuggingFace processor.
            random_image: Random PIL image.
        
        Returns:
            dict: A dictionary containing all inputs needed for model forward pass:
                - input_ids: Token IDs [batch, seq_len]
                - attention_mask: Attention mask [batch, seq_len]
                - pixel_values: Image pixel values [batch, channels, height, width]
                - image_grid_thw: Image grid dimensions [num_images, 3] (temporal, height, width)
                - pixel_values_videos: Video pixel values (None for images only)
                - video_grid_thw: Video grid dimensions (None for images only)
        """
        # Create a sample message with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": random_image,  # Pass PIL Image directly
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]

        # Process inputs using HuggingFace processor
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'pixel_values': inputs.get('pixel_values'),
            'image_grid_thw': inputs.get('image_grid_thw'),
            'pixel_values_videos': inputs.get('pixel_values_videos'),
            'video_grid_thw': inputs.get('video_grid_thw'),
            'position_ids': None,
            'labels': None,
        }
        
        # Move tensors to CUDA if available
        if torch.cuda.is_available():
            for key, value in batch.items():
                if value is not None and isinstance(value, torch.Tensor):
                    batch[key] = value.cuda()
        
        return batch

    @pytest.mark.parametrize("tp_size,ep_size,pp_size", [
        (1, 1, 1),  # No parallelism
        (2, 2, 1),  # tp=2, ep=2, pp=1
        (2, 2, 2),  # tp=2, ep=2, pp=2
    ])
    def test_model_forward(self, tp_size, ep_size, pp_size, hf_config, processor, random_image):
        """Test Qwen3VL model initialization and forward pass with different parallelism configs."""
        self._setup_parallel_state(tp_size=tp_size, ep_size=ep_size, pp_size=pp_size)
        
        # Verify parallel configuration
        assert parallel_state.get_tensor_model_parallel_world_size() == tp_size
        assert parallel_state.get_expert_model_parallel_world_size() == ep_size
        assert parallel_state.get_pipeline_model_parallel_world_size() == pp_size
        
        vision_transformer_config = self.get_vision_transformer_config()
        language_transformer_config = self.get_language_transformer_config(hf_config)
        language_model_layer_spec = self.get_language_model_layer_spec()

        model = Qwen3VLModel(
            vision_transformer_config=vision_transformer_config,
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_model_layer_spec,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
        )

        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")

        inputs = self.get_data_batch(processor, random_image)
        
        with torch.no_grad():
            output = model(**inputs)
        
        assert output is not None
        assert output.shape[0] == inputs['input_ids'].shape[0]