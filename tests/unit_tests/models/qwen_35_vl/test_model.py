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
'''
uv run python -m torch.distributed.run --nproc_per_node=8 ./tests/unit_tests/models/qwen_35_vl/test_model.py
'''
import torch
import torch.nn.functional as F

from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_3_vl.transformer_block import Qwen3VLTransformerBlock as Qwen3VLTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoProcessor
from transformers import Qwen3VLMoeConfig


class TestQwen3VLModel:

    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
       



    def init_parallel_state(self, tp_size=1, cp_size=1, pp_size=1, ep_size=1, vpp_size=None,etp_size=None):
        """Initialize parallel state for testing."""
        
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(torch.distributed.get_rank() % torch.cuda.device_count())

        
        parallel_state.initialize_model_parallel(
           tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
        )

        model_parallel_cuda_manual_seed(123)

    def get_vision_transformer_config(self):
        """Create a vision transformer config for testing.
        
        Returns:
            TransformerConfig: Configuration for the vision model.
        """
        return TransformerConfig(
            num_layers=2, # not used, use HF model
            hidden_size=128,   # not used, use HF model
            num_attention_heads=8,  # not used, use HF model
            bf16=False,
            use_cpu_initialization=True, # not used, use HF model

        )

    def get_language_transformer_config(self):
        """Create a language transformer config for testing.
        
        Uses actual Qwen3-VL-30B-A3B model sizes to ensure compatibility
        with the vision model output (2048 hidden size).
        
        Returns:
            Qwen3VLTransformerConfig: Configuration for the language model.
        """
        # Load actual Qwen3-VL config to get real dimensions

        hf_config = Qwen3VLMoeConfig.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        
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

    def get_language_model_layer_spec(self):
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

    def get_data_batch(self):
        """Generate a batch of data for model forward pass.
        
        Returns:
            dict: A dictionary containing all inputs needed for model forward pass:
                - input_ids: Token IDs [batch, seq_len]
                - attention_mask: Attention mask [batch, seq_len]
                - pixel_values: Image pixel values [batch, channels, height, width]
                - image_grid_thw: Image grid dimensions [num_images, 3] (temporal, height, width)
                - pixel_values_videos: Video pixel values (None for images only)
                - video_grid_thw: Video grid dimensions (None for images only)
        """
        # Use cached processor
        processor = self.processor
        
        # Create a sample message with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]

        # Process inputs using HuggingFace processor
        # This returns a BatchFeature (dict-like) with all necessary keys
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        batch = {
            'input_ids': inputs['input_ids'],                        # Required
            'attention_mask': inputs.get('attention_mask'),          # Optional
            'pixel_values': inputs.get('pixel_values'),              # Required for vision
            'image_grid_thw': inputs.get('image_grid_thw'),          # Required for vision
            'pixel_values_videos': inputs.get('pixel_values_videos'), # None for images only
            'video_grid_thw': inputs.get('video_grid_thw'),          # None for images only
            'position_ids': None,  # Will be computed in model
            'labels': None,        # No labels for inference
        }
        
        # Move tensors to CUDA if available
        if torch.cuda.is_available():
            for key, value in batch.items():
                if value is not None and isinstance(value, torch.Tensor):
                    batch[key] = value.cuda()
        
        return batch


    def test_qwen3_vl_model_init(self):
        """Test Qwen3VL model initialization."""
        
        self.init_parallel_state()

        
        vision_transformer_config = self.get_vision_transformer_config()
        language_transformer_config = self.get_language_transformer_config()
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

    def test_qwen35_vl_test_fwd_pass(self):
        """Test Qwen3VL model forward pass."""

        self.init_parallel_state()

        vision_transformer_config = self.get_vision_transformer_config()
        language_transformer_config = self.get_language_transformer_config()
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
        model.to("cuda")

        inputs = self.get_data_batch()
        print(f"[rank {torch.distributed.get_rank()}] [test_qwen35_vl_test_fwd_pass] input_ids {inputs['input_ids'].shape} pixel_values {inputs['pixel_values'].shape}")
        output = model(**inputs)
        print(f"[rank {torch.distributed.get_rank()}] [test_qwen35_vl_test_fwd_pass] output shape  {output.shape}")




if __name__ == "__main__":
    test_qwen3_vl_model = TestQwen3VLModel()
    test_qwen3_vl_model.test_qwen3_vl_model_init()

