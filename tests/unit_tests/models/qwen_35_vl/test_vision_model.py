



'''
uv run ./tests/unit_tests/models/qwen_35_vl/test_vision_model.py
'''

import torch
from transformers import AutoProcessor
from unittest.mock import Mock
from megatron.bridge.models.qwen_3_vl.vision_model import Qwen3VLVisionModel


class TestVisionModel:

    def get_mock_transformer_config(self):
        """Get mock transformer config."""
        config = Mock()
        config.bf16 = True
        return config

    def test_vision_model(self):
        """Test vision model."""
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        model = Qwen3VLVisionModel(config = self.get_mock_transformer_config())
        model.eval()
        model.to("cuda")
        
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

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # These keys are what the vision tower expects
        pixel_values = inputs["pixel_values"].to("cuda")
        # Dynamic-resolution grid for Qwen2/3-VL families:
        image_grid_thw = inputs.get("image_grid_thw", None)  # present for non-square / high-res
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to("cuda")
        
        print(f"pixel_values shape is {pixel_values.shape}")
        print(f"image_grid_thw shape is {image_grid_thw.shape}")
 
        with torch.no_grad():
            hidden_states, deepstack_feature_lists  = model(hidden_states=pixel_values, grid_thw=image_grid_thw)
        
        print(f"output hidden state shape is {hidden_states.shape}")
        print(f"output deepstack feature lists shape is {len(deepstack_feature_lists)} first element shape is {deepstack_feature_lists[0].shape}")

if __name__ == "__main__":
    test_vision_model = TestVisionModel()
    test_vision_model.test_vision_model()