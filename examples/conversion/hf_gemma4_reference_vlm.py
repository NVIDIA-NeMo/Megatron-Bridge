"""HuggingFace reference inference for Gemma4 VLM — used to compare against Bridge output.

Supports both base model (google/gemma-4-26B-A4B) and IT model (google/gemma-4-26B-A4B-it).
"""

import argparse

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

IMAGE_URL = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png"
PROMPT = "What is shown in this image?"
MAX_NEW_TOKENS = 200

parser = argparse.ArgumentParser()
parser.add_argument("--hf_model_path", default="google/gemma-4-26B-A4B-it")
args = parser.parse_args()

print(f"Loading model: {args.hf_model_path}")
processor = AutoProcessor.from_pretrained(args.hf_model_path)
model = AutoModelForImageTextToText.from_pretrained(
    args.hf_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_URL},
            {"type": "text", "text": PROMPT},
        ],
    }
]

# Try the preferred apply_chat_template(tokenize=True, return_dict=True) path first.
# Falls back to image_token prepend for base models without a chat template.
try:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    print("Used processor.apply_chat_template (tokenize=True)")
except (ValueError, AttributeError):
    print("No processor chat template — using image_token fallback")
    image_token = getattr(processor, "image_token", None)
    if image_token is None:
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is not None:
            image_token = processor.tokenizer.decode([image_token_id])
    text = (image_token or "") + PROMPT
    image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

# Move all tensors to the model's first device
first_device = next(model.parameters()).device
inputs = {k: v.to(first_device) if hasattr(v, "to") else v for k, v in inputs.items()}

input_len = inputs["input_ids"].shape[-1]
print(f"Input length: {input_len} tokens")
print(f"Pixel values shape: {inputs['pixel_values'].shape}")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
print("\n===== HF REFERENCE OUTPUT =====")
print(f"Model: {args.hf_model_path}")
print(f"Prompt: {PROMPT}")
print(f"Image: {IMAGE_URL}")
print(f"Response: {response}")
print("================================")
