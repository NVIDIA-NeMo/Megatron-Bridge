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

import torch
from rich.progress import track
from transformers import AutoTokenizer

from megatron.hub import CausalLMBridge


HF_MODEL_ID = "meta-llama/Llama-3.2-1B"
bridge = CausalLMBridge.from_pretrained(HF_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)


def generate_sequence(prompt, model, max_new_tokens=100):
    """Generate text sequence"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.cuda()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)

    generated_tokens = []
    cur_input_ids = input_ids
    cur_position_ids = position_ids
    cur_attention_mask = attention_mask

    for _ in track(range(max_new_tokens), description="Generating..."):
        # Move inputs to GPU
        cur_input_ids = cur_input_ids.cuda()
        cur_position_ids = cur_position_ids.cuda()
        cur_attention_mask = cur_attention_mask.cuda()

        # Forward inference with the model
        with torch.no_grad():
            model[0].cuda()
            output = model[0].module(cur_input_ids, cur_position_ids, cur_attention_mask)

        # Get the next token
        next_token = output.argmax(dim=-1)[:, -1]

        # Debug: print output statistics
        if _ < 10:  # Only for first few iterations
            print(f"\nStep {_}: input shape={cur_input_ids.shape}, pos_ids={cur_position_ids[0].tolist()}")
            print(f"Output shape: {output.shape}, var={output.var():.4f}")
            # Get top 5 predictions
            logits = output[0, -1, :]
            top5_vals, top5_ids = torch.topk(logits, 5)
            top5_tokens = [tokenizer.decode([idx]) for idx in top5_ids]
            print(f"Top 5: {list(zip(top5_tokens, top5_vals.tolist()))}")
            print(f"Selected: '{tokenizer.decode([next_token.item()])}' (id={next_token.item()})")
        generated_tokens.append(next_token.item())

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Update input sequence
        cur_input_ids = torch.cat([input_ids, torch.tensor([generated_tokens], device=input_ids.device)], dim=1)
        cur_position_ids = torch.arange(cur_input_ids.shape[1], device=cur_input_ids.device).unsqueeze(0)
        cur_attention_mask = torch.ones_like(cur_input_ids, dtype=torch.bool)

    # Decode the generated token sequence
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated text:\n{generated_text}")
    return generated_text


if __name__ == "__main__":
    megatron_model = bridge.to_model(wrap_with_ddp=False)

    prompt = "Hello, how are you?"
    generate_sequence(prompt, megatron_model, HF_MODEL_ID)
