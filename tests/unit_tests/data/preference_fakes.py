"""Chat-template stand-in shared by the preference dataset and DPO builder tests."""


class ChatMLTokenizer:
    """Character-level tokenizer that renders exactly like the Qwen2.5 ChatML template."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(c) for c in text]
