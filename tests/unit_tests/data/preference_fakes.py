"""Chat-template stand-in shared by the preference dataset and DPO builder tests."""

import json


class ChatMLTokenizer:
    """Character-level tokenizer that renders exactly like the Qwen2.5 ChatML template."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, tools=None):
        text = f"<|im_start|>system\n<tools>{json.dumps(tools)}</tools><|im_end|>\n" if tools else ""
        text += "".join(f"<|im_start|>{m['role']}\n{self._body(m)}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(c) for c in text]

    @staticmethod
    def _body(message):
        calls = message.get("tool_calls") or []
        return message["content"] + "".join(f"<tool_call>\n{json.dumps(c['function'])}\n</tool_call>" for c in calls)
