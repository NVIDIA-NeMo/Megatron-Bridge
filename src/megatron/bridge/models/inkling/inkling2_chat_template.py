# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Chat template for Inkling2 SFT data.

This is the ``thinkingmachines/Inkling-2-Small`` release's ``chat_template.jinja`` (revision
``78cb6c0e``, Apache-2.0) with two additions. An assistant message's reasoning may also be in ``thinking`` (the field Bridge's dataset
adapters write) or ``reasoning``; the first of ``reasoning_content``, ``thinking`` and ``reasoning`` that is set
is rendered, the precedence Transformers itself applies from version 5.22 before it renders a template. And
each assistant turn is marked as a ``{% generation %}`` span for assistant-only loss masks. Every conversation
that uses only ``reasoning_content`` renders byte for byte as the release's template does.

The template frames each block as ``<|message_ROLE|>[name]<|content_KIND|>payload<|end_message|>``,
declares tools first, writes a ``Thinking effort level`` system line (the ``reasoning_effort`` argument,
0.9 by default) before the first non-system message, and ends every assistant turn with
``<|content_model_end_sampling|>``. The model writes its own ``<|message_model|>`` headers, so the whole
assistant turn is the supervised span and the template adds no generation prompt.
"""

INKLING2_CHAT_TEMPLATE_REVISION = "78cb6c0ef68e1f089b77e78cfd94783d05b5c41b"  # pragma: allowlist secret
INKLING2_CHAT_TEMPLATE = """{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}
{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"} -%}

{%- macro emit_thinking_effort() -%}
    {%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
    {%- if eff is string -%}
        {%- set key = eff | trim -%}
        {%- if key not in effort_map -%}
            {{- raise_exception("Unknown reasoning_effort: " ~ eff) -}}
        {%- endif -%}
        {%- set num = effort_map[key] -%}
    {%- else -%}
        {%- set num = eff | float -%}
    {%- endif -%}
    {%- if num < 0.0 or num > 0.99 -%}
        {{- raise_exception("reasoning_effort must be in [0.0, 0.99]") -}}
    {%- endif -%}
    {{- "<|message_system|><|content_text|>Thinking effort level: " -}}
    {%- if num == 0.0 -%}0{%- else -%}{{ num }}{%- endif -%}
    {{- "<|end_message|>" -}}
{%- endmacro -%}

{%- macro render_content(message, rtok, reasoning) -%}
    {%- if message.content is string and (message.content or not (message.role == "assistant" and (message.tool_calls or reasoning))) -%}
        {{- rtok ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}
    {%- elif message.content -%}
        {%- for part in message.content -%}
            {%- if part is string -%}
                {{- rtok ~ "<|content_text|>" ~ part ~ "<|end_message|>" -}}
            {%- elif part.type is not defined or part.type in ("text", "input_text") -%}
                {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
                {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
            {%- elif part.type in ("image", "input_image", "image_url") -%}
                {%- if message.role not in ("user", "tool") -%}
                    {{- raise_exception("Image content is only supported for user and tool messages") -}}
                {%- endif -%}
                {{- rtok ~ "<|content_image|><|unused_200054|><|end_message|>" -}}
            {%- elif part.type in ("audio", "input_audio", "audio_url") -%}
                {{- raise_exception("Inkling 2 does not accept audio input") -}}
            {%- else -%}
                {{- raise_exception("Unsupported content part type: " ~ part.type) -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}

{%- if tools -%}
    {%- set tool_state = namespace(specs=[]) -%}
    {%- for tool in tools -%}
        {%- set fn = tool.function if tool.function is defined else tool -%}
        {%- set spec = {
            "description": (fn.description if fn.description is defined and fn.description else ""),
            "name": fn.name,
            "parameters": (fn.parameters if fn.parameters is defined and fn.parameters else {}),
            "type": (tool.type if tool.type is defined and tool.type else "function"),
        } -%}
        {%- set tool_state.specs = tool_state.specs + [spec] -%}
    {%- endfor -%}
    {{- "<|message_system|>tool_declare<|content_xml|>" -}}
    {{- tool_state.specs | tojson(sort_keys=true, separators=(",", ":")) -}}
    {{- "<|end_message|>" -}}
{%- endif -%}

{%- set state = namespace(effort_emitted=false) -%}
{%- for message in messages -%}
    {%- if message.role not in role_token -%}
        {{- raise_exception("Unknown message role: " ~ message.role) -}}
    {%- endif -%}
    {%- if not state.effort_emitted and message.role != "system" -%}
        {{- emit_thinking_effort() -}}
        {%- set state.effort_emitted = true -%}
    {%- endif -%}

    {%- set header = namespace(text=role_token[message.role]) -%}
    {%- if message.role == "tool" -%}
        {%- if message.name is defined and message.name -%}
            {%- set header.text = header.text ~ message.name -%}
        {%- elif message.tool_call_id is defined and message.tool_call_id -%}
            {%- for prev in messages -%}
                {%- if prev.role == "assistant" and prev.tool_calls -%}
                    {%- for tc in prev.tool_calls -%}
                        {%- if tc.id is defined and tc.id == message.tool_call_id and tc.function.name is defined -%}
                            {%- set header.text = header.text ~ tc.function.name -%}
                        {%- endif -%}
                    {%- endfor -%}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
    {%- elif message.role == "assistant" and message.name is defined and message.name -%}
        {%- set header.text = header.text ~ message.name -%}
    {%- endif -%}
    {%- set rtok = header.text -%}
    {%- if message.reasoning_content is defined and message.reasoning_content is not none -%}
        {%- set reasoning = message.reasoning_content -%}
    {%- elif message.thinking is defined and message.thinking is not none -%}
        {%- set reasoning = message.thinking -%}
    {%- elif message.reasoning is defined and message.reasoning is not none -%}
        {%- set reasoning = message.reasoning -%}
    {%- else -%}
        {%- set reasoning = none -%}
    {%- endif -%}

    {%- if message.role == "assistant" -%}
    {%- generation -%}
    {%- if reasoning -%}
        {{- rtok ~ "<|content_thinking|>" ~ reasoning ~ "<|end_message|>" -}}
    {%- endif -%}
    {{- render_content(message, rtok, reasoning) -}}
    {%- if message.role == "assistant" and message.tool_calls -%}
        {%- for tc in message.tool_calls -%}
            {%- set fn = tc.function -%}
            {%- if fn.name is not defined or fn.name is not string -%}
                {{- raise_exception("tool call function name must be a string") -}}
            {%- endif -%}
            {%- if message.name is defined and message.name and message.name != fn.name -%}
                {{- raise_exception("assistant name " ~ message.name ~ " does not match its tool call " ~ fn.name) -}}
            {%- endif -%}
            {%- set args = fn.arguments if fn.arguments is defined and fn.arguments else {} -%}
            {%- if args is string -%}
                {{- raise_exception("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream") -}}
            {%- endif -%}
            {%- if args is not mapping -%}
                {{- raise_exception("tool call arguments must be an object") -}}
            {%- endif -%}
            {{- "<|message_model|>" ~ fn.name ~ "<|content_invoke_tool_json|>" -}}
            {{- '{"name":' ~ (fn.name | tojson(sort_keys=true, separators=(",", ":"))) ~ ',"args":' -}}
            {{- (args | tojson(sort_keys=true, separators=(",", ":"))) -}}
            {{- "}<|end_message|>" -}}
        {%- endfor -%}
    {%- endif -%}

    {{- "<|content_model_end_sampling|>" -}}
    {%- endgeneration -%}
    {%- else -%}
    {{- render_content(message, rtok, reasoning) -}}
    {%- endif -%}
{%- endfor -%}

{%- if not state.effort_emitted -%}
    {{- emit_thinking_effort() -}}
{%- endif -%}
"""


__all__ = ["INKLING2_CHAT_TEMPLATE", "INKLING2_CHAT_TEMPLATE_REVISION"]
