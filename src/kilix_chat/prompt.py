"""Turn an OpenAI-style conversation into the one system and one user turn
that llama-diffusion-cli accepts, and read tool calls back out of a reply.

llama-diffusion-cli always applies the model's chat template to exactly one
system message and one user message. Earlier turns and tool results are
written into that user message as a labelled transcript.
"""
from __future__ import annotations

import json
import re
import uuid

TOOL_INSTRUCTIONS = (
    "You can call tools. To call a tool, reply with only this, and nothing else:\n"
    '<tool_call>{"name": "TOOL_NAME", "arguments": {...}}</tool_call>\n'
    "You may make several calls, one tag each. After the tools answer you will "
    "see their results. When you need no tool, reply normally.\n"
    "Available tools:\n"
)


def text_of(content):
    """OpenAI content may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def tool_catalog(tools):
    rows = []
    for tool in tools or []:
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            continue
        rows.append(json.dumps({"name": spec["name"], "description": spec.get("description", ""),
                                "parameters": spec.get("parameters", {})}, ensure_ascii=False))
    return "\n".join(rows)


def _call_text(call):
    function = call.get("function", {}) if isinstance(call, dict) else {}
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return f'<tool_call>{{"name": {json.dumps(function.get("name", ""))}, "arguments": {arguments}}}</tool_call>'


def _line(message):
    role = message.get("role")
    content = text_of(message.get("content")).strip()
    if role == "assistant":
        calls = "\n".join(_call_text(c) for c in message.get("tool_calls") or [])
        return "Assistant: " + "\n".join(filter(None, (content, calls)))
    if role == "tool":
        name = message.get("name") or message.get("tool_name") or message.get("tool_call_id") or "tool"
        return f"Tool result ({name}):\n{content}"
    return "User: " + content


def flatten(messages, tools=None):
    """Return (system, user) for llama-diffusion-cli."""
    systems, turns = [], []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") in ("system", "developer") and not turns:
            systems.append(text_of(message.get("content")).strip())
        else:
            turns.append(message)
    catalog = tool_catalog(tools)
    if catalog:
        systems.append(TOOL_INSTRUCTIONS + catalog)
    system = "\n\n".join(s for s in systems if s)
    if len(turns) == 1 and turns[0].get("role") == "user":
        return system, text_of(turns[0].get("content"))
    if not turns:
        return system, ""
    *earlier, last = turns
    parts = []
    if earlier:
        parts.append("The conversation so far:\n\n" + "\n\n".join(_line(m) for m in earlier))
    if last.get("role") == "user":
        parts.append("Reply as the assistant to the user's latest message:\n\n" + text_of(last.get("content")).strip())
    else:
        parts.append("Latest message:\n\n" + _line(last) + "\n\nReply as the assistant.")
    return system, "\n\n".join(parts)


_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def _decode_call(text):
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
        return None
    arguments = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": value["name"], "arguments": json.dumps(arguments, ensure_ascii=False)}}


def tool_calls(reply, tools):
    """Return (content, calls). Only names from the offered tools are accepted."""
    names = {(t.get("function", t) or {}).get("name") for t in tools or [] if isinstance(t, dict)}
    if not names:
        return reply, []
    calls = []
    for match in _TAGGED.finditer(reply):
        call = _decode_call(match.group(1))
        if call and call["function"]["name"] in names:
            calls.append(call)
    if calls:
        return _TAGGED.sub("", reply).strip(), calls
    stripped = reply.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        call = _decode_call(stripped)
        if call and call["function"]["name"] in names:
            return "", [call]
    return reply, []
