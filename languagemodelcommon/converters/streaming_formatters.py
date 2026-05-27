import json
from typing import Any, Dict, Optional

from langchain_core.messages import ToolMessage

from languagemodelcommon.utilities.text_humanizer import Humanizer


def make_tool_key(
    tool_name: Optional[str], tool_input: Optional[Dict[str, Any]]
) -> str:
    if tool_name is None:
        tool_name = "unknown"
    try:
        tool_input_str = json.dumps(tool_input, sort_keys=True, default=str)
    except Exception:
        tool_input_str = str(tool_input)
    return f"{tool_name}:{hash(tool_input_str)}"


def safe_json(string: str) -> Any:
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        return None


def convert_message_content_into_string(*, tool_message: ToolMessage) -> str:
    if isinstance(tool_message.content, str):
        return _format_text_resource_contents(text=tool_message.content)

    if (
        isinstance(tool_message.content, list)
        and len(tool_message.content) == 1
        and isinstance(tool_message.content[0], dict)
        and "text" in tool_message.content[0]
    ):
        text = tool_message.content[0]["text"]
        json_object: dict[str, Any] | None = safe_json(text)
        if json_object is not None and isinstance(json_object, dict):
            if "result" in json_object:
                return str(json_object.get("result"))

    return " ".join([str(c) for c in tool_message.content])


def get_structured_content_from_tool_message(
    *, tool_message: ToolMessage
) -> dict[str, Any] | None:
    if isinstance(tool_message.content, dict):
        return tool_message.content
    elif (
        isinstance(tool_message.content, list)
        and len(tool_message.content) == 1
        and isinstance(tool_message.content[0], dict)
    ):
        return tool_message.content[0]
    return None


def format_message_content(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def format_tool_input_labels(*, tool_input: Dict[str, Any] | None) -> str:
    if not tool_input:
        return "none"
    hidden_keys = {"auth_token", "state", "runtime"}
    labels: list[str] = []
    for key in tool_input.keys():
        if key in hidden_keys:
            continue
        labels.append(Humanizer.humanize_tool_name(key))
    return ", ".join(labels) if labels else "none"


def extract_reasoning_text(block: dict[str, Any]) -> str | None:
    block_type = block.get("type")
    if block_type == "reasoning_content":
        rc = block.get("reasoning_content", {})
        if isinstance(rc, dict):
            return rc.get("text")
    elif block_type == "reasoning":
        return block.get("reasoning")
    return None


def _format_text_resource_contents(text: str) -> str:
    result = ""
    json_object: Any = safe_json(text)
    if json_object is not None and isinstance(json_object, dict):
        if "result" in json_object:
            result += str(json_object.get("result")) + "\n"
        if "error" in json_object:
            result += "Error: " + str(json_object.get("error")) + "\n"
        if "meta" in json_object:
            meta = json_object.get("meta", {})
            if isinstance(meta, dict) and len(meta) > 0:
                result += "Metadata:\n"
                for key, value in meta.items():
                    result += f"- {key}: {value}\n"
        if "urls" in json_object:
            urls = json_object.get("urls", [])
            if isinstance(urls, list) and len(urls) > 0:
                result += "Related URLs:\n"
                for url in urls:
                    result += f"- {url}\n"
        if "result" not in json_object and "error" not in json_object:
            result += text + "\n"
    else:
        result += text + "\n"
    return result
