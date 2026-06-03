import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from langchain_core.tools import BaseTool
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.text_humanizer import Humanizer

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.TOOLS)


class ToolDisplayNameMapper:
    """Provide user-facing tool names for streaming progress updates."""

    def __init__(self, *, name_to_display_name: Dict[str, str] | None = None) -> None:
        self._name_to_display_name: Dict[str, str] = {
            name: display_name
            for name, display_name in (name_to_display_name or {}).items()
            if display_name is not None
        }

    @classmethod
    def from_mapping(
        cls, *, name_to_display_name: Mapping[str, str]
    ) -> "ToolDisplayNameMapper":
        return cls(name_to_display_name=dict(name_to_display_name))

    @classmethod
    def from_config_path(cls, *, config_path: str | None) -> "ToolDisplayNameMapper":
        if not config_path:
            return cls()
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Tool display name config path does not exist: %s",
                path,
            )
            return cls()
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: Any = json.loads(raw_text)
        except Exception:
            logger.exception(
                "Failed to load tool display name config from %s",
                path,
            )
            return cls()
        if not isinstance(data, dict):
            logger.warning(
                "Tool display name config must be a JSON object: %s",
                path,
            )
            return cls()
        mapping: Dict[str, str] = {
            str(key): value for key, value in data.items() if isinstance(value, str)
        }
        return cls(name_to_display_name=mapping)

    def register_from_tools(self, tools: Sequence[BaseTool]) -> None:
        """Populate display names from tool metadata.

        Extracts ``mcp_title`` from each tool's ``metadata`` dict
        (set during MCP-to-LangChain conversion) and registers it
        as the display name.  Entries already present in the static
        config are not overwritten — the config file always wins.
        """
        for tool in tools:
            if tool.name in self._name_to_display_name:
                continue
            if not tool.metadata:
                continue
            mcp_title = tool.metadata.get("mcp_title")
            if isinstance(mcp_title, str):
                stripped_title = mcp_title.strip()
                if stripped_title:
                    self._name_to_display_name[tool.name] = stripped_title

    @staticmethod
    def _starts_with_emoji(text: str) -> bool:
        if not text:
            return False
        first_char = text[0]
        cp = ord(first_char)
        return cp > 0x2000

    _PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")

    @classmethod
    def _substitute_params(cls, *, template: str, params: Dict[str, Any]) -> str:
        """Replace {param} or {param|default} placeholders with values from params.

        Supports: {name}, {name|fallback text}
        If a param is missing and no default is specified, the placeholder is left as-is.
        """
        if "{" not in template:
            return template

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if "|" in token:
                key, default = token.split("|", 1)
            else:
                key, default = token, None
            value = params.get(key)
            if value is not None:
                return str(value)
            if default is not None:
                return default
            return match.group(0)

        return cls._PLACEHOLDER_RE.sub(_replace, template)

    def get_display_name(
        self, *, tool_name: str, tool_input: Dict[str, Any] | None = None
    ) -> str:
        display_name = self._name_to_display_name.get(tool_name)
        if display_name is None:
            return "🛠️ " + Humanizer.humanize_tool_name(tool_name)
        if display_name == "":
            return ""
        if tool_input:
            display_name = self._substitute_params(
                template=display_name, params=tool_input
            )
        if self._starts_with_emoji(display_name):
            return display_name
        return "🛠️ " + display_name

    def get_message_for_tool(
        self, *, tool_name: str | None, tool_input: Dict[str, Any] | None
    ) -> str:
        if not tool_name:
            return ""

        name_for_tool: str = self.get_name_for_tool(
            tool_name=tool_name, tool_input=tool_input
        )
        if not name_for_tool:
            return ""

        if self._starts_with_emoji(name_for_tool):
            return f"\n{name_for_tool}.\n"
        return f"\n🛠️ {name_for_tool}.\n"

    def get_name_for_tool(
        self, *, tool_name: str | None, tool_input: Dict[str, Any] | None
    ) -> str:
        if not tool_name:
            return ""

        inputs = tool_input or {}
        if tool_name == "call_tool":
            return self._get_name_for_call_tool(inputs=inputs)
        return self.get_display_name(tool_name=tool_name, tool_input=inputs)

    def _get_name_for_call_tool(self, *, inputs: Dict[str, Any]) -> str:
        target_tool_name = str(inputs.get("name") or "")
        if not target_tool_name:
            return "Call Tool"
        configured = self._name_to_display_name.get(target_tool_name)
        if configured is None:
            return Humanizer.humanize_tool_name(key=target_tool_name)
        if configured == "":
            return ""
        tool_arguments = inputs.get("arguments") or {}
        return self._substitute_params(template=configured, params=tool_arguments)
