import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS
from languagemodelcommon.utilities.text_humanizer import Humanizer

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.TOOLS)


class ToolFriendlyNameMapper:
    """Provide user-facing tool names for streaming progress updates."""

    def __init__(self, *, name_to_display_name: Dict[str, str] | None = None) -> None:
        self._name_to_display_name: Dict[str, str] = {
            name: display_name
            for name, display_name in (name_to_display_name or {}).items()
            if display_name
        }

    @classmethod
    def from_mapping(
        cls, *, name_to_display_name: Mapping[str, str]
    ) -> "ToolFriendlyNameMapper":
        return cls(name_to_display_name=dict(name_to_display_name))

    @classmethod
    def from_config_path(cls, *, config_path: str | None) -> "ToolFriendlyNameMapper":
        if not config_path:
            return cls()
        path = Path(config_path)
        if not path.exists():
            logger.warning(
                "Tool friendly name config path does not exist: %s",
                path,
            )
            return cls()
        try:
            raw_text = path.read_text(encoding="utf-8")
            data: Any = json.loads(raw_text)
        except Exception:
            logger.exception(
                "Failed to load tool friendly name config from %s",
                path,
            )
            return cls()
        if not isinstance(data, dict):
            logger.warning(
                "Tool friendly name config must be a JSON object: %s",
                path,
            )
            return cls()
        mapping: Dict[str, str] = {
            str(key): value
            for key, value in data.items()
            if isinstance(value, str) and value.strip()
        }
        return cls(name_to_display_name=mapping)

    def get_display_name(self, *, tool_name: str) -> str:
        display_name = self._name_to_display_name.get(tool_name)
        if display_name:
            return display_name
        return "🛠️ " + Humanizer.humanize_tool_name(tool_name)

    def get_message_for_tool(
        self, *, tool_name: str | None, tool_input: Dict[str, Any] | None
    ) -> str:
        if not tool_name:
            return ""
        if tool_name == "load_skill":
            skill_name_value = tool_input.get("skill_name") if tool_input else None
            skill_name = Humanizer.humanize_tool_name(skill_name_value or "unknown")
            return f"\n🧠 Using skill: {skill_name}.\n"
        display_name = self.get_display_name(tool_name=tool_name)
        return f"\n{display_name}.\n"
