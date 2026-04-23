import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from langchain_ai_skills_framework.langchain.tools.load_skill_tool import LoadSkillTool
from langchain_ai_skills_framework.langchain.tools.read_skill_resource_tool import (
    ReadSkillResourceTool,
)
from langchain_ai_skills_framework.langchain.tools.run_python_script_tool import (
    RunPythonScriptTool,
)
from langchain_ai_skills_framework.langchain.tools.run_skill_script_tool import RunSkillScriptTool
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
            if display_name
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

        name_for_tool: str = self.get_name_for_tool(
            tool_name=tool_name, tool_input=tool_input
        )
        if tool_name == "load_skill":
            return f"\n🧠 Using skill: {name_for_tool}.\n"
        elif tool_name == "run_skill_script":
            return f"\n⚡ Running script from skill: {name_for_tool}.\n"
        elif tool_name == "read_skill_resource":
            return f"\n📖 Reading resource from skill: {name_for_tool}.\n"
        elif tool_name == "run_python_script":
            return f"\n🐍 Running Python script: {name_for_tool}.\n"
        else:
            return f"\n{name_for_tool}.\n"

    def get_name_for_tool(
        self, *, tool_name: str | None, tool_input: Dict[str, Any] | None
    ) -> str:
        if not tool_name:
            return ""

        # TODO: Come up with a better way so we don't have to hardcode these
        if tool_name == "load_skill":
            display_name = LoadSkillTool.get_friendly_name(tool_input=tool_input or {})
        elif tool_name == "run_skill_script":
            display_name = RunSkillScriptTool.get_friendly_name(
                tool_input=tool_input or {}
            )
        elif tool_name == "read_skill_resource":
            display_name = ReadSkillResourceTool.get_friendly_name(
                tool_input=tool_input or {}
            )
        elif tool_name == "run_python_script":
            display_name = RunPythonScriptTool.get_friendly_name(
                tool_input=tool_input or {}
            )
        else:
            display_name = self.get_display_name(tool_name=tool_name)
        return f"{display_name}"
