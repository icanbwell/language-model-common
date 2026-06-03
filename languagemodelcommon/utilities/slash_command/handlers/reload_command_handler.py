from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


class ReloadTarget(Enum):
    MODELS = "models"
    SKILLS = "skills"


@dataclass(frozen=True, slots=True)
class ReloadCommandEffect(SlashCommandEffect):
    target: ReloadTarget


class ReloadCommandHandler:
    _COMMAND_MAP: dict[str, ReloadTarget] = {
        "reload": ReloadTarget.MODELS,
        "reload-skills": ReloadTarget.SKILLS,
    }

    @property
    def command_names(self) -> set[str]:
        return set(self._COMMAND_MAP.keys())

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        target = self._COMMAND_MAP.get(context.command_name)
        if target is None:
            return None
        return ReloadCommandEffect(target=target)
