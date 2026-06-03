from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


@runtime_checkable
class SkillContentResolver(Protocol):
    def get_available_skill_names(self) -> set[str]: ...
    async def resolve(self, *, skill_name: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class SkillCommandEffect(SlashCommandEffect):
    skill_name: str
    remaining_message: str


class SkillCommandHandler:
    def __init__(self, *, resolver: SkillContentResolver) -> None:
        self._resolver = resolver

    @property
    def command_names(self) -> set[str]:
        return self._resolver.get_available_skill_names()

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        available = self._resolver.get_available_skill_names()
        if context.command_name not in available:
            return None

        return SkillCommandEffect(
            skill_name=context.command_name,
            remaining_message=context.remaining_message,
        )
