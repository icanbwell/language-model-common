from __future__ import annotations

from dataclasses import dataclass

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)


@dataclass(frozen=True, slots=True)
class DebugCommandEffect(SlashCommandEffect):
    stripped_content: str


class DebugCommandHandler:
    def __init__(self, *, command_names: set[str] | None = None) -> None:
        self._command_names = command_names or {"debug"}

    @property
    def command_names(self) -> set[str]:
        return self._command_names

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        return DebugCommandEffect(stripped_content=context.remaining_message)
