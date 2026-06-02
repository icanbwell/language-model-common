from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SlashCommandContext:
    command_name: str
    remaining_message: str
    original_content: str


class SlashCommandEffect:
    """Base class for effects produced by command handlers."""


@runtime_checkable
class SlashCommandHandler(Protocol):
    @property
    def command_names(self) -> set[str]: ...

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None: ...
