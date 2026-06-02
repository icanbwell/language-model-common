from __future__ import annotations

from typing import Sequence

from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
    SlashCommandHandler,
)
from languagemodelcommon.utilities.slash_command.slash_command_parser import (
    parse_slash_command,
)


class SlashCommandProcessor:
    def __init__(self, *, handlers: Sequence[SlashCommandHandler]) -> None:
        self._handlers = list(handlers)

    def process(self, *, content: str) -> SlashCommandEffect | None:
        match = parse_slash_command(content=content)
        if match is None:
            return None

        context = SlashCommandContext(
            command_name=match.command_name,
            remaining_message=match.remaining_message,
            original_content=content,
        )

        for handler in self._handlers:
            if match.command_name in handler.command_names:
                effect = handler.handle(context=context)
                if effect is not None:
                    return effect

        return None
