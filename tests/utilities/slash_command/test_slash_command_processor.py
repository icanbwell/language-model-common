from __future__ import annotations

from dataclasses import dataclass


from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
    SlashCommandEffect,
)
from languagemodelcommon.utilities.slash_command.slash_command_processor import (
    SlashCommandProcessor,
)


@dataclass
class _FakeEffect(SlashCommandEffect):
    handled_command: str = ""
    handled_remaining: str = ""


class _FakeHandler:
    def __init__(self, *, commands: set[str]) -> None:
        self._commands = commands

    @property
    def command_names(self) -> set[str]:
        return self._commands

    def handle(self, *, context: SlashCommandContext) -> SlashCommandEffect | None:
        return _FakeEffect(
            handled_command=context.command_name,
            handled_remaining=context.remaining_message,
        )


class TestSlashCommandProcessor:
    def test_processes_matching_command(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="/debug hello")

        assert effect is not None
        assert isinstance(effect, _FakeEffect)
        assert effect.handled_command == "debug"
        assert effect.handled_remaining == "hello"

    def test_returns_none_for_unmatched_command(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="/unknown hello")

        assert effect is None

    def test_returns_none_for_no_slash(self) -> None:
        handler = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler])

        effect = processor.process(content="hello world")

        assert effect is None

    def test_first_matching_handler_wins(self) -> None:
        handler1 = _FakeHandler(commands={"debug"})
        handler2 = _FakeHandler(commands={"debug"})
        processor = SlashCommandProcessor(handlers=[handler1, handler2])

        effect = processor.process(content="/debug hello")

        assert effect is not None

    def test_multiple_handlers_different_commands(self) -> None:
        handler1 = _FakeHandler(commands={"debug"})
        handler2 = _FakeHandler(commands={"skill"})
        processor = SlashCommandProcessor(handlers=[handler1, handler2])

        effect1 = processor.process(content="/debug hello")
        effect2 = processor.process(content="/skill load")

        assert effect1 is not None
        assert effect2 is not None
        assert isinstance(effect1, _FakeEffect)
        assert effect1.handled_command == "debug"
        assert isinstance(effect2, _FakeEffect)
        assert effect2.handled_command == "skill"
