from __future__ import annotations


from languagemodelcommon.utilities.slash_command.handlers.debug_command_handler import (
    DebugCommandHandler,
    DebugCommandEffect,
)
from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
)


class TestDebugCommandHandler:
    def test_handles_debug_command(self) -> None:
        handler = DebugCommandHandler()
        context = SlashCommandContext(
            command_name="debug",
            remaining_message="hello",
            original_content="/debug hello",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == "hello"

    def test_command_names_includes_debug(self) -> None:
        handler = DebugCommandHandler()
        assert "debug" in handler.command_names

    def test_handles_custom_command_names(self) -> None:
        handler = DebugCommandHandler(command_names={"debug", "verbose"})
        context = SlashCommandContext(
            command_name="verbose",
            remaining_message="test",
            original_content="/verbose test",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == "test"

    def test_empty_remaining_message(self) -> None:
        handler = DebugCommandHandler()
        context = SlashCommandContext(
            command_name="debug",
            remaining_message="",
            original_content="/debug",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, DebugCommandEffect)
        assert effect.stripped_content == ""
