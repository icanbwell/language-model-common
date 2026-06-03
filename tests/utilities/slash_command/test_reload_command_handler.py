from __future__ import annotations

import pytest

from languagemodelcommon.utilities.slash_command import (
    ReloadCommandEffect,
    ReloadCommandHandler,
    ReloadTarget,
    SlashCommandContext,
)


class TestReloadCommandHandler:
    def test_command_names_contains_reload_and_reload_skills(self) -> None:
        handler = ReloadCommandHandler()
        assert handler.command_names == {"reload", "reload-skills"}

    @pytest.mark.parametrize(
        "command_name,expected_target",
        [
            ("reload", ReloadTarget.MODELS),
            ("reload-skills", ReloadTarget.SKILLS),
        ],
    )
    def test_handle_returns_effect_with_correct_target(
        self, *, command_name: str, expected_target: ReloadTarget
    ) -> None:
        handler = ReloadCommandHandler()
        context = SlashCommandContext(
            command_name=command_name,
            remaining_message="",
            original_content=f"/{command_name}",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, ReloadCommandEffect)
        assert effect.target == expected_target

    def test_handle_returns_none_for_unknown_command(self) -> None:
        handler = ReloadCommandHandler()
        context = SlashCommandContext(
            command_name="unknown",
            remaining_message="",
            original_content="/unknown",
        )

        effect = handler.handle(context=context)

        assert effect is None
