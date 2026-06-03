from __future__ import annotations


from languagemodelcommon.utilities.slash_command.handlers.skill_command_handler import (
    SkillCommandHandler,
    SkillCommandEffect,
)
from languagemodelcommon.utilities.slash_command.slash_command_handler import (
    SlashCommandContext,
)


class _FakeResolver:
    def __init__(self, *, skills: dict[str, str]) -> None:
        self._skills = skills

    def get_available_skill_names(self) -> set[str]:
        return set(self._skills.keys())

    async def resolve(self, *, skill_name: str) -> str | None:
        return self._skills.get(skill_name)


class TestSkillCommandHandler:
    def test_matches_registered_skill_name(self) -> None:
        resolver = _FakeResolver(
            skills={"uspstf-depression-screening": "Skill content here"}
        )
        handler = SkillCommandHandler(resolver=resolver)

        assert "uspstf-depression-screening" in handler.command_names

    def test_produces_skill_effect_for_known_skill(self) -> None:
        resolver = _FakeResolver(
            skills={"uspstf-depression-screening": "Skill content here"}
        )
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="uspstf-depression-screening",
            remaining_message="check patient age 45",
            original_content="/uspstf-depression-screening check patient age 45",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, SkillCommandEffect)
        assert effect.skill_name == "uspstf-depression-screening"
        assert effect.remaining_message == "check patient age 45"

    def test_returns_none_for_unknown_skill(self) -> None:
        resolver = _FakeResolver(skills={"depression": "content"})
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="unknown-skill",
            remaining_message="hello",
            original_content="/unknown-skill hello",
        )

        effect = handler.handle(context=context)

        assert effect is None

    def test_empty_remaining_message(self) -> None:
        resolver = _FakeResolver(skills={"list-skills": "list all skills"})
        handler = SkillCommandHandler(resolver=resolver)
        context = SlashCommandContext(
            command_name="list-skills",
            remaining_message="",
            original_content="/list-skills",
        )

        effect = handler.handle(context=context)

        assert isinstance(effect, SkillCommandEffect)
        assert effect.remaining_message == ""
