"""Tests for prompt-caching block assembly in create_graph_for_llm_async.

Regression coverage for BAI-706 Phase 0: verifies the current, live
per-block `cache_control` mechanism (PromptConfig.cache -> cache_control on
that block) rather than the model-level `.bind(cache_control=...)` approach
that was tried and reverted (commit d028a59) — a future edit should not
silently reintroduce a global/model-level cache toggle in place of this
explicit, per-block opt-in.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage

from languagemodelcommon.configs.schemas.config_schema import PromptConfig
from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)


def _build_converter() -> LangGraphToOpenAIConverter:
    """Bypass __init__'s isinstance checks; create_graph_for_llm_async does
    not read instance state, so an empty shell is sufficient (same pattern
    as _build_converter in test_langgraph_to_openai_converter.py)."""
    return object.__new__(LangGraphToOpenAIConverter)


class TestCreateGraphForLlmAsyncPromptCaching:
    """LangGraphToOpenAIConverter.create_graph_for_llm_async"""

    async def _system_prompt_blocks(
        self, system_prompts: list[PromptConfig]
    ) -> list[dict[str, Any]]:
        converter = _build_converter()
        with patch(
            "languagemodelcommon.converters.langgraph_to_openai_converter.create_agent"
        ) as mock_create_agent:
            mock_create_agent.return_value = MagicMock()
            await converter.create_graph_for_llm_async(
                llm=MagicMock(),
                tools=[],
                store=None,
                checkpointer=None,
                system_prompts=system_prompts,
                tool_catalog=None,
            )
        system_prompt = mock_create_agent.call_args.kwargs["system_prompt"]
        assert isinstance(system_prompt, SystemMessage)
        return cast(list[dict[str, Any]], system_prompt.content)

    @pytest.mark.asyncio
    async def test_cache_true_block_gets_cache_control(self) -> None:
        blocks = await self._system_prompt_blocks(
            [PromptConfig(content="stable instructions", cache=True)]
        )
        assert blocks == [
            {
                "type": "text",
                "text": "stable instructions",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @pytest.mark.asyncio
    async def test_cache_none_or_false_block_has_no_cache_control(self) -> None:
        for cache_value in (None, False):
            blocks = await self._system_prompt_blocks(
                [PromptConfig(content="volatile instructions", cache=cache_value)]
            )
            assert blocks == [{"type": "text", "text": "volatile instructions"}]

    @pytest.mark.asyncio
    async def test_each_block_caches_independently_not_globally(self) -> None:
        """Guards against regressing toward the reverted model-level
        .bind(cache_control=...) approach, which applied one cache marker
        to every eligible block rather than an explicit per-block opt-in."""
        blocks = await self._system_prompt_blocks(
            [
                PromptConfig(content="bailey_system_prompt", cache=True),
                PromptConfig(content="skills", cache=True),
                PromptConfig(content="datetime_context", cache=None),
            ]
        )
        assert blocks == [
            {
                "type": "text",
                "text": "bailey_system_prompt",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": "skills",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "datetime_context"},
        ]

    @pytest.mark.asyncio
    async def test_volatile_block_stays_after_cached_blocks_in_order(self) -> None:
        """The code preserves input order rather than reordering by cache
        flag — the volatile (uncached) block must be listed last in
        PromptConfig input to land after the cache breakpoints, exactly as
        prod's bailey.json orders bailey_system_prompt, skills, then the
        uncached datetime_context_message_format block."""
        blocks = await self._system_prompt_blocks(
            [
                PromptConfig(content="datetime_context", cache=None),
                PromptConfig(content="bailey_system_prompt", cache=True),
            ]
        )
        assert [b["text"] for b in blocks] == [
            "datetime_context",
            "bailey_system_prompt",
        ]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_empty_or_whitespace_content_block_is_dropped(self) -> None:
        blocks = await self._system_prompt_blocks(
            [
                PromptConfig(content="   ", cache=True),
                PromptConfig(content="", cache=True),
                PromptConfig(content="real content", cache=True),
            ]
        )
        assert blocks == [
            {
                "type": "text",
                "text": "real content",
                "cache_control": {"type": "ephemeral"},
            }
        ]
