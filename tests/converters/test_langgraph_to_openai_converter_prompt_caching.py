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
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


def _build_converter() -> LangGraphToOpenAIConverter:
    """Bypass __init__'s isinstance checks; create_graph_for_llm_async only
    reads self.environment_variables beyond that, so a shell with a real
    (default-valued) environment_variables instance is sufficient (same
    pattern as _build_converter in test_langgraph_to_openai_converter.py)."""
    converter = object.__new__(LangGraphToOpenAIConverter)
    converter.environment_variables = LanguageModelCommonEnvironmentVariables()
    return converter


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


class TestCreateGraphForLlmAsyncHistoryCacheMiddlewareGating:
    """BAI-706 ADR Phase 2: HistoryCacheMiddleware must be off by default and
    only wired in when ENABLE_HISTORY_PROMPT_CACHING is explicitly enabled --
    it's gated behind a PHI/EA sign-off (ADR Open Question 4), not a rollout
    convenience."""

    async def _middleware(
        self, *, monkeypatch: pytest.MonkeyPatch, enable_history_prompt_caching: bool
    ) -> list[Any]:
        if enable_history_prompt_caching:
            monkeypatch.setenv("ENABLE_HISTORY_PROMPT_CACHING", "true")
        else:
            monkeypatch.delenv("ENABLE_HISTORY_PROMPT_CACHING", raising=False)
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
                system_prompts=None,
                tool_catalog=None,
            )
        return cast(list[Any], mock_create_agent.call_args.kwargs["middleware"])

    @pytest.mark.asyncio
    async def test_flag_off_by_default_excludes_history_cache_middleware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        middleware = await self._middleware(
            monkeypatch=monkeypatch, enable_history_prompt_caching=False
        )
        assert not any(type(m).__name__ == "HistoryCacheMiddleware" for m in middleware)

    @pytest.mark.asyncio
    async def test_flag_on_includes_history_cache_middleware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        middleware = await self._middleware(
            monkeypatch=monkeypatch, enable_history_prompt_caching=True
        )
        assert any(type(m).__name__ == "HistoryCacheMiddleware" for m in middleware)
