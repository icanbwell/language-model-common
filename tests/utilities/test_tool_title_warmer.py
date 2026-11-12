import asyncio
import json
from typing import Any

import pytest
from key_value.aio.stores.memory import MemoryStore
from mcp.types import CallToolResult, InputRequiredResult, TextContent

from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.utilities.learned_title_store import (
    LearnedTitleStore,
    TitleReplaceResult,
)
from languagemodelcommon.utilities.tool_title_warmer import (
    ToolTitleWarmer,
    ToolTitleWarmResult,
)

_CALL = "languagemodelcommon.utilities.tool_title_warmer.call_mcp_tool_raw"
_COLLECTION = "learned_tool_titles"
_CONFIG = MCPConnectionConfig(
    url="https://catalog.test/mcp", headers={"Authorization": "Bearer secret-token"}
)


def _result(payload: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
    )


def _store() -> LearnedTitleStore:
    return LearnedTitleStore(
        store=MemoryStore(),
        collection=_COLLECTION,
        ttl_seconds=3600,
        timeout_seconds=1,
    )


def _patch_call(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake(**kwargs: Any) -> Any:
        captured.update(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return await outcome()
        return outcome

    monkeypatch.setattr(_CALL, fake)
    return captured


async def _warm(warmer: ToolTitleWarmer) -> ToolTitleWarmResult:
    return await warmer.warm(config=_CONFIG, server_name="tool-catalog")


class TestWarmSuccess:
    @pytest.mark.asyncio
    async def test_replaces_stored_titles_and_forwards_caller_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store()
        await store.replace_titles(titles={"old_tool": "Old"})
        captured = _patch_call(
            monkeypatch,
            _result(
                {
                    "titles": {"load_skill": "📚 Loading {name}", "search_skills": "S"},
                    "unresolved_servers": [],
                    "future_field": 1,
                }
            ),
        )

        result = await _warm(ToolTitleWarmer(store=store))

        assert (result.ok, result.fetched, result.accepted) == (True, 2, 2)
        assert (result.written, result.deleted, result.partial) == (2, 1, False)
        assert await store.get_title(tool_name="load_skill") == "📚 Loading {name}"
        assert await store.get_title(tool_name="old_tool") is None
        assert captured["config"] is _CONFIG
        assert captured["tool_name"] == "list_tool_titles"
        assert captured["arguments"] == {}

    @pytest.mark.asyncio
    async def test_reads_json_text_when_no_structured_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store()
        _patch_call(
            monkeypatch,
            CallToolResult(
                content=[
                    TextContent(type="text", text=json.dumps({"titles": {"a": "A"}}))
                ]
            ),
        )

        result = await _warm(ToolTitleWarmer(store=store))

        assert (result.ok, result.written) == (True, 1)

    @pytest.mark.asyncio
    async def test_does_not_log_credentials(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _patch_call(monkeypatch, RuntimeError("boom Bearer secret-token"))
        caplog.set_level("DEBUG")

        await _warm(ToolTitleWarmer(store=_store()))

        assert "secret-token" not in caplog.text


class TestPartialAndInvalidResponses:
    @pytest.mark.asyncio
    async def test_unresolved_servers_upsert_without_deleting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store()
        await store.replace_titles(titles={"down_server_tool": "Keep me"})
        _patch_call(
            monkeypatch,
            _result({"titles": {"a": "A"}, "unresolved_servers": ["fhir"]}),
        )

        result = await _warm(ToolTitleWarmer(store=store))

        assert (result.ok, result.partial, result.unresolved_servers) == (
            True,
            True,
            1,
        )
        assert result.deleted == 0
        assert await store.get_title(tool_name="down_server_tool") == "Keep me"
        assert await store.get_title(tool_name="a") == "A"

    @pytest.mark.asyncio
    async def test_invalid_entries_are_dropped_and_prevent_pruning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _store()
        await store.replace_titles(titles={"kept_tool": "Keep me"})
        _patch_call(
            monkeypatch,
            _result(
                {
                    "titles": {
                        "good": "G",
                        "bad_type": 5,
                        "blank": "  ",
                        "long": "x" * 257,
                        "n" * 129: "T",
                    }
                }
            ),
        )

        result = await _warm(ToolTitleWarmer(store=store))

        assert (result.ok, result.fetched, result.accepted) == (True, 5, 1)
        assert result.partial is True
        assert await store.get_title(tool_name="kept_tool") == "Keep me"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("payload", "reason"),
        [
            ({"titles": ["a"]}, "invalid_titles"),
            ({"no_titles": {}}, "invalid_titles"),
            ({"titles": {}}, "no_valid_titles"),
            ({"titles": {"a": 1}}, "no_valid_titles"),
            ({"titles": {"a": "A", "b": "B"}}, "too_many_titles"),
            ({"titles": {"a": "x" * 200}}, "oversized_or_invalid_response"),
            (["not", "a", "dict"], "oversized_or_invalid_response"),
        ],
    )
    async def test_unusable_response_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, payload: Any, reason: str
    ) -> None:
        store = _store()
        await store.replace_titles(titles={"kept_tool": "Keep me"})
        _patch_call(monkeypatch, _result(payload))

        result = await _warm(
            ToolTitleWarmer(store=store, max_titles=1, max_response_bytes=100)
        )

        assert (result.ok, result.reason) == (False, reason)
        assert await store.get_title(tool_name="kept_tool") == "Keep me"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("outcome", "reason"),
        [
            (CallToolResult(content=[], is_error=True), "tool_error"),
            (InputRequiredResult(request_state="s"), "unexpected_result"),
            (
                CallToolResult(content=[TextContent(type="text", text="not json")]),
                "oversized_or_invalid_response",
            ),
        ],
    )
    async def test_error_and_unexpected_results(
        self, monkeypatch: pytest.MonkeyPatch, outcome: Any, reason: str
    ) -> None:
        _patch_call(monkeypatch, outcome)

        result = await _warm(ToolTitleWarmer(store=_store()))

        assert (result.ok, result.reason) == (False, reason)


class TestFailures:
    @pytest.mark.asyncio
    async def test_catalog_call_timeout_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def slow() -> Any:
            await asyncio.sleep(5)

        _patch_call(monkeypatch, slow)

        result = await _warm(ToolTitleWarmer(store=_store(), timeout_seconds=0.05))

        assert (result.ok, result.reason) == (False, "timeout")

    @pytest.mark.asyncio
    async def test_catalog_call_error_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_call(monkeypatch, ConnectionError("down"))

        result = await _warm(ToolTitleWarmer(store=_store()))

        assert (result.ok, result.reason) == (False, "error:ConnectionError")

    @pytest.mark.asyncio
    async def test_mongo_failure_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _DownStore(MemoryStore):
            async def get(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("mongo down")

        store = LearnedTitleStore(
            store=_DownStore(),
            collection=_COLLECTION,
            ttl_seconds=3600,
            timeout_seconds=1,
        )
        _patch_call(monkeypatch, _result({"titles": {"a": "A"}}))

        result = await _warm(ToolTitleWarmer(store=store))

        assert (result.ok, result.reason) == (False, "store_failed")

    @pytest.mark.asyncio
    async def test_store_that_raises_is_still_contained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _RaisingBulk:
            async def replace_titles(self, **kwargs: Any) -> TitleReplaceResult:
                raise RuntimeError("bug")

            async def clear(self) -> TitleReplaceResult:
                return TitleReplaceResult(ok=True)

        _patch_call(monkeypatch, _result({"titles": {"a": "A"}}))

        result = await _warm(ToolTitleWarmer(store=_RaisingBulk()))

        assert (result.ok, result.reason) == (False, "error:RuntimeError")
