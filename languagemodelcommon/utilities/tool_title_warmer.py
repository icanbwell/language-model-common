"""Pre-warm the persistent tool-title store from the tool catalog (BAI-1732).

The tool-catalog MCP server exposes a read-only ``list_tool_titles`` tool that
returns every ``call_tool`` target's title template. ``ToolTitleWarmer`` reads it
through the repo's MCP session machinery (``call_mcp_tool_raw``) with the
caller's connection config (so the caller's auth headers are forwarded exactly as
for the discovery meta-tools) and replaces the stored set. It never raises, is
bounded in time and size, and logs counts only.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from mcp.types import CallToolResult, TextContent

from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig
from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw
from languagemodelcommon.utilities.learned_title_store import (
    LearnedTitleBulkStoreProtocol,
)
from languagemodelcommon.utilities.tool_display_name_mapper import is_storable_title

logger = logging.getLogger(__name__)

MCP_LIST_TOOL_TITLES_NAME = "list_tool_titles"


@dataclass(frozen=True)
class ToolTitleWarmResult:
    """Outcome of one warm. ``ok`` is False when nothing was written."""

    ok: bool
    reason: str | None = None
    fetched: int = 0
    accepted: int = 0
    written: int = 0
    deleted: int = 0
    unresolved_servers: int = 0
    # True when the catalog reported unresolved servers: titles were upserted
    # but nothing was deleted, so a server that is down keeps its titles.
    partial: bool = False


class ToolTitleWarmerProtocol(Protocol):
    async def warm(
        self, *, config: MCPConnectionConfig, server_name: str
    ) -> ToolTitleWarmResult: ...


class ToolTitleWarmer:
    """Fetch ``list_tool_titles`` from the catalog and replace the stored titles."""

    def __init__(
        self,
        *,
        store: LearnedTitleBulkStoreProtocol,
        timeout_seconds: float = 10.0,
        max_titles: int = 5000,
        max_response_bytes: int = 1_048_576,
        tool_name: str = MCP_LIST_TOOL_TITLES_NAME,
    ) -> None:
        self._store = store
        self._timeout_seconds = timeout_seconds
        self._max_titles = max_titles
        self._max_response_bytes = max_response_bytes
        self._tool_name = tool_name

    async def warm(
        self, *, config: MCPConnectionConfig, server_name: str
    ) -> ToolTitleWarmResult:
        try:
            result = await self._warm(config=config, server_name=server_name)
        except asyncio.TimeoutError:
            result = ToolTitleWarmResult(ok=False, reason="timeout")
        except Exception as exc:
            # Type only: exception text can carry URLs or header values.
            result = ToolTitleWarmResult(ok=False, reason=f"error:{type(exc).__name__}")
        logger.info(
            "Tool title warm finished ok=%s reason=%s fetched=%d accepted=%d "
            "written=%d deleted=%d unresolved_servers=%d partial=%s",
            result.ok,
            result.reason,
            result.fetched,
            result.accepted,
            result.written,
            result.deleted,
            result.unresolved_servers,
            result.partial,
        )
        return result

    async def _warm(
        self, *, config: MCPConnectionConfig, server_name: str
    ) -> ToolTitleWarmResult:
        raw = await asyncio.wait_for(
            call_mcp_tool_raw(
                config=config,
                tool_name=self._tool_name,
                arguments={},
                server_name=server_name,
            ),
            timeout=self._timeout_seconds,
        )
        if not isinstance(raw, CallToolResult):
            return ToolTitleWarmResult(ok=False, reason="unexpected_result")
        if raw.is_error:
            return ToolTitleWarmResult(ok=False, reason="tool_error")
        payload = self._payload(result=raw)
        if payload is None:
            return ToolTitleWarmResult(ok=False, reason="oversized_or_invalid_response")
        titles_raw = payload.get("titles")
        if not isinstance(titles_raw, dict):
            return ToolTitleWarmResult(ok=False, reason="invalid_titles")
        if len(titles_raw) > self._max_titles:
            return ToolTitleWarmResult(
                ok=False, reason="too_many_titles", fetched=len(titles_raw)
            )
        titles = self._valid_titles(titles_raw=titles_raw)
        unresolved = payload.get("unresolved_servers")
        unresolved_count = len(unresolved) if isinstance(unresolved, list) else 0
        if not titles:
            return ToolTitleWarmResult(
                ok=False,
                reason="no_valid_titles",
                fetched=len(titles_raw),
                unresolved_servers=unresolved_count,
            )
        partial = unresolved_count > 0 or len(titles) < len(titles_raw)
        stored = await self._store.replace_titles(titles=titles, prune=not partial)
        return ToolTitleWarmResult(
            ok=stored.ok,
            reason=None if stored.ok else "store_failed",
            fetched=len(titles_raw),
            accepted=len(titles),
            written=stored.written,
            deleted=stored.deleted,
            unresolved_servers=unresolved_count,
            partial=partial,
        )

    def _payload(self, *, result: CallToolResult) -> dict[str, Any] | None:
        structured: Any = result.structured_content
        if structured is None:
            for block in result.content:
                if isinstance(block, TextContent):
                    if len(block.text) > self._max_response_bytes:
                        return None
                    try:
                        structured = json.loads(block.text)
                    except ValueError:
                        return None
                    break
        if not isinstance(structured, dict):
            return None
        try:
            size = len(json.dumps(structured))
        except (TypeError, ValueError):
            return None
        return structured if size <= self._max_response_bytes else None

    @staticmethod
    def _valid_titles(*, titles_raw: dict[Any, Any]) -> dict[str, str]:
        valid: dict[str, str] = {}
        for name, title in titles_raw.items():
            if not isinstance(name, str) or not isinstance(title, str):
                continue
            stripped = title.strip()
            if stripped and is_storable_title(tool_name=name, title=stripped):
                valid[name] = stripped
        return valid
