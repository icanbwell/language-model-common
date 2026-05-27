"""Cache for MCP tool task-support declarations per server URL."""

import logging
import time
from dataclasses import dataclass

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class _CachedTaskSupport:
    tool_map: dict[str, str | None]
    expires_at: float


class ToolTaskSupportCache:
    """TTL cache for per-tool task-support declarations, keyed by server URL.

    Avoids redundant list_tools round-trips when checking whether a tool
    supports task-augmented execution. Each entry maps tool names to their
    ``execution.taskSupport`` value ("optional", "required", or None).
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, _CachedTaskSupport] = {}

    def get(self, server_url: str) -> dict[str, str | None] | None:
        entry = self._cache.get(server_url)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._cache[server_url]
            return None
        return entry.tool_map

    def put(self, server_url: str, tool_map: dict[str, str | None]) -> None:
        self._cache[server_url] = _CachedTaskSupport(
            tool_map=tool_map,
            expires_at=time.monotonic() + self._ttl,
        )

    def evict(self, server_url: str) -> None:
        self._cache.pop(server_url, None)

    def clear(self) -> None:
        self._cache.clear()
