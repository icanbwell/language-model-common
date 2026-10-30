"""Validate MCP tool results from the shared tool-list cache (BAI-1388).

``ClientSession.call_tool`` calls the public ``validate_tool_result`` after every
successful result. When the session has not listed tools yet (a fresh one-shot
session never has), the SDK issues a downstream ``tools/list`` just to learn the
tool's output schema. The tool list is already in the shared ``ToolListCache``,
so this module makes a session validate from that cache instead, and defers to
the SDK's own validation whenever the cache cannot answer. See
``adrs/0008-validate-tool-output-from-the-shared-tool-list-cache.md``.

Nothing here caches data in-process: the only cache read is the (Mongo-backed,
URL-keyed) ``ToolListCache`` that the caller already owns.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mcp import ClientSession
from mcp.types import CallToolResult

from languagemodelcommon.mcp.mcp_client.tool_list_cache import (
    ToolListCache,
    list_all_tools,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolOutputSchema:
    """What a tool list says about one tool's output schema.

    ``output_schema is None`` means the tool is listed and declares no output
    schema (nothing to validate), which is different from the tool being
    unknown (represented by the source returning ``None`` instead).
    """

    output_schema: dict[str, Any] | None


class ToolOutputSchemaSource(Protocol):
    """Where a validating session learns a tool's output schema."""

    async def lookup(self, *, tool_name: str) -> ToolOutputSchema | None:
        """Schema from the shared source, or None when it cannot answer
        (no entry, tool not listed, or the source failed)."""
        ...

    async def refresh(
        self, *, session: ClientSession, tool_name: str
    ) -> ToolOutputSchema | None:
        """Re-list tools on ``session``, update the shared source, and return
        the tool's fresh schema, or None when that could not be done."""
        ...


class ToolListCacheSchemaSource:
    """``ToolOutputSchemaSource`` backed by a ``ToolListCache`` entry.

    ``cache_key`` is the key the cache was populated under
    (``ToolListCache.make_key(url, ...)``), so a lookup can only ever see the
    tool list of the one server this call is addressed to.
    """

    def __init__(self, *, cache: ToolListCache, cache_key: str) -> None:
        self._cache = cache
        self._cache_key = cache_key

    async def lookup(self, *, tool_name: str) -> ToolOutputSchema | None:
        try:
            tools = await self._cache.get_async(key=self._cache_key)
        except Exception:
            logger.warning(
                "Tool list cache read failed; validating via the SDK (tool=%s)",
                tool_name,
                exc_info=True,
            )
            return None
        if tools is None:
            return None
        for tool in tools:
            if tool.name == tool_name:
                return ToolOutputSchema(output_schema=tool.output_schema)
        return None

    async def refresh(
        self, *, session: ClientSession, tool_name: str
    ) -> ToolOutputSchema | None:
        fetched_at = time.time()
        try:
            tools = await list_all_tools(session)
        except Exception:
            logger.warning(
                "Refetching the tool list failed; validating via the SDK (tool=%s)",
                tool_name,
                exc_info=True,
            )
            return None
        try:
            await self._cache.put_async(
                key=self._cache_key, tools=tools, fetched_at=fetched_at
            )
        except Exception:
            logger.warning(
                "Tool list cache write failed after a schema refetch (tool=%s)",
                tool_name,
                exc_info=True,
            )
        for tool in tools:
            if tool.name == tool_name:
                return ToolOutputSchema(output_schema=tool.output_schema)
        return None


def check_structured_content(
    *,
    tool_name: str,
    output_schema: dict[str, Any],
    result: CallToolResult,
) -> None:
    """Validate structured content against an output schema.

    Same rules and error messages as the SDK's ``validate_tool_result``:
    ``RuntimeError`` for missing structured content, non-conforming content,
    an invalid schema, or a ``$ref`` that does not resolve within the schema.
    "Missing" follows mcp 2.3: the field absent from the result. An explicit
    ``structuredContent: null`` is validated as a value (so a schema that
    allows null accepts it); mcp 2.2 rejects it as missing.
    """
    # jsonschema and referencing are declared direct dependencies in pyproject.toml
    # (also pulled in by mcp). jsonschema ships no type stubs.
    from jsonschema import SchemaError  # type: ignore[import-untyped]
    from jsonschema import exceptions as jsonschema_exceptions
    from jsonschema.validators import validator_for  # type: ignore[import-untyped]
    from referencing import Registry
    from referencing.exceptions import Unresolvable

    structured_content = result.structured_content
    if (
        structured_content is None
        and "structured_content" not in result.model_fields_set
    ):
        raise RuntimeError(
            f"Tool {tool_name} has an output schema but did not return structured content"
        )
    validator_cls = validator_for(output_schema)
    try:
        validator_cls.check_schema(output_schema)
    except SchemaError as e:
        raise RuntimeError(f"Invalid schema for tool {tool_name}: {e}")
    # Explicit empty registry, as in the SDK: `$ref`s resolve within the schema
    # document and the bundled metaschemas only (no network or filesystem).
    validator = validator_cls(output_schema, registry=Registry())
    try:
        error = jsonschema_exceptions.best_match(
            validator.iter_errors(structured_content)
        )
    except Unresolvable as e:
        raise RuntimeError(f"Invalid schema for tool {tool_name}: {e}") from e
    if error is not None:
        raise RuntimeError(
            f"Invalid structured content returned by tool {tool_name}: {error}"
        ) from error


ToolResultValidator = Callable[[str, CallToolResult], Awaitable[None]]


class CachedOutputSchemaValidator:
    """Drop-in for ``ClientSession.validate_tool_result`` that reads the output
    schema from a ``ToolOutputSchemaSource`` instead of listing tools.

    - source has the tool with no output schema: nothing to validate.
    - source has a schema: validate locally; on failure refetch the tool list
      once and revalidate against the fresh schema before raising.
    - source cannot answer (or a refetch cannot be done): delegate to
      ``fallback`` (the SDK's own validation), so validation is never skipped.
    """

    def __init__(
        self,
        *,
        session: ClientSession,
        source: ToolOutputSchemaSource,
        fallback: ToolResultValidator,
    ) -> None:
        self._session = session
        self._source = source
        self._fallback = fallback

    # Positional parameters: the SDK calls ``self.validate_tool_result(name, result)``.
    async def __call__(self, name: str, result: CallToolResult) -> None:
        """Validate ``result`` for tool ``name``; raise ``RuntimeError`` as the SDK does."""
        schema = await self._source.lookup(tool_name=name)
        if schema is None:
            await self._fallback(name, result)
            return
        if schema.output_schema is None:
            if result.structured_content is None:
                return
            # Cached "no schema" but the server now returns structured content:
            # the entry may be stale, so re-check once against a fresh listing.
            fresh = await self._source.refresh(session=self._session, tool_name=name)
            if fresh is None:
                await self._fallback(name, result)
                return
            if fresh.output_schema is not None:
                check_structured_content(
                    tool_name=name,
                    output_schema=fresh.output_schema,
                    result=result,
                )
            return
        try:
            check_structured_content(
                tool_name=name,
                output_schema=schema.output_schema,
                result=result,
            )
            return
        except RuntimeError as e:
            logger.info(
                "Result failed validation against the cached output schema; "
                "refetching the tool list once (tool=%s, error=%s)",
                name,
                type(e).__name__,
            )
        fresh = await self._source.refresh(session=self._session, tool_name=name)
        if fresh is None:
            await self._fallback(name, result)
            return
        if fresh.output_schema is None:
            return
        check_structured_content(
            tool_name=name,
            output_schema=fresh.output_schema,
            result=result,
        )


def install_cached_output_validation(
    *, session: ClientSession, source: ToolOutputSchemaSource
) -> bool:
    """Make ``session`` validate tool results through ``source``.

    Overrides the public ``validate_tool_result`` on this instance only (the
    SDK ``Client`` wraps a ``ClientSession``, so subclassing is not available).
    Returns False, leaving the session untouched, when the SDK does not expose
    that method. Safe to call repeatedly on a pooled session: the fallback is
    always the class's own method, never a previous override.
    """
    original = getattr(type(session), "validate_tool_result", None)
    if not callable(original):
        logger.warning(
            "ClientSession.validate_tool_result not found; "
            "output validation from the tool list cache is disabled"
        )
        return False
    session.validate_tool_result = CachedOutputSchemaValidator(  # type: ignore[method-assign]
        session=session,
        source=source,
        fallback=original.__get__(session),
    )
    return True
