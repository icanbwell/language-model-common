# MCP Guard-Tool (SEP-2322) Elicitation Support — language-model-common Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let this repo's MCP client stack surface a server's `InputRequiredResult` (SEP-2322 guard-tool ask) to its caller instead of raising `RuntimeError`, and let a caller resubmit the same tool call with the collected answer — without this package knowing anything about how the caller (e.g. baileyai's LangGraph agent) actually asks the human.

**Architecture:** Widen the existing tool-call plumbing (`MCPToolCallRequest`/`MCPToolCallResult`, `_execute_tool_call_with_heartbeat`, `_make_execute_tool`, `call_mcp_tool_raw`) to pass `allow_input_required=True`, `input_responses`, and `request_state` through to `ClientSession.call_tool`. At the LangChain boundary (`mcp_tool_to_langchain_tool`), an `InputRequiredResult` is not silently converted to tool output — it's raised as a new typed exception, `MCPInputRequiredError`, carrying everything needed to retry the exact same call (tool name, arguments, connection, server name, the embedded `input_requests`, and `request_state`). This mirrors `mcp-fhir-agent`'s own `ElicitationRequired` decision (that repo's `adrs/0023-guard-tool-elicitation-pattern.md`): an exception-based short-circuit so intermediate layers (the interceptor chain, `ToolCatalogMcpClient`) don't need any awareness of the new control flow, only the top-level catcher does.

**Tech Stack:** Python 3.13, `mcp` SDK (`ClientSession.call_tool(..., allow_input_required=..., input_responses=..., request_state=...)`, already present in the pinned `mcp` dependency — see `mcp/client/session.py`), `langchain-core` `StructuredTool`, `pytest` + `pytest-asyncio`.

**Spec:** `baileyai/adrs/006-mcp-guard-tool-elicitation-support.md` (design points 1 and the "upstream (language-model-common) change" note in Consequences)

## Global Constraints

- All public functions/methods/constructors are keyword-only (`*` separator); all call sites use keyword arguments. No exceptions in this plan's code.
- `__init__.py` files stay empty — import from the defining module directly (e.g. `from languagemodelcommon.mcp.mcp_client.langchain_adapter import MCPInputRequiredError`), never re-exported.
- No new dependency: `types.InputRequiredResult`/`types.InputResponses`/`types.InputRequest` already ship in the pinned `mcp` package (confirmed via `mcp/client/session.py`'s existing `call_tool` overloads).
- Minimal diff: do not touch the MCP *task*-protocol path (`_execute_tool_as_task`/`_tool_supports_tasks`) — guard-tool and the task protocol are unrelated MCP mechanisms; a tool using one is not assumed to use the other.

---

### Task 1: Widen `MCPToolCallResult` and add retry fields to `MCPToolCallRequest`

**Files:**
- Modify: `languagemodelcommon/mcp/interceptors/types.py`
- Test: `tests/mcp/test_mcp_client.py`

**Interfaces:**
- Produces: `MCPToolCallResult = CallToolResult | InputRequiredResult` (was `CallToolResult` alone). `MCPToolCallRequest.input_responses: InputResponses | None = None`, `MCPToolCallRequest.request_state: str | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_mcp_client.py — add to TestBuildInterceptorChain or a new class
def test_mcp_tool_call_request_carries_input_responses_and_request_state() -> None:
    """MCPToolCallRequest accepts the SEP-2322 retry fields, defaulting to None."""
    from mcp.types import ElicitResult

    bare = MCPToolCallRequest(name="test", args={}, server_name="s1")
    assert bare.input_responses is None
    assert bare.request_state is None

    retry = MCPToolCallRequest(
        name="test",
        args={},
        server_name="s1",
        input_responses={"confirm": ElicitResult(action="accept", content={"confirm": True})},
        request_state="opaque-state-123",
    )
    assert retry.request_state == "opaque-state-123"
    assert retry.input_responses is not None
    assert retry.input_responses["confirm"].action == "accept"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k input_responses_and_request_state -v`
Expected: FAIL with `TypeError: MCPToolCallRequest.__init__() got an unexpected keyword argument 'input_responses'`

- [ ] **Step 3: Write minimal implementation**

```python
# languagemodelcommon/mcp/interceptors/types.py
from mcp.types import CallToolResult, InputRequiredResult, InputResponses

# Result type — matches what interceptors and handlers return. Widened for
# SEP-2322: a guard-tool-gated tool may return InputRequiredResult instead
# of a terminal CallToolResult.
MCPToolCallResult = CallToolResult | InputRequiredResult


class _MCPToolCallRequestOverrides(TypedDict, total=False):
    name: NotRequired[str]
    args: NotRequired[dict[str, Any]]
    headers: NotRequired[dict[str, Any] | None]
    input_responses: NotRequired["InputResponses | None"]
    request_state: NotRequired[str | None]


@dataclass
class MCPToolCallRequest:
    """Tool execution request passed to MCP tool call interceptors.

    Modifiable fields (override to change behavior):
        name: Tool name to invoke.
        args: Tool arguments as key-value pairs.
        headers: HTTP headers for applicable transports.
        input_responses: Answers to a prior call's InputRequiredResult
            (SEP-2322 guard-tool retry). None on a first-round call.
        request_state: Opaque state echoed from a prior InputRequiredResult.
            Must be passed through byte-exact; never inspected here.

    Context fields (read-only, for routing/logging):
        server_name: Name of the MCP server handling the tool.
    """

    name: str
    args: dict[str, Any]
    server_name: str
    headers: dict[str, Any] | None = None
    input_responses: InputResponses | None = None
    request_state: str | None = None

    def override(self, **overrides: Unpack[_MCPToolCallRequestOverrides]) -> Self:
        return replace(self, **overrides)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k input_responses_and_request_state -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/mcp/interceptors/types.py tests/mcp/test_mcp_client.py
git commit -m "BAI-774 widen MCPToolCallResult/Request for SEP-2322 guard-tool retries"
```

---

### Task 2: Thread `allow_input_required`/`input_responses`/`request_state` through `_execute_tool_call_with_heartbeat` and `_make_execute_tool`

**Files:**
- Modify: `languagemodelcommon/mcp/mcp_client/tool_invocation.py`
- Test: `tests/mcp/test_mcp_client.py`

**Interfaces:**
- Consumes: `MCPToolCallRequest.input_responses`/`.request_state` (Task 1).
- Produces: `_execute_tool_call_with_heartbeat(..., input_responses=..., request_state=..., allow_input_required=...) -> CallToolResult | InputRequiredResult`. `_make_execute_tool(...)`'s returned `execute_tool` handler always calls the session with `allow_input_required=True` and forwards the request's retry fields, so any caller (guard-tool-aware or not) gets back whatever the server actually returned instead of a raised `RuntimeError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_mcp_client.py
import pytest
from unittest.mock import AsyncMock
from mcp.types import InputRequiredResult, ElicitRequest, ElicitRequestFormParams
from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    _execute_tool_call_with_heartbeat,
)


class TestExecuteToolCallWithHeartbeat:
    @pytest.mark.asyncio
    async def test_returns_input_required_result_when_allowed(self) -> None:
        """A guard-tool ask is returned to the caller, not raised, when
        allow_input_required=True."""
        input_required = InputRequiredResult(
            result_type="input_required",
            input_requests={
                "confirm": ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        message="Proceed?",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean"}},
                            "required": ["confirm"],
                        },
                    ),
                )
            },
        )
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=input_required)

        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="save_fhir_resource",
            arguments={"resource": {"resourceType": "Patient"}},
            progress_callback=None,
            server_name="mcp-fhir-agent",
            heartbeat_interval_seconds=15.0,
            input_responses=None,
            request_state=None,
            allow_input_required=True,
        )

        assert result is input_required
        session.call_tool.assert_awaited_once_with(
            "save_fhir_resource",
            {"resource": {"resourceType": "Patient"}},
            progress_callback=None,
            input_responses=None,
            request_state=None,
            allow_input_required=True,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k returns_input_required_result_when_allowed -v`
Expected: FAIL with `TypeError: _execute_tool_call_with_heartbeat() got an unexpected keyword argument 'input_responses'`

- [ ] **Step 3: Write minimal implementation**

```python
# languagemodelcommon/mcp/mcp_client/tool_invocation.py
from mcp.types import CallToolResult, InputRequiredResult, InputResponses


async def _execute_tool_call_with_heartbeat(
    *,
    session: Any,
    name: str,
    arguments: dict[str, Any],
    progress_callback: Any,
    server_name: str,
    heartbeat_interval_seconds: float,
    input_responses: InputResponses | None = None,
    request_state: str | None = None,
    allow_input_required: bool = False,
) -> CallToolResult | InputRequiredResult:
    """Call ``session.call_tool``, emitting a synthetic ``mcp_tool_heartbeat``
    custom event every ``heartbeat_interval_seconds`` while the call is in
    flight, regardless of whether the tool reports real progress.

    ``input_responses``/``request_state`` are SEP-2322 guard-tool retry
    fields, forwarded byte-exact to ``session.call_tool``. When
    ``allow_input_required`` is True, an ``InputRequiredResult`` is returned
    to the caller instead of raising -- see ``ClientSession.call_tool``'s own
    docstring for the raise-vs-return contract.

    Uses ``asyncio.shield`` so a heartbeat tick's wait_for timeout never
    cancels the underlying tool call -- only the local wait is abandoned and
    retried on the same call_task. If this coroutine itself is cancelled
    (e.g. the overall chat turn is aborted), the inner call_task is
    cancelled too rather than left running in the background.
    """
    call_task: "asyncio.Task[CallToolResult | InputRequiredResult]" = asyncio.ensure_future(
        session.call_tool(
            name,
            arguments,
            progress_callback=progress_callback,
            input_responses=input_responses,
            request_state=request_state,
            allow_input_required=allow_input_required,
        )
    )
    elapsed_seconds = 0.0
    try:
        while True:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(call_task), timeout=heartbeat_interval_seconds
                )
            except asyncio.TimeoutError:
                elapsed_seconds += heartbeat_interval_seconds
                try:
                    await adispatch_custom_event(
                        "mcp_tool_heartbeat",
                        {
                            "server_name": server_name,
                            "tool_name": name,
                            "elapsed_seconds": elapsed_seconds,
                        },
                    )
                except RuntimeError as e:
                    logger.debug(
                        "Skipping mcp_tool_heartbeat event dispatch: %s (tool=%s)",
                        e,
                        name,
                    )
    finally:
        if not call_task.done():
            call_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await call_task
```

Then update all three call sites of `_execute_tool_call_with_heartbeat` inside `_make_execute_tool`'s `execute_tool()` — the `session_pool is not None` branch, and, within the one-shot-session fallback branch, both the post-`TaskProtocolError` retry and the plain `else` branch (used when the tool doesn't support tasks) — to forward `input_responses=request.input_responses, request_state=request.request_state, allow_input_required=True` — always `True`: a caller that never sends `input_responses` on a non-gated tool gets a plain `CallToolResult` back exactly as before, since only a guard-tool-gated tool ever returns `InputRequiredResult` in the first place. Also widen `execute_tool`'s return type annotation from `MCPToolCallResult` (already widened in Task 1) — no further signature change needed there since `MCPToolCallRequest`/`MCPToolCallResult` were already updated.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k returns_input_required_result_when_allowed -v`
Expected: PASS

- [ ] **Step 5: Run the full existing suite for this module to check for regressions**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/ -v`
Expected: All PASS, including the pre-existing `TestBuildInterceptorChain`/`TestConvertMcpContentToLcBlock`/`TestConvertCallToolResult` tests (unaffected — they never touch heartbeat/session code).

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/mcp/mcp_client/tool_invocation.py tests/mcp/test_mcp_client.py
git commit -m "BAI-774 forward SEP-2322 guard-tool retry fields through execute_tool"
```

---

### Task 3: Thread the same fields through the public `call_mcp_tool_raw` entrypoint

**Files:**
- Modify: `languagemodelcommon/mcp/mcp_client/tool_invocation.py`
- Test: `tests/mcp/test_mcp_client.py`

**Interfaces:**
- Consumes: `_make_execute_tool` (Task 2, already forwards retry fields from the `MCPToolCallRequest` it's given).
- Produces: `call_mcp_tool_raw(..., input_responses: InputResponses | None = None, request_state: str | None = None) -> CallToolResult | InputRequiredResult`. This is the function baileyai's LangGraph-side bridge (see the companion `baileyai` plan) calls directly to resubmit a guard-tool answer — it does not go through `mcp_tool_to_langchain_tool`'s LangChain-specific coroutine for the retry leg, since that leg isn't driven by a fresh LLM tool call.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_mcp_client.py
from languagemodelcommon.mcp.mcp_client.tool_invocation import call_mcp_tool_raw
from languagemodelcommon.mcp.mcp_client.session import MCPConnectionConfig


class TestCallMcpToolRaw:
    @pytest.mark.asyncio
    async def test_forwards_input_responses_and_request_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """call_mcp_tool_raw forwards retry fields to the underlying session call."""
        from mcp.types import ElicitResult

        captured: dict[str, Any] = {}

        async def fake_execute_tool_call_with_heartbeat(**kwargs: Any) -> CallToolResult:
            captured.update(kwargs)
            return CallToolResult(content=[TextContent(type="text", text="saved")])

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation._execute_tool_call_with_heartbeat",
            fake_execute_tool_call_with_heartbeat,
        )
        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.tool_invocation.create_mcp_session",
            lambda *args, **kwargs: _fake_session_cm(),
        )

        result = await call_mcp_tool_raw(
            config=MCPConnectionConfig(url="https://example.test/mcp"),
            tool_name="save_fhir_resource",
            arguments={"resource": {"resourceType": "Patient"}},
            server_name="mcp-fhir-agent",
            input_responses={
                "confirm": ElicitResult(action="accept", content={"confirm": True})
            },
            request_state="opaque-state-123",
        )

        assert isinstance(result, CallToolResult)
        assert captured["request_state"] == "opaque-state-123"
        assert captured["input_responses"]["confirm"].action == "accept"
        assert captured["allow_input_required"] is True
```

`_fake_session_cm` is a small local async-context-manager helper yielding an `AsyncMock()` with `.initialize = AsyncMock()` — follow the pattern already used elsewhere in this test file for mocking `create_mcp_session` (grep the file for existing `create_mcp_session` mocks before writing a new helper; reuse one if it already exists).

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k forwards_input_responses_and_request_state -v`
Expected: FAIL with `TypeError: call_mcp_tool_raw() got an unexpected keyword argument 'input_responses'`

- [ ] **Step 3: Write minimal implementation**

```python
# languagemodelcommon/mcp/mcp_client/tool_invocation.py
async def call_mcp_tool_raw(
    *,
    config: MCPConnectionConfig,
    tool_name: str,
    arguments: dict[str, Any],
    server_name: str,
    callbacks: Callbacks | None = None,
    tool_interceptors: list[ToolCallInterceptor] | None = None,
    session_pool: McpSessionPool | None = None,
    tool_list_cache: ToolListCache | None = None,
    heartbeat_interval_seconds: float = 15.0,
    input_responses: InputResponses | None = None,
    request_state: str | None = None,
) -> CallToolResult | InputRequiredResult:
    """Call an MCP tool and return the raw CallToolResult (or, for a
    guard-tool-gated tool, an InputRequiredResult).

    This is used by the call_tool meta-tool to proxy calls without
    converting to LangChain format, and by callers resubmitting a SEP-2322
    guard-tool answer (``input_responses``/``request_state``) directly,
    bypassing the LangChain tool-call path entirely since the retry leg
    isn't driven by a fresh LLM tool call.
    """
    mcp_callbacks = (
        callbacks.to_mcp_format(
            context=CallbackContext(server_name=server_name, tool_name=tool_name)
        )
        if callbacks is not None
        else _MCPCallbacks()
    )

    execute_tool = _make_execute_tool(
        config=config,
        mcp_callbacks=mcp_callbacks,
        session_pool=session_pool,
        tool_list_cache=tool_list_cache,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    handler = build_interceptor_chain(
        base_handler=execute_tool, tool_interceptors=tool_interceptors
    )
    request = MCPToolCallRequest(
        name=tool_name,
        args=arguments,
        server_name=server_name,
        headers=None,
        input_responses=input_responses,
        request_state=request_state,
    )
    return await handler(request)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k forwards_input_responses_and_request_state -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/mcp/mcp_client/tool_invocation.py tests/mcp/test_mcp_client.py
git commit -m "BAI-774 forward SEP-2322 retry fields through call_mcp_tool_raw"
```

---

### Task 4: Raise `MCPInputRequiredError` from the LangChain tool boundary

**Files:**
- Modify: `languagemodelcommon/mcp/mcp_client/langchain_adapter.py`
- Test: `tests/mcp/test_mcp_client.py` (or a new `tests/mcp/test_langchain_adapter.py` if that file doesn't already exist — check first)

**Interfaces:**
- Consumes: `MCPToolCallResult` (Task 1, now `CallToolResult | InputRequiredResult`); `MCPConnectionConfig` (existing).
- Produces: `MCPInputRequiredError` — a `RuntimeError` subclass with fields `tool_name: str`, `arguments: dict[str, Any]`, `connection: MCPConnectionConfig`, `server_name: str | None`, `input_requests: dict[str, InputRequest]`, `request_state: str | None`. This is the type baileyai's own LangGraph bridge (companion plan) catches to build an `interrupt()` payload and, on resume, calls `call_mcp_tool_raw(..., input_responses=..., request_state=e.request_state, allow_input_required=True)` to complete the round trip — it does not need to inspect `mcp_tool_to_langchain_tool` internals to do so, only this exception's public fields.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_mcp_client.py
from languagemodelcommon.mcp.mcp_client.langchain_adapter import (
    MCPInputRequiredError,
    mcp_tool_to_langchain_tool,
)
from mcp.types import Tool as MCPTool


class TestMcpToolToLangchainToolInputRequired:
    @pytest.mark.asyncio
    async def test_raises_mcp_input_required_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An InputRequiredResult from the handler chain surfaces as
        MCPInputRequiredError, not silently stringified tool output."""
        input_required = InputRequiredResult(
            result_type="input_required",
            input_requests={
                "confirm": ElicitRequest(
                    method="elicitation/create",
                    params=ElicitRequestFormParams(
                        message="About to save a Patient. Proceed?",
                        requested_schema={
                            "type": "object",
                            "properties": {"confirm": {"type": "boolean"}},
                            "required": ["confirm"],
                        },
                    ),
                )
            },
            request_state="opaque-state-123",
        )

        async def fake_execute_tool(request: MCPToolCallRequest) -> InputRequiredResult:
            return input_required

        monkeypatch.setattr(
            "languagemodelcommon.mcp.mcp_client.langchain_adapter._make_execute_tool",
            lambda **kwargs: fake_execute_tool,
        )

        mcp_tool = MCPTool(
            name="save_fhir_resource",
            description="Save a FHIR resource",
            inputSchema={"type": "object", "properties": {}},
        )
        connection = MCPConnectionConfig(url="https://mcp-fhir-agent.test/mcp")
        tool = mcp_tool_to_langchain_tool(
            tool=mcp_tool, connection=connection, server_name="mcp-fhir-agent"
        )

        with pytest.raises(MCPInputRequiredError) as exc_info:
            await tool.coroutine(resource={"resourceType": "Patient"})

        err = exc_info.value
        assert err.tool_name == "save_fhir_resource"
        assert err.arguments == {"resource": {"resourceType": "Patient"}}
        assert err.server_name == "mcp-fhir-agent"
        assert err.request_state == "opaque-state-123"
        assert "confirm" in err.input_requests
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k raises_mcp_input_required_error -v`
Expected: FAIL with `ImportError: cannot import name 'MCPInputRequiredError'`

- [ ] **Step 3: Write minimal implementation**

```python
# languagemodelcommon/mcp/mcp_client/langchain_adapter.py
from typing import Any

from mcp.types import InputRequest, InputRequiredResult
from mcp.types import Tool as MCPTool


class MCPInputRequiredError(RuntimeError):
    """Raised when an MCP tool call returns InputRequiredResult (SEP-2322
    guard-tool ask) instead of a terminal CallToolResult.

    Carries everything a caller needs to resubmit the *exact same* call via
    ``call_mcp_tool_raw`` once it has collected an answer to
    ``input_requests`` -- this package has no opinion on how that answer is
    collected (a human-in-the-loop UI, a LangGraph ``interrupt()``, or
    anything else); it only guarantees the retry leg has what it needs.

    Deliberately a plain ``RuntimeError`` subclass, not ``BaseException``:
    unlike mcp-fhir-agent's server-side ``ElicitationRequired`` (which must
    survive broad `except Exception` handlers several call layers below the
    tool function), this exception is raised directly at the LangChain tool
    boundary with no intermediate layer in this package that could swallow
    it. A caller building a broad `except Exception` around a tool
    invocation should already expect to see this type explicitly if it
    wants guard-tool support -- see the companion baileyai ADR
    (`adrs/006-mcp-guard-tool-elicitation-support.md`) for how it's
    consumed.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        connection: "MCPConnectionConfig",
        server_name: str | None,
        input_requests: dict[str, InputRequest],
        request_state: str | None,
    ) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.connection = connection
        self.server_name = server_name
        self.input_requests = input_requests
        self.request_state = request_state
        super().__init__(
            f"MCP tool {tool_name!r} requires input before it can proceed "
            f"(fields: {sorted(input_requests)})"
        )
```

Then, in `mcp_tool_to_langchain_tool`'s `call_tool` closure:

```python
    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[list[ToolMessageContentBlock], None]:
        request = MCPToolCallRequest(
            name=tool.name,
            args=arguments,
            server_name=server_name or "unknown",
            headers=None,
        )
        call_tool_result = await handler(request)
        if isinstance(call_tool_result, InputRequiredResult):
            raise MCPInputRequiredError(
                tool_name=tool.name,
                arguments=arguments,
                connection=connection,
                server_name=server_name,
                input_requests=call_tool_result.input_requests or {},
                request_state=call_tool_result.request_state,
            )
        content = convert_call_tool_result(call_tool_result)
        return content, None
```

`_make_execute_tool` is called without `allow_input_required` at this call site — that's fine, because Task 2 made `_make_execute_tool`'s inner `execute_tool()` always pass `allow_input_required=True` to the session, regardless of caller; the raw `MCPToolCallRequest` built above has `input_responses=None`/`request_state=None` (this is always a fresh LLM-driven call, never a retry), which is exactly what an unanswered first guard-tool round expects.

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/test_mcp_client.py -k raises_mcp_input_required_error -v`
Expected: PASS

- [ ] **Step 5: Run the full mcp test suite**

Run: `docker compose run --rm --name languagemodelcommon dev pytest tests/mcp/ -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/mcp/mcp_client/langchain_adapter.py tests/mcp/test_mcp_client.py
git commit -m "BAI-774 raise MCPInputRequiredError on SEP-2322 guard-tool asks"
```

---

### Task 5: Open the PR

- [ ] **Step 1:** Push the branch (`IQ-BAI-774` or a fresh `IQ-BAI-774-lmc` if `IQ-BAI-774` is already in use by the baileyai-side branch of the same ticket — branch names aren't shared across repos, but check for confusion before naming).
- [ ] **Step 2:** Open a PR against `main` titled `BAI-774 support SEP-2322 guard-tool elicitation in MCP client`, body summarizing: what changed (Tasks 1-4), that this is the upstream half of `baileyai/adrs/006-mcp-guard-tool-elicitation-support.md`, and that baileyai's PR consuming this depends on it merging + releasing first.
- [ ] **Step 3:** Do not merge without the org's stacked-PR rule check (`gh pr list --json number,baseRefName`) — this PR itself is standalone (base `main`), but flag in the PR description that baileyai's companion PR depends on this one merging and a new release being cut, so merge order matters even though there's no GitHub-level stack.

## Self-Review Notes

- **Spec coverage:** ADR 006's "upstream (language-model-common) change" bullet (design point 1) is fully covered: `allow_input_required=True` is now always passed (Task 2), `InputRequiredResult` is surfaced instead of raising (Tasks 2-3), and the LangChain boundary gets a typed signal (Task 4) — mirroring the exception-based short-circuit `mcp-fhir-agent`'s ADR 0023 already chose server-side, which was the ADR's explicit design goal.
- **Type consistency:** `MCPToolCallResult` (Task 1) is used consistently as the return type of `_execute_tool_call_with_heartbeat` (Task 2) and `call_mcp_tool_raw` (Task 3); `MCPInputRequiredError` (Task 4) exposes exactly the fields (`tool_name`, `arguments`, `connection`, `server_name`, `input_requests`, `request_state`) that the companion baileyai plan's bridge consumes for its retry call to `call_mcp_tool_raw`.
- **No placeholders:** every step has runnable code and an exact test command.
