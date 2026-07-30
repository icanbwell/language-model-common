# MCP Tool Progress Forwarding & Heartbeat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** During an in-flight MCP tool call, forward real MCP progress notifications to the SSE stream, and emit a synthetic heartbeat when a tool call runs long without reporting progress — so downstream clients (e.g. baileyai-skills-service's chat UI) never see a multi-tens-of-seconds silent gap that trips their own idle/read timeouts.

**Architecture:** `language-model-common` already has a fully-wired `mcp_task_progress` custom-event → `task.progress` SSE pipeline (`streaming_manager.py` → `ChatRequestWrapper.create_task_progress_sse_event`), but today only the experimental MCP task-polling path feeds it — most tools never trigger it. This plan (1) wires the *standard* MCP `progress_callback` (which already fires for any tool that calls `ctx.report_progress()`) into that same pipeline, and (2) adds a generic elapsed-time watchdog around `session.call_tool(...)` that emits a synthetic heartbeat on a fixed interval regardless of whether the tool reports progress at all — using a *separate* custom-event name and a *separate* opt-in flag for the Chat-Completions content-delta path, so existing `EMIT_TASK_PROGRESS_IN_CHAT_COMPLETIONS` consumers don't suddenly get noisier without a second explicit opt-in.

**Tech Stack:** Python 3.12, `asyncio`, `langchain-core` (`adispatch_custom_event`), MCP Python SDK (`mcp.ClientSession.call_tool(progress_callback=...)`), pytest (`asyncio_mode = "auto"`).

## Global Constraints

- No library upgrade is needed or in scope. Confirmed by inspecting the pinned `mcp` SDK (`ClientSession.call_tool`, `mcp/client/session.py:386-394`): the `progress_callback` parameter already exists and is unchanged across the pinned (1.27.2) and latest (1.28.1) versions. This is a pure application-level wiring gap.
- Existing behavior for every current deployment must be unchanged by default. `EMIT_TASK_PROGRESS_IN_CHAT_COMPLETIONS` stays exactly as-is (default `false`, unchanged semantics). The new heartbeat path gets its own env flag, default `false`.
- `baileyai` depends on `language-model-common>=2.0.85` as a published PyPI package (`baileyai/pyproject.toml:39`), not a path dependency — a fix here requires a version bump + release before `baileyai` can pick it up.
- Follow this repo's keyword-only-argument convention (`*` first parameter) for all new functions/methods, matching the surrounding code.

---

## File Structure

| File | Responsibility |
|---|---|
| `languagemodelcommon/utilities/environment/language_model_common_environment_variables.py` | New env vars: heartbeat interval, Chat-Completions heartbeat opt-in |
| `languagemodelcommon/structures/openai/request/chat_request_wrapper.py` | New no-op base method `create_tool_heartbeat_sse_event` |
| `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py` | Override: emit `task.progress` SSE (reuses existing wire format baileyai-skills-service already parses) |
| `languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py` | Override: emit a content delta, gated behind the new flag |
| `languagemodelcommon/converters/streaming_manager.py` | Route a new `mcp_tool_heartbeat` custom event to the new SSE method |
| `languagemodelcommon/mcp/mcp_tool_provider.py` | `on_mcp_tool_progress` forwards real progress notifications via `adispatch_custom_event("mcp_task_progress", ...)` |
| `languagemodelcommon/mcp/mcp_client/tool_invocation.py` | New `_execute_tool_call_with_heartbeat` watchdog wrapping `session.call_tool(...)` |
| `VERSION` | Bumped for release |
| `baileyai/pyproject.toml`, `baileyai/uv.lock` | Pin bump to the new release |

---

### Task 1: Add heartbeat environment variables

**Files:**
- Modify: `languagemodelcommon/utilities/environment/language_model_common_environment_variables.py:287-290` (after `tool_call_timeout_seconds`)
- Test: `tests/utilities/test_language_model_common_environment_variables_heartbeat.py`

**Interfaces:**
- Produces: `LanguageModelCommonEnvironmentVariables.mcp_tool_heartbeat_interval_seconds -> float`, `LanguageModelCommonEnvironmentVariables.emit_tool_heartbeat_in_chat_completions -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/utilities/test_language_model_common_environment_variables_heartbeat.py
import pytest

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)


def test_mcp_tool_heartbeat_interval_seconds_defaults_to_15(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.mcp_tool_heartbeat_interval_seconds == 15.0


def test_mcp_tool_heartbeat_interval_seconds_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", "5")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.mcp_tool_heartbeat_interval_seconds == 5.0


def test_emit_tool_heartbeat_in_chat_completions_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", raising=False)
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.emit_tool_heartbeat_in_chat_completions is False


def test_emit_tool_heartbeat_in_chat_completions_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", "true")
    env_vars = LanguageModelCommonEnvironmentVariables()
    assert env_vars.emit_tool_heartbeat_in_chat_completions is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/utilities/test_language_model_common_environment_variables_heartbeat.py -v`
Expected: FAIL with `AttributeError: 'LanguageModelCommonEnvironmentVariables' object has no attribute 'mcp_tool_heartbeat_interval_seconds'`

- [ ] **Step 3: Write minimal implementation**

Insert after `tool_call_timeout_seconds` (`language_model_common_environment_variables.py:287-290`):

```python
    @property
    def mcp_tool_heartbeat_interval_seconds(self) -> float:
        """Interval in seconds between synthetic heartbeat events emitted
        while an MCP tool call is in flight without reporting progress."""
        return float(os.environ.get("MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS", "15"))

    @property
    def emit_tool_heartbeat_in_chat_completions(self) -> bool:
        """When True, synthetic MCP tool heartbeats are emitted as content
        deltas in the Chat Completions streaming format. Separate from
        emit_task_progress_in_chat_completions so enabling one does not
        change the volume/behavior of the other."""
        return self.str2bool(
            os.environ.get("EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS", "false")
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/utilities/test_language_model_common_environment_variables_heartbeat.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/utilities/environment/language_model_common_environment_variables.py tests/utilities/test_language_model_common_environment_variables_heartbeat.py
git commit -m "feat: add MCP tool heartbeat environment variables"
```

---

### Task 2: Add `create_tool_heartbeat_sse_event` to the wrapper hierarchy

**Files:**
- Modify: `languagemodelcommon/structures/openai/request/chat_request_wrapper.py:260-273` (add new method after `create_task_progress_sse_event`)
- Modify: `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py:358-374` (add override after `create_task_progress_sse_event`)
- Modify: `languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py:262-279` (add override after `create_task_progress_sse_event`)
- Test: `tests/structures/openai/request/test_tool_heartbeat_sse_event.py`

**Interfaces:**
- Consumes: `LanguageModelCommonEnvironmentVariables.emit_tool_heartbeat_in_chat_completions` (Task 1)
- Produces: `ChatRequestWrapper.create_tool_heartbeat_sse_event(*, request_id: str, tool_name: str, elapsed_seconds: float) -> str | None`, overridden identically in both subclasses' constructors.

- [ ] **Step 1: Write the failing test**

First, locate the existing test directory to confirm import paths:

```bash
find tests -path "*structures/openai/request*" -name "*.py" | grep -v pycache
```

Then write:

```python
# tests/structures/openai/request/test_tool_heartbeat_sse_event.py
import json

from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)


def test_base_wrapper_tool_heartbeat_is_noop() -> None:
    class _Bare(ChatRequestWrapper):
        pass

    # ChatRequestWrapper is abstract in practice; call the method directly
    # via the base implementation to confirm the no-op default.
    result = ChatRequestWrapper.create_tool_heartbeat_sse_event(
        object.__new__(ChatRequestWrapper),
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is None


def test_responses_api_wrapper_emits_task_progress_event(
    responses_api_request_wrapper_factory,
) -> None:
    wrapper = responses_api_request_wrapper_factory()
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is not None
    assert result.startswith("data: ")
    payload = json.loads(result[len("data: ") :].strip())
    assert payload["type"] == "task.progress"
    assert payload["task_id"] == ""
    assert "propose_skill" in payload["message"]
    assert "15" in payload["message"]


def test_chat_completion_wrapper_tool_heartbeat_disabled_by_default(
    chat_completion_api_request_wrapper_factory,
) -> None:
    wrapper = chat_completion_api_request_wrapper_factory(
        emit_tool_heartbeat_in_chat_completions=False,
    )
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is None


def test_chat_completion_wrapper_tool_heartbeat_enabled_emits_delta(
    chat_completion_api_request_wrapper_factory,
) -> None:
    wrapper = chat_completion_api_request_wrapper_factory(
        emit_tool_heartbeat_in_chat_completions=True,
    )
    result = wrapper.create_tool_heartbeat_sse_event(
        request_id="req-1",
        tool_name="propose_skill",
        elapsed_seconds=15.0,
    )
    assert result is not None
    assert "propose_skill" in result
```

Check whether `responses_api_request_wrapper_factory` / `chat_completion_api_request_wrapper_factory` fixtures already exist in `tests/structures/openai/request/conftest.py`:

```bash
grep -rn "def responses_api_request_wrapper_factory\|def chat_completion_api_request_wrapper_factory" tests/structures/openai/request/
```

If they don't exist, construct the wrappers directly in the test instead of via fixtures — inspect the constructors' required args first:

```bash
grep -n "def __init__" languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py
```

and adapt the two factory-based tests to construct the wrapper directly with whatever minimal required arguments those constructors need (mirroring how nearby existing tests in the same test directory construct these wrappers — match that pattern exactly rather than inventing a new one).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/structures/openai/request/test_tool_heartbeat_sse_event.py -v`
Expected: FAIL with `AttributeError: 'ChatRequestWrapper' object has no attribute 'create_tool_heartbeat_sse_event'`

- [ ] **Step 3: Write minimal implementation**

In `chat_request_wrapper.py`, after `create_task_progress_sse_event` (line 273):

```python
    def create_tool_heartbeat_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        elapsed_seconds: float,
    ) -> str | None:
        """Emit a synthetic heartbeat while a tool call is in flight without
        reporting real progress.

        The default implementation returns None (no-op).  Subclasses override
        to emit the event in their respective SSE formats.
        """
        return None
```

In `responses_api_request_wrapper.py`, after `create_task_progress_sse_event` (line 374):

```python
    @override
    def create_tool_heartbeat_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        elapsed_seconds: float,
    ) -> str | None:
        """Emit a synthetic heartbeat as a ``task.progress`` SSE event.

        Reuses the exact same wire format as create_task_progress_sse_event
        so existing consumers (e.g. baileyai-skills-service's frontend, which
        already treats any `type: "task.progress"` payload as a non-content
        trace event) require no changes.
        """
        event: Dict[str, Any] = {
            "type": "task.progress",
            "task_id": "",
            "status": "in_progress",
            "message": f"Still running {tool_name}... ({elapsed_seconds:.0f}s)",
        }
        return f"data: {json.dumps(event)}\n\n"
```

In `chat_completion_api_request_wrapper.py`, after `create_task_progress_sse_event` (line 279):

```python
    @override
    def create_tool_heartbeat_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        elapsed_seconds: float,
    ) -> str | None:
        if not self._emit_tool_heartbeat:
            return None
        return self.create_sse_message(
            request_id=request_id,
            content=f"\n[Still running {tool_name}... ({elapsed_seconds:.0f}s)]\n",
            usage_metadata=None,
            source="tool_heartbeat",
        )
```

`chat_completion_api_request_wrapper.py` needs a new `self._emit_tool_heartbeat` attribute set in `__init__`, mirroring `self._emit_task_progress` (`chat_completion_api_request_wrapper.py:82-85`). Read the constructor first:

```bash
sed -n '60,90p' languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py
```

Add a new constructor parameter and assignment analogous to the existing `emit_task_progress` one:

```python
        emit_tool_heartbeat: bool | None = None,
```

```python
        self._emit_tool_heartbeat: bool = (
            emit_tool_heartbeat
            if emit_tool_heartbeat is not None
            else environment_variables.emit_tool_heartbeat_in_chat_completions
        )
```

Find every call site that constructs `ChatCompletionApiRequestWrapper` (`grep -rn "ChatCompletionApiRequestWrapper(" languagemodelcommon/`) and confirm the new parameter is optional with a safe default (it is, via `environment_variables.emit_tool_heartbeat_in_chat_completions`) so no call site needs updating.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/structures/openai/request/test_tool_heartbeat_sse_event.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/structures/openai/request/chat_request_wrapper.py languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py languagemodelcommon/structures/openai/request/chat_completion_api_request_wrapper.py tests/structures/openai/request/test_tool_heartbeat_sse_event.py
git commit -m "feat: add create_tool_heartbeat_sse_event to request wrapper hierarchy"
```

---

### Task 3: Route `mcp_tool_heartbeat` custom events in the streaming manager

**Files:**
- Modify: `languagemodelcommon/converters/streaming_manager.py:364-383` (`_handle_on_custom_event`)
- Test: `tests/converters/test_streaming_manager.py` (add to existing file)

**Interfaces:**
- Consumes: `ChatRequestWrapper.create_tool_heartbeat_sse_event` (Task 2)
- Produces: `LangGraphStreamingManager._handle_on_custom_event` now also handles `name == "mcp_tool_heartbeat"`, reading `event["data"] == {"tool_name": str, "elapsed_seconds": float}` and calling `chat_request_wrapper.create_tool_heartbeat_sse_event(request_id=..., tool_name=..., elapsed_seconds=...)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/converters/test_streaming_manager.py`, extending `_FakeChatRequestWrapper` (top of file) with:

```python
    def create_tool_heartbeat_sse_event(
        self,
        *,
        request_id: str,
        tool_name: str,
        elapsed_seconds: float,
    ) -> str | None:
        return f"heartbeat:{tool_name}:{elapsed_seconds:.0f}"
```

Then add the test:

```python
@pytest.mark.asyncio
async def test_custom_event_mcp_tool_heartbeat_forwards_to_wrapper(
    streaming_manager_factory: Callable[[], LangGraphStreamingManager],
) -> None:
    manager = streaming_manager_factory()
    request_information = RequestInformation(request_id="req-1")
    chat_request_wrapper = _FakeChatRequestWrapper(enable_debug_logging=False)

    event = cast(
        CustomStreamEvent,
        {
            "event": "on_custom_event",
            "name": "mcp_tool_heartbeat",
            "data": {"tool_name": "propose_skill", "elapsed_seconds": 15.0},
        },
    )

    chunks = [
        chunk
        async for chunk in manager.handle_langchain_event(
            event=event,
            chat_request_wrapper=cast(ChatRequestWrapper, chat_request_wrapper),
            request_information=request_information,
            tool_start_times={},
        )
    ]
    assert chunks == ["heartbeat:propose_skill:15"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/converters/test_streaming_manager.py::test_custom_event_mcp_tool_heartbeat_forwards_to_wrapper -v`
Expected: FAIL (`AttributeError` on the fake wrapper, or no chunk is yielded because `_handle_on_custom_event` doesn't recognize the event name yet — either way, not a pass)

- [ ] **Step 3: Write minimal implementation**

In `streaming_manager.py`, replace `_handle_on_custom_event` (lines 364-383):

```python
    async def _handle_on_custom_event(
        self,
        *,
        event: StandardStreamEvent | CustomStreamEvent,
        chat_request_wrapper: ChatRequestWrapper,
        request_information: RequestInformation,
    ) -> AsyncGenerator[str, None]:
        name = event.get("name")
        if name == "mcp_task_progress":
            data: Dict[str, Any] = dict(event.get("data", {}))
            chunk = chat_request_wrapper.create_task_progress_sse_event(
                request_id=request_information.request_id,
                task_id=data.get("task_id", ""),
                status=data.get("status", ""),
                message=data.get("message"),
            )
            if chunk:
                yield chunk
        elif name == "mcp_tool_heartbeat":
            data = dict(event.get("data", {}))
            chunk = chat_request_wrapper.create_tool_heartbeat_sse_event(
                request_id=request_information.request_id,
                tool_name=data.get("tool_name", "unknown"),
                elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            )
            if chunk:
                yield chunk
        else:
            logger.debug("Skipped custom event: %s", name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/converters/test_streaming_manager.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add languagemodelcommon/converters/streaming_manager.py tests/converters/test_streaming_manager.py
git commit -m "feat: route mcp_tool_heartbeat custom events to SSE"
```

---

### Task 4: Forward real MCP progress notifications

**Files:**
- Modify: `languagemodelcommon/mcp/mcp_tool_provider.py:202-212` (`on_mcp_tool_progress`)
- Test: `tests/mcp/test_mcp_tool_provider_progress_forwarding.py`

**Interfaces:**
- Produces: `MCPToolProvider.on_mcp_tool_progress` now dispatches `adispatch_custom_event("mcp_task_progress", {"task_id": "", "status": ..., "message": ..., "server_name": ..., "tool_name": ...})` in addition to logging.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_mcp_tool_provider_progress_forwarding.py
from unittest.mock import AsyncMock, patch

import pytest

from languagemodelcommon.mcp.callbacks import CallbackContext
from languagemodelcommon.mcp.mcp_tool_provider import MCPToolProvider


@pytest.mark.asyncio
async def test_on_mcp_tool_progress_dispatches_custom_event() -> None:
    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        await MCPToolProvider.on_mcp_tool_progress(
            progress=2.0,
            total=10.0,
            message="Validating skill",
            context=CallbackContext(server_name="skills-publisher", tool_name="propose_skill"),
        )

    mock_dispatch.assert_awaited_once()
    call_args = mock_dispatch.call_args
    assert call_args.args[0] == "mcp_task_progress"
    payload = call_args.args[1]
    assert payload["server_name"] == "skills-publisher"
    assert payload["tool_name"] == "propose_skill"
    assert "Validating skill" in payload["message"]


@pytest.mark.asyncio
async def test_on_mcp_tool_progress_swallows_runtime_error_outside_callback_context() -> None:
    """adispatch_custom_event raises RuntimeError when called outside an
    active LangChain callback-manager context (e.g. this direct unit test
    invocation without a real run). on_mcp_tool_progress must not propagate
    that error -- mirrors the existing pattern in
    _execute_tool_as_task (tool_invocation.py:203-209)."""
    with patch(
        "languagemodelcommon.mcp.mcp_tool_provider.adispatch_custom_event",
        new_callable=AsyncMock,
        side_effect=RuntimeError("no callback manager in context"),
    ):
        await MCPToolProvider.on_mcp_tool_progress(
            progress=1.0,
            total=None,
            message=None,
            context=CallbackContext(server_name="skills-publisher"),
        )
    # No exception raised -- test passes by not raising.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_mcp_tool_provider_progress_forwarding.py -v`
Expected: FAIL — `mock_dispatch.assert_awaited_once()` fails because `on_mcp_tool_progress` only logs today and never imports/calls `adispatch_custom_event` (the `patch(...)` target attribute doesn't exist yet either, which itself errors first).

- [ ] **Step 3: Write minimal implementation**

At the top of `mcp_tool_provider.py`, add the import alongside the other imports (this makes it patchable at module scope, matching the test's `patch("...mcp_tool_provider.adispatch_custom_event", ...)`):

```python
from langchain_core.callbacks.manager import adispatch_custom_event
```

Replace `on_mcp_tool_progress` (lines 202-212):

```python
    @staticmethod
    async def on_mcp_tool_progress(
        *,
        progress: float,
        total: float | None,
        message: str | None,
        context: CallbackContext,
    ) -> None:
        logger.info(
            f"MCP Tool Progress - Server: {context.server_name}, Progress: {progress}, Total: {total}, Message: {message}"
        )
        try:
            await adispatch_custom_event(
                "mcp_task_progress",
                {
                    "task_id": "",
                    "status": f"{progress:g}/{total:g}" if total else f"{progress:g}",
                    "message": message,
                    "server_name": context.server_name,
                    "tool_name": context.tool_name,
                },
            )
        except RuntimeError as e:
            logger.debug(
                "Skipping mcp_task_progress event dispatch: %s (server=%s, tool=%s)",
                e,
                context.server_name,
                context.tool_name,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/mcp/test_mcp_tool_provider_progress_forwarding.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full MCP test suite to confirm no regressions**

Run: `uv run pytest tests/mcp/ -v`
Expected: All PASS (this method is also exercised indirectly by `test_mcp_tool_provider_auth_propagation.py` and `test_mcp_tool_provider_auth_discovery.py` fixtures — confirm those still pass unchanged)

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/mcp/mcp_tool_provider.py tests/mcp/test_mcp_tool_provider_progress_forwarding.py
git commit -m "feat: forward real MCP progress notifications to the SSE stream"
```

---

### Task 5: Generic heartbeat watchdog around `session.call_tool`

**Files:**
- Modify: `languagemodelcommon/mcp/mcp_client/tool_invocation.py` (add new helper; wire into `_make_execute_tool.execute_tool`, both call sites at what are currently lines 288-292 and 327-337)
- Test: `tests/mcp/test_tool_invocation_heartbeat.py`

**Interfaces:**
- Consumes: `LanguageModelCommonEnvironmentVariables.mcp_tool_heartbeat_interval_seconds` (Task 1)
- Produces: `_execute_tool_call_with_heartbeat(*, session, name, arguments, progress_callback, server_name, heartbeat_interval_seconds) -> CallToolResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_tool_invocation_heartbeat.py
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from languagemodelcommon.mcp.mcp_client.tool_invocation import (
    _execute_tool_call_with_heartbeat,
)


@pytest.mark.asyncio
async def test_fast_call_emits_no_heartbeat() -> None:
    session = AsyncMock()
    fast_result = CallToolResult(content=[TextContent(type="text", text="ok")])
    session.call_tool = AsyncMock(return_value=fast_result)

    with patch(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="propose_skill",
            arguments={},
            progress_callback=None,
            server_name="skills-publisher",
            heartbeat_interval_seconds=10.0,
        )

    assert result is fast_result
    mock_dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_call_emits_periodic_heartbeats() -> None:
    session = AsyncMock()
    slow_result = CallToolResult(content=[TextContent(type="text", text="ok")])

    async def _slow_call_tool(name: str, arguments: dict, progress_callback=None):
        await asyncio.sleep(0.25)
        return slow_result

    session.call_tool = _slow_call_tool

    with patch(
        "languagemodelcommon.mcp.mcp_client.tool_invocation.adispatch_custom_event",
        new_callable=AsyncMock,
    ) as mock_dispatch:
        result = await _execute_tool_call_with_heartbeat(
            session=session,
            name="slow_tool",
            arguments={},
            progress_callback=None,
            server_name="slow-server",
            heartbeat_interval_seconds=0.1,
        )

    assert result is slow_result
    assert mock_dispatch.await_count >= 2
    first_call = mock_dispatch.call_args_list[0]
    assert first_call.args[0] == "mcp_tool_heartbeat"
    assert first_call.args[1]["tool_name"] == "slow_tool"
    assert first_call.args[1]["server_name"] == "slow-server"


@pytest.mark.asyncio
async def test_outer_cancellation_cancels_inner_call_task() -> None:
    session = AsyncMock()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _hanging_call_tool(name: str, arguments: dict, progress_callback=None):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    session.call_tool = _hanging_call_tool

    task = asyncio.ensure_future(
        _execute_tool_call_with_heartbeat(
            session=session,
            name="hanging_tool",
            arguments={},
            progress_callback=None,
            server_name="hanging-server",
            heartbeat_interval_seconds=5.0,
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_tool_invocation_heartbeat.py -v`
Expected: FAIL with `ImportError: cannot import name '_execute_tool_call_with_heartbeat'`

- [ ] **Step 3: Write minimal implementation**

Add `import asyncio` to the top of `tool_invocation.py` (it currently has no `asyncio` import) and add the top-level import so it's patchable at module scope for the test above:

```python
import asyncio
import logging
```

```python
from langchain_core.callbacks.manager import adispatch_custom_event
```

Add this new function right before `_make_execute_tool` (before what is currently line 225):

```python
async def _execute_tool_call_with_heartbeat(
    *,
    session: Any,
    name: str,
    arguments: dict[str, Any],
    progress_callback: Any,
    server_name: str,
    heartbeat_interval_seconds: float,
) -> CallToolResult:
    """Call ``session.call_tool``, emitting a synthetic ``mcp_tool_heartbeat``
    custom event every ``heartbeat_interval_seconds`` while the call is in
    flight, regardless of whether the tool reports real progress.

    Uses ``asyncio.shield`` so a heartbeat tick's wait_for timeout never
    cancels the underlying tool call -- only the local wait is abandoned and
    retried on the same call_task. If this coroutine itself is cancelled
    (e.g. the overall chat turn is aborted), the inner call_task is
    cancelled too rather than left running in the background.
    """
    call_task: "asyncio.Task[CallToolResult]" = asyncio.ensure_future(
        session.call_tool(name, arguments, progress_callback=progress_callback)
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
    except asyncio.CancelledError:
        call_task.cancel()
        raise
```

Now wire it into both `session.call_tool(...)` call sites inside `execute_tool` in `_make_execute_tool`. Replace the session-pool branch call (currently):

```python
                return await session.call_tool(
                    request.name,
                    request.args,
                    progress_callback=mcp_callbacks.progress_callback,
                )
```

with:

```python
                return await _execute_tool_call_with_heartbeat(
                    session=session,
                    name=request.name,
                    arguments=request.args,
                    progress_callback=mcp_callbacks.progress_callback,
                    server_name=request.server_name,
                    heartbeat_interval_seconds=heartbeat_interval_seconds,
                )
```

And replace the one-shot fallback branch call (currently):

```python
                else:
                    result = await session.call_tool(
                        request.name,
                        request.args,
                        progress_callback=mcp_callbacks.progress_callback,
                    )
```

with:

```python
                else:
                    result = await _execute_tool_call_with_heartbeat(
                        session=session,
                        name=request.name,
                        arguments=request.args,
                        progress_callback=mcp_callbacks.progress_callback,
                        server_name=request.server_name,
                        heartbeat_interval_seconds=heartbeat_interval_seconds,
                    )
```

(Leave the third call site — the `TaskProtocolError` fallback branch right after `_execute_tool_as_task` fails — using the same replacement pattern, since it's the same `session.call_tool(request.name, request.args, progress_callback=mcp_callbacks.progress_callback)` shape.)

`_make_execute_tool` needs a new parameter to receive the interval. Update its signature (currently):

```python
def _make_execute_tool(
    *,
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
    session_pool: McpSessionPool | None = None,
    tool_list_cache: ToolListCache | None = None,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
```

to:

```python
def _make_execute_tool(
    *,
    config: MCPConnectionConfig,
    mcp_callbacks: _MCPCallbacks,
    session_pool: McpSessionPool | None = None,
    tool_list_cache: ToolListCache | None = None,
    heartbeat_interval_seconds: float = 15.0,
) -> Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]:
```

Find every call site of `_make_execute_tool` (`grep -rn "_make_execute_tool(" languagemodelcommon/`) and `call_mcp_tool_raw` (its public wrapper, `tool_invocation.py:348` onward — read it to find where it calls `_make_execute_tool`) and `mcp_tool_to_langchain_tool` (`languagemodelcommon/mcp/mcp_client/langchain_adapter.py` — grep for `_make_execute_tool` there too). Thread a `heartbeat_interval_seconds: float = 15.0` parameter through each of those callers' signatures the same way, and update `MCPToolProvider` (`mcp_tool_provider.py`) call sites of `mcp_tool_to_langchain_tool` and `call_mcp_tool_raw` (there are four: `get_lazy_tools`, `get_tools_by_url_async`, `execute_mcp_tool`) to pass `heartbeat_interval_seconds=self.environment_variables.mcp_tool_heartbeat_interval_seconds`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/mcp/test_tool_invocation_heartbeat.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full MCP test suite to confirm no regressions from the signature threading**

Run: `uv run pytest tests/mcp/ tests/converters/ -v`
Expected: All PASS. Pay particular attention to any test that constructs `_make_execute_tool`, `call_mcp_tool_raw`, or `mcp_tool_to_langchain_tool` directly with positional-looking keyword sets — the new parameter has a default so no existing call site should break, but confirm.

- [ ] **Step 6: Commit**

```bash
git add languagemodelcommon/mcp/mcp_client/tool_invocation.py languagemodelcommon/mcp/mcp_client/langchain_adapter.py languagemodelcommon/mcp/mcp_tool_provider.py tests/mcp/test_tool_invocation_heartbeat.py
git commit -m "feat: add heartbeat watchdog around session.call_tool"
```

---

### Task 6: Full test suite, version bump, and release

**Files:**
- Modify: `VERSION`
- Modify: `CHANGELOG.md` (if one exists — check first: `ls CHANGELOG.md`)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS. This is the first point where Tasks 1-5's cross-file wiring is exercised together.

- [ ] **Step 2: Run pre-commit / lint / type-check**

Run whatever this repo's equivalent of `make run-pre-commit` is — check `Makefile` or `CLAUDE.md` for the exact command:

```bash
grep -n "pre-commit\|mypy\|ruff" Makefile CLAUDE.md 2>/dev/null | head -20
```

Run the discovered command and fix any lint/type errors surfaced by the new code before proceeding.

- [ ] **Step 3: Bump VERSION and open a PR**

This repo's `python-publish.yml` workflow triggers on a GitHub **release** being created (`on: release: types: [created]`), which sets `VERSION` from the release tag at publish time — so no manual `VERSION` file edit is needed in the PR itself. Confirm this is still accurate:

```bash
cat .github/workflows/python-publish.yml
```

Open a PR with all commits from Tasks 1-5. Title: "Forward MCP tool progress and heartbeats to the SSE stream". In the PR description, call out explicitly: default behavior is unchanged for all existing consumers (both new flags default to producing no new visible output); `EMIT_TOOL_HEARTBEAT_IN_CHAT_COMPLETIONS` is a separate opt-in from the existing `EMIT_TASK_PROGRESS_IN_CHAT_COMPLETIONS` so nobody's current configuration gets noisier without a second explicit change.

- [ ] **Step 4: After merge, create a GitHub release**

Follow this repo's existing release process (check whether releases are cut manually via `gh release create` or through another automated flow — grep recent release history: `gh release list --limit 5`) to cut a new version. Use the next patch or minor version after whatever `gh release list` shows as latest (this plan does not hardcode a version number since the actual latest tag at execution time may differ from what was observed while writing this plan).

---

### Task 7: Bump the pin in `baileyai` and verify end-to-end

**Files:**
- Modify: `/Users/imranqureshi/git/baileyai/pyproject.toml:39`
- Modify: `/Users/imranqureshi/git/baileyai/uv.lock` (regenerated, not hand-edited)

**Interfaces:**
- Consumes: the new `language-model-common` release published in Task 6.

- [ ] **Step 1: Bump the version pin**

In `/Users/imranqureshi/git/baileyai/pyproject.toml:39`, change:

```toml
    "language-model-common>=2.0.85",
```

to the new version published in Task 6, e.g.:

```toml
    "language-model-common>=2.0.86",
```

- [ ] **Step 2: Re-lock dependencies**

Run whatever this repo's dependency lock command is (check `baileyai`'s own `Makefile`/`CLAUDE.md` first — likely something like `make uv.lock` mirroring `baileyai-skills-service`'s convention):

```bash
cd /Users/imranqureshi/git/baileyai && grep -n "uv.lock\|uv lock" Makefile 2>/dev/null
```

Run the discovered command (or `uv lock` directly if no Makefile target exists) so `uv.lock` picks up the new `language-model-common` version and its transitive dependency tree.

- [ ] **Step 3: Run baileyai's test suite**

Run this repo's test command (check `Makefile`/`CLAUDE.md`) to confirm nothing in baileyai's own code broke from the dependency bump.

- [ ] **Step 4: Manual verification against a slow tool**

Start baileyai locally against an MCP server/tool known to take longer than the configured `MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS` (or temporarily set `MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS=2` for a fast manual check), trigger a chat turn that invokes that tool, and confirm via the raw SSE response (e.g. `curl -N` against the responses endpoint, or browser devtools Network tab if testing through baileyai-skills-service) that `task.progress` frames with `"Still running <tool>... (Ns)"` messages appear every ~15s while the tool call is in flight — and confirm the baileyai-skills-service chat UI does not show that text inline in the assistant's message bubble (only as a separate trace/progress indicator, if the UI surfaces one at all).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump language-model-common for MCP tool progress/heartbeat forwarding"
```

---

## Known Limitations / Follow-ups (not in scope for this plan)

- The heartbeat only fires for tool calls that go through `_make_execute_tool.execute_tool` (i.e. real `session.call_tool` invocations). If a future code path calls `session.call_tool` directly without going through this helper, it won't get heartbeats — grep for other `session.call_tool(` call sites before assuming full coverage is automatic for new code.
- This plan reuses the `task.progress` SSE type for heartbeats on the Responses API path rather than inventing a new wire type, to avoid needing a corresponding baileyai-skills-service frontend change. If a future consumer needs to visually distinguish "real tool progress" from "synthetic heartbeat" in its UI, it would need a new `type` value end-to-end (a separate, larger change spanning both repos) — not attempted here.
