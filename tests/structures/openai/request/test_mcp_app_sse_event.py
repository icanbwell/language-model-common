"""Tests for ChatRequestWrapper.create_mcp_app_sse_event's type/protocolVersion
discriminator (BAI-960). Both ChatCompletionApiRequestWrapper and
ResponsesApiRequestWrapper override this the same way -- unlike
create_image_output_sse_event, this one is not asymmetric."""

import json
from typing import Any
from unittest.mock import MagicMock

from languagemodelcommon.schema.openai.completions import ChatRequest
from languagemodelcommon.schema.openai.responses import ResponsesRequest
from languagemodelcommon.structures.openai.request.chat_completion_api_request_wrapper import (
    ChatCompletionApiRequestWrapper,
)
from languagemodelcommon.structures.openai.request.chat_request_wrapper import (
    ChatRequestWrapper,
)
from languagemodelcommon.structures.openai.request.responses_api_request_wrapper import (
    ResponsesApiRequestWrapper,
)


def _make_env() -> MagicMock:
    env = MagicMock()
    env.debug_prefixes = ("DEBUG:", "/debug ")
    env.emit_task_progress_in_chat_completions = False
    env.emit_tool_heartbeat_in_chat_completions = False
    return env


def _make_responses_wrapper() -> ResponsesApiRequestWrapper:
    request = ResponsesRequest(
        model="gpt-4",
        input="hello",
        stream=False,
        instructions=None,
        previous_response_id=None,
        store=False,
        temperature=None,
        top_p=None,
        max_output_tokens=None,
        tools=None,
        parallel_tool_calls=None,
        tool_choice=None,
        metadata=None,
    )
    return ResponsesApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


def _make_chat_completion_wrapper() -> ChatCompletionApiRequestWrapper:
    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4",
        stream=False,
    )
    return ChatCompletionApiRequestWrapper(
        chat_request=request,
        enable_debug_logging=False,
        environment_variables=_make_env(),
    )


def _parse_payload(raw: str) -> dict[str, Any]:
    assert raw.startswith("event: mcp_app\n")
    return dict(json.loads(raw[len("event: mcp_app\ndata: ") :]))


def test_base_wrapper_mcp_app_is_concrete_not_a_noop() -> None:
    # Unlike this base class's other create_*_sse_event defaults,
    # create_mcp_app_sse_event is concrete here (not a no-op) since both
    # subclasses emitted an identical body -- extracted per BAI-960 code
    # review to remove that duplication. The base implementation never
    # touches `self`, so call it as an unbound function to confirm this
    # without instantiating the ABC (mirrors test_base_wrapper_tool_heartbeat_is_noop
    # / test_base_wrapper_image_output_is_noop's unbound-call technique).
    result = ChatRequestWrapper.create_mcp_app_sse_event(
        object(),  # type: ignore[arg-type]
        html="<div/>",
    )
    assert result is not None
    payload = _parse_payload(result)
    assert payload["type"] == "mcp_app"
    assert payload["html"] == "<div/>"


def test_responses_api_wrapper_includes_type_and_protocol_version() -> None:
    wrapper = _make_responses_wrapper()
    raw = wrapper.create_mcp_app_sse_event(html="<div>hi</div>")
    assert raw is not None
    payload = _parse_payload(raw)
    assert payload["type"] == "mcp_app"
    assert payload["protocolVersion"] == "2026-01-26"
    assert payload["html"] == "<div>hi</div>"


def test_chat_completion_wrapper_includes_type_and_protocol_version() -> None:
    wrapper = _make_chat_completion_wrapper()
    raw = wrapper.create_mcp_app_sse_event(html="<div>hi</div>")
    assert raw is not None
    payload = _parse_payload(raw)
    assert payload["type"] == "mcp_app"
    assert payload["protocolVersion"] == "2026-01-26"
    assert payload["html"] == "<div>hi</div>"


def test_omits_resource_uri_when_not_given() -> None:
    for wrapper in (_make_responses_wrapper(), _make_chat_completion_wrapper()):
        raw = wrapper.create_mcp_app_sse_event(html="<div/>")
        assert raw is not None
        assert "resourceUri" not in _parse_payload(raw)


def test_includes_resource_uri_when_given() -> None:
    for wrapper in (_make_responses_wrapper(), _make_chat_completion_wrapper()):
        raw = wrapper.create_mcp_app_sse_event(
            html="<div/>", resource_uri="ui://server/widget"
        )
        assert raw is not None
        assert _parse_payload(raw)["resourceUri"] == "ui://server/widget"


def test_preserves_existing_optional_fields() -> None:
    for wrapper in (_make_responses_wrapper(), _make_chat_completion_wrapper()):
        raw = wrapper.create_mcp_app_sse_event(
            html="<div/>",
            title="My App",
            csp={"connectDomains": ["https://api.example.com"]},
            permissions={"camera": {}},
            prefers_border=True,
            display_mode="inline",
        )
        assert raw is not None
        payload = _parse_payload(raw)
        assert payload["title"] == "My App"
        assert payload["csp"] == {"connectDomains": ["https://api.example.com"]}
        assert payload["permissions"] == {"camera": {}}
        assert payload["prefersBorder"] is True
        assert payload["displayMode"] == "inline"


def test_omits_optional_fields_when_not_given() -> None:
    for wrapper in (_make_responses_wrapper(), _make_chat_completion_wrapper()):
        raw = wrapper.create_mcp_app_sse_event(html="<div/>")
        assert raw is not None
        payload = _parse_payload(raw)
        assert "title" not in payload
        assert "csp" not in payload
        assert "permissions" not in payload
        assert "prefersBorder" not in payload
        assert "displayMode" not in payload
