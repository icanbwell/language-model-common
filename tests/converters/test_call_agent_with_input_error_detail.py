"""Regression tests for a Gecko finding: call_agent_with_input's
non-streaming exception handlers used to embed raw exception text directly
into the HTTPException `detail` field unconditionally, leaking internal
implementation details to API callers. The streaming path already gated
this behind `enable_debug_logging`; the non-streaming path did not.
"""

from typing import Any, List

import pytest
from botocore.exceptions import NoCredentialsError, TokenRetrievalError
from fastapi import HTTPException
from langchain_core.messages import AnyMessage

from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)
from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.request_information import RequestInformation


class _FakeChatRequestWrapper:
    """Minimal wrapper covering only what the non-streaming branch of
    call_agent_with_input touches."""

    def __init__(self, *, enable_debug_logging: bool) -> None:
        self.enable_debug_logging = enable_debug_logging
        self.stream = False
        self.response_format = None
        self.appended_messages: list[Any] = []

    def create_system_message(self, *, content: str) -> str:
        return content

    def append_message(self, *, message: Any) -> None:
        self.appended_messages.append(message)

    def create_non_streaming_response(
        self,
        *,
        request_id: str,
        responses: List[AnyMessage],
        json_output_requested: bool,
    ) -> dict[str, Any]:
        return {"id": request_id}


def _build_converter(monkeypatch: pytest.MonkeyPatch) -> LangGraphToOpenAIConverter:
    monkeypatch.setenv("LANGGRAPH_RECURSION_LIMIT", "88")
    converter = object.__new__(LangGraphToOpenAIConverter)
    converter.environment_variables = LanguageModelCommonEnvironmentVariables()
    return converter


def _request_information() -> RequestInformation:
    return RequestInformation(
        request_id="req-1", conversation_thread_id="thread-1", user_id="user-1"
    )


async def _raise(exc: BaseException) -> List[AnyMessage]:
    raise exc


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_debug_logging", [False, True])
async def test_generic_exception_detail_gated_by_debug_logging(
    monkeypatch: pytest.MonkeyPatch, enable_debug_logging: bool
) -> None:
    converter = _build_converter(monkeypatch)
    secret = "s3cr3t-internal-path/etc/shadow-ish-detail"
    monkeypatch.setattr(
        converter, "ainvoke", lambda **_kwargs: _raise(RuntimeError(secret))
    )

    with pytest.raises(HTTPException) as exc_info:
        await converter.call_agent_with_input(
            chat_request_wrapper=_FakeChatRequestWrapper(
                enable_debug_logging=enable_debug_logging
            ),
            compiled_state_graph=None,  # type: ignore[arg-type]
            system_messages=[],
            request_information=_request_information(),
            config=None,
            state=None,
        )

    assert exc_info.value.status_code == 500
    detail = str(exc_info.value.detail)
    if enable_debug_logging:
        assert secret in detail
    else:
        assert secret not in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_debug_logging", [False, True])
async def test_token_retrieval_error_detail_gated_by_debug_logging(
    monkeypatch: pytest.MonkeyPatch, enable_debug_logging: bool
) -> None:
    converter = _build_converter(monkeypatch)
    secret = "arn:aws:iam::123456789012:role/super-secret-role"
    monkeypatch.setattr(
        converter,
        "ainvoke",
        lambda **_kwargs: _raise(TokenRetrievalError(provider="sso", error_msg=secret)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await converter.call_agent_with_input(
            chat_request_wrapper=_FakeChatRequestWrapper(
                enable_debug_logging=enable_debug_logging
            ),
            compiled_state_graph=None,  # type: ignore[arg-type]
            system_messages=[],
            request_information=_request_information(),
            config=None,
            state=None,
        )

    assert exc_info.value.status_code == 401
    detail = str(exc_info.value.detail)
    assert "re-authenticate" in detail.lower()
    if enable_debug_logging:
        assert secret in detail
    else:
        assert secret not in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_debug_logging", [False, True])
async def test_no_credentials_error_detail_gated_by_debug_logging(
    monkeypatch: pytest.MonkeyPatch, enable_debug_logging: bool
) -> None:
    converter = _build_converter(monkeypatch)
    monkeypatch.setattr(
        converter,
        "ainvoke",
        lambda **_kwargs: _raise(NoCredentialsError()),
    )
    # NoCredentialsError's message is a fixed botocore string ("Unable to
    # locate credentials"), not caller-supplied -- assert on that literal
    # rather than an injected secret.
    with pytest.raises(HTTPException) as exc_info:
        await converter.call_agent_with_input(
            chat_request_wrapper=_FakeChatRequestWrapper(
                enable_debug_logging=enable_debug_logging
            ),
            compiled_state_graph=None,  # type: ignore[arg-type]
            system_messages=[],
            request_information=_request_information(),
            config=None,
            state=None,
        )

    assert exc_info.value.status_code == 401
    detail = str(exc_info.value.detail)
    assert "re-authenticate" in detail.lower()
    if enable_debug_logging:
        assert "Unable to locate credentials" in detail
    else:
        assert "Unable to locate credentials" not in detail
