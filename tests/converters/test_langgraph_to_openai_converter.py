from languagemodelcommon.converters.langgraph_to_openai_converter import (
    LangGraphToOpenAIConverter,
)


class _ReadTimeoutNamedException(Exception):
    pass


class _GenericException(Exception):
    pass


def test_is_timeout_exception_returns_true_for_builtin_timeout() -> None:
    assert LangGraphToOpenAIConverter._is_timeout_exception(TimeoutError("timeout"))


def test_is_timeout_exception_returns_true_for_named_timeout_class() -> None:
    assert LangGraphToOpenAIConverter._is_timeout_exception(
        _ReadTimeoutNamedException("timed out")
    )


def test_is_timeout_exception_returns_true_for_wrapped_timeout() -> None:
    wrapped_exception = _GenericException("wrapper")
    wrapped_exception.__cause__ = _ReadTimeoutNamedException("inner timeout")

    assert LangGraphToOpenAIConverter._is_timeout_exception(wrapped_exception)


def test_is_timeout_exception_returns_false_for_non_timeout_error() -> None:
    assert not LangGraphToOpenAIConverter._is_timeout_exception(
        _GenericException("not timeout")
    )
