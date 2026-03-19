"""Tests for ExceptionLogger user-friendly error handling."""

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    DEFAULT_GENERIC_ERROR_MESSAGE,
)
from languagemodelcommon.utilities.logger.exception_formatter import (
    EXCEPTION_TYPE_CODES,
)
from languagemodelcommon.utilities.logger.exception_logger import ExceptionLogger


class CustomError(Exception):
    pass


class TestGetUserFriendlyMessage:
    """Tests for ExceptionLogger.get_user_friendly_message()."""

    def test_returns_generic_message_when_debug_disabled(self) -> None:
        """When debug logging is disabled, should return the generic message."""
        error = ValueError("Internal technical error with stack trace")

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=False,
        )

        expected_code = EXCEPTION_TYPE_CODES[ValueError]
        assert result == f"{DEFAULT_GENERIC_ERROR_MESSAGE} (Code: {expected_code})"

    def test_returns_custom_generic_message_when_debug_disabled(self) -> None:
        """When debug logging is disabled with custom message, should return the custom message."""
        error = ValueError("NoneType' object has no attribute 'tools'")
        custom_message = "Something went wrong. Please try again."

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=False,
            generic_message=custom_message,
        )

        expected_code = EXCEPTION_TYPE_CODES[ValueError]
        assert result == f"{custom_message} (Code: {expected_code})"

    def test_returns_error_details_when_debug_enabled(self) -> None:
        """When debug logging is enabled, should return the full error details."""
        error = ValueError("Specific error message for debugging")

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=True,
        )

        assert "Specific error message for debugging" in result

    def test_returns_error_details_with_custom_message_when_debug_enabled(self) -> None:
        """When debug logging is enabled, custom message is ignored and full details shown."""
        error = ValueError("Technical error details")
        custom_message = "Generic error"

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=True,
            generic_message=custom_message,
        )

        assert "Technical error details" in result
        assert result != custom_message

    def test_handles_exception_group_when_debug_disabled(self) -> None:
        """Should return generic message for ExceptionGroup when debug disabled."""
        exceptions = [
            ValueError("First error"),
            TypeError("Second error"),
        ]
        error = ExceptionGroup("Multiple errors", exceptions)

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=False,
        )

        expected_code = EXCEPTION_TYPE_CODES[ExceptionGroup]
        assert result == f"{DEFAULT_GENERIC_ERROR_MESSAGE} (Code: {expected_code})"
        # Should not contain any technical details
        assert "First error" not in result
        assert "Second error" not in result

    def test_handles_exception_group_when_debug_enabled(self) -> None:
        """Should return full details for ExceptionGroup when debug enabled."""
        exceptions = [
            ValueError("First error"),
            TypeError("Second error"),
        ]
        error = ExceptionGroup("Multiple errors", exceptions)

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=True,
        )

        assert "Exception Group" in result
        assert "First error" in result
        assert "Second error" in result

    def test_falls_back_to_generic_when_extract_returns_none(self) -> None:
        """If extract_error_details returns None, should still return generic message."""
        from unittest.mock import patch

        error = ValueError("Some error")

        # Mock extract_error_details to return None to test fallback behavior
        with patch.object(ExceptionLogger, "extract_error_details", return_value=None):
            result = ExceptionLogger.get_user_friendly_message(
                error,
                enable_debug_logging=True,
                generic_message="Fallback message",
            )

            expected_code = EXCEPTION_TYPE_CODES[ValueError]
            # Should fall back to generic message when extract returns None
            assert result == f"Fallback message (Code: {expected_code})"

    def test_does_not_expose_stack_trace_when_debug_disabled(self) -> None:
        """Ensure stack traces are not exposed to users when debug logging is disabled."""

        def cause_nested_error() -> None:
            try:
                raise RuntimeError("Inner error in langchain_mcp_adapters")
            except RuntimeError as inner:
                raise ValueError("Outer error") from inner

        try:
            cause_nested_error()
        except ValueError as e:
            result = ExceptionLogger.get_user_friendly_message(
                e,
                enable_debug_logging=False,
            )

            expected_code = EXCEPTION_TYPE_CODES[ValueError]
            # Should not contain any technical details
            assert result == f"{DEFAULT_GENERIC_ERROR_MESSAGE} (Code: {expected_code})"
            assert "langchain_mcp_adapters" not in result
            assert "RuntimeError" not in result
            assert "Inner error" not in result
            assert "Traceback" not in result

    def test_returns_exception_name_when_no_mapping_exists(self) -> None:
        """Unmapped exceptions should use the exception class name as the code."""
        error = CustomError("Unmapped error")

        result = ExceptionLogger.get_user_friendly_message(
            error,
            enable_debug_logging=False,
        )

        assert result == f"{DEFAULT_GENERIC_ERROR_MESSAGE} (Code: CustomError)"


class TestExtractErrorDetails:
    """Tests for ExceptionLogger.extract_error_details() - maintains existing behavior."""

    def test_extracts_simple_exception_message(self) -> None:
        """Should extract the message from a simple exception."""
        error = ValueError("Test error message")

        result = ExceptionLogger.extract_error_details(error)

        assert result is not None
        assert "Test error message" in result

    def test_extracts_exception_group_message(self) -> None:
        """Should extract messages from an ExceptionGroup."""
        exceptions = [
            ValueError("Error one"),
            TypeError("Error two"),
        ]
        error = ExceptionGroup("unhandled errors in a TaskGroup", exceptions)

        result = ExceptionLogger.extract_error_details(error)

        assert result is not None
        assert "Exception Group: unhandled errors in a TaskGroup" in result
        assert "Error one" in result
        assert "Error two" in result

    def test_handles_nested_exception_chain(self) -> None:
        """Should handle exceptions with __cause__."""
        try:
            try:
                raise RuntimeError("Root cause")
            except RuntimeError as root:
                raise ValueError("Wrapped error") from root
        except ValueError as e:
            result = ExceptionLogger.extract_error_details(e)

            assert result is not None
            assert "Wrapped error" in result
            # The cause should be included in the message chain
            assert "Root cause" in result
