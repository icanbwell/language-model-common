"""Tests for ExceptionFormatter's per-exception-type code/message overrides."""

from oidcauthlib.auth.exceptions.authorization_bearer_token_invalid_exception import (
    AuthorizationBearerTokenInvalidException,
)

from languagemodelcommon.utilities.logger.exception_formatter import (
    EXCEPTION_TYPE_CODES,
    ExceptionFormatter,
)


class TestGetExceptionCode:
    """Tests for ExceptionFormatter.get_exception_code()."""

    def test_returns_dedicated_code_for_auth_token_invalid(self) -> None:
        """AuthorizationBearerTokenInvalidException should get its own code,
        not fall through to the raw class name."""
        error = AuthorizationBearerTokenInvalidException(
            message="bad token", token="tok"
        )

        code = ExceptionFormatter.get_exception_code(error)

        assert code == EXCEPTION_TYPE_CODES[AuthorizationBearerTokenInvalidException]
        assert code != "AuthorizationBearerTokenInvalidException"


class TestFormatGenericMessage:
    """Tests for ExceptionFormatter.format_generic_message()."""

    def test_uses_specific_message_for_auth_token_invalid(self) -> None:
        """The caller's generic_message should be overridden by the more
        actionable, exception-specific message for this type."""
        error = AuthorizationBearerTokenInvalidException(
            message="bad token", token="tok"
        )

        result = ExceptionFormatter.format_generic_message(
            error,
            generic_message="An error occurred processing your request.",
            default_message="Fallback",
        )

        # Doesn't leak the raw exception class name or the caller's generic text.
        assert "AuthorizationBearerTokenInvalidException" not in result
        assert "An error occurred processing your request." not in result
        # Gives the user two concrete next steps instead.
        assert "logging out and back in" in result
        assert "contact support" in result
        expected_code = EXCEPTION_TYPE_CODES[AuthorizationBearerTokenInvalidException]
        assert result.endswith(f"(Code: {expected_code})")

    def test_unmapped_exception_still_uses_caller_generic_message(self) -> None:
        """Exception types without a specific-message override keep the
        existing caller-supplied generic_message behavior."""
        error = ValueError("some internal detail")

        result = ExceptionFormatter.format_generic_message(
            error,
            generic_message="Custom message",
            default_message="Fallback",
        )

        assert result.startswith("Custom message")
        assert "some internal detail" not in result
