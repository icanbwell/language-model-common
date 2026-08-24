from oidcauthlib.auth.exceptions.authorization_bearer_token_invalid_exception import (
    AuthorizationBearerTokenInvalidException,
)

from languagemodelcommon.exceptions.bailey_exception import BaileyException

EXCEPTION_TYPE_CODES: dict[type[BaseException], str] = {
    ValueError: "100",
    TypeError: "101",
    KeyError: "102",
    TimeoutError: "103",
    ConnectionError: "104",
    PermissionError: "401",
    ExceptionGroup: "199",
    BaileyException: "500",
    # Distinct from PermissionError (401): this means the *caller's own*
    # bearer token doesn't validate (unrecognized issuer/audience, missing
    # claims, bad signature/kid) - not that the caller lacks permission for
    # an otherwise-valid identity.
    AuthorizationBearerTokenInvalidException: "498",
}

# Per-exception-type end-user messages that are more actionable than the
# generic default. Checked before falling back to the caller-supplied
# generic_message/default_message. Only add an exception type here when a
# *specific, accurate* message is possible - don't add one that would guess
# at a root cause we can't actually attribute from the exception type alone.
EXCEPTION_TYPE_MESSAGES: dict[type[BaseException], str] = {
    # Root-caused from a real incident (Aug 2026): this exception fires for
    # several distinct token problems (malformed token, missing aud/client_id
    # claim, unrecognized signing key, or - the one seen in practice most
    # often - an audience/issuer that doesn't match any of this service's
    # configured auth providers, e.g. a newly onboarded org/environment
    # whose trust config hasn't been added yet). Logging out and back in
    # only helps the first two; it never helps the last one, since a fresh
    # login from the same client produces a token with the identical
    # issuer/audience. The message below covers both without asserting
    # which applies, and gives the user a next step either way instead of a
    # raw exception class name.
    AuthorizationBearerTokenInvalidException: (
        "We couldn't verify your session for this environment. Try logging "
        "out and back in — if that doesn't resolve it, this environment may "
        "not be fully set up yet; please contact support."
    ),
}


class ExceptionFormatter:
    @staticmethod
    def get_exception_code(error: BaseException) -> str:
        for exception_type, code in EXCEPTION_TYPE_CODES.items():
            if isinstance(error, exception_type):
                return code
        return error.__class__.__name__

    @staticmethod
    def format_generic_message(
        error: BaseException,
        *,
        generic_message: str | None,
        default_message: str,
    ) -> str:
        for exception_type, message in EXCEPTION_TYPE_MESSAGES.items():
            if isinstance(error, exception_type):
                return (
                    f"{message} (Code: {ExceptionFormatter.get_exception_code(error)})"
                )
        base_message = generic_message or default_message
        return f"{base_message} (Code: {ExceptionFormatter.get_exception_code(error)})"
