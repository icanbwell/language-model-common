from __future__ import annotations

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
        base_message = generic_message or default_message
        return f"{base_message} (Code: {ExceptionFormatter.get_exception_code(error)})"
