from typing import Optional

from languagemodelcommon.exceptions.bailey_exception import BaileyException


class RateLimitException(BaileyException):
    """Raised when an upstream model rejects a request with a rate limit (HTTP 429).

    Subclasses :class:`BaileyException` so existing callers that catch the base
    type continue to work. Carries an optional ``retry_after_seconds`` hint
    parsed from the upstream ``Retry-After`` header so callers can honor the
    server's requested backoff.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
