"""Tests for LoggingResponse's DEBUG-log hardening.

BAI-955: `aiter_bytes` used to log every streamed response chunk verbatim at
DEBUG. Not a live leak today (LOG_LEVEL is INFO everywhere in Helm values),
but this closes the rollout-hardening gap so enabling DEBUG never leaks a
streamed chunk's content.
"""

import logging

import httpx2
import pytest

from languagemodelcommon.utilities.logger.logging_response import LoggingResponse


class TestLoggingResponseStreamedChunks:
    @pytest.mark.asyncio
    async def test_does_not_log_raw_chunk_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        marker = "MARKER-DO-NOT-LEAK-STREAMED-CHUNK"
        request = httpx2.Request("GET", "https://example.test/resource")
        response = LoggingResponse(
            200,
            request=request,
            content=marker.encode("utf-8"),
        )

        with caplog.at_level(
            logging.DEBUG,
            logger="languagemodelcommon.utilities.logger.logging_response",
        ):
            async for _ in response.aiter_bytes():
                pass

        assert marker not in caplog.text
