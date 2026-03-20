"""Tests for source-level logger settings."""

import pytest

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS


@pytest.mark.parametrize("source_name", ["HTTP_TRACING", "HTTP", "CONFIG", "TOOLS"])
def test_source_log_levels_support_attribute_and_key_access(source_name: str) -> None:
    """Known source names should be readable through both access styles."""
    assert SRC_LOG_LEVELS[source_name] == getattr(SRC_LOG_LEVELS, source_name)


def test_source_log_levels_missing_attribute_raises_attribute_error() -> None:
    """Unknown attribute access should raise AttributeError."""
    with pytest.raises(AttributeError):
        _ = SRC_LOG_LEVELS.NOT_A_REAL_SOURCE


def test_source_log_levels_missing_key_raises_key_error() -> None:
    """Unknown key access should raise KeyError."""
    with pytest.raises(KeyError):
        _ = SRC_LOG_LEVELS["NOT_A_REAL_SOURCE"]
