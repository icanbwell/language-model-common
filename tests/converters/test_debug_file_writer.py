from pathlib import Path
import re

import pytest

from languagemodelcommon.aws.aws_client_factory import AwsClientFactory
from languagemodelcommon.converters.debug_file_writer import DebugFileWriter
from languagemodelcommon.file_managers.file_manager_factory import FileManagerFactory


@pytest.fixture
def debug_file_writer() -> DebugFileWriter:
    return DebugFileWriter(
        file_manager_factory=FileManagerFactory(aws_client_factory=AwsClientFactory()),
    )


@pytest.mark.parametrize(
    "tool_name,expected_prefix",
    [
        ("tool name/with spaces", "tool_name_with_spaces_"),
        (None, "unknown_"),
        ("***", "unknown_"),
    ],
)
def test_generate_secure_filename(
    debug_file_writer: DebugFileWriter,
    tool_name: str | None,
    expected_prefix: str,
) -> None:
    filename = debug_file_writer.generate_secure_filename(
        tool_name=tool_name,
        user_id="user-123",
    )

    assert filename.startswith(expected_prefix)
    assert filename.endswith(".txt")
    assert "user-123" not in filename
    assert re.match(r"^[A-Za-z0-9._-]+\.txt$", filename) is not None


@pytest.mark.asyncio
async def test_write_content_saves_file_and_returns_url(
    debug_file_writer: DebugFileWriter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_GENERATION_URL", "https://example.test/files")

    result = await debug_file_writer.write_content(
        content="debug output",
        output_folder=str(tmp_path),
        filename="test-debug.txt",
    )

    assert result.file_path is not None
    assert Path(result.file_path).exists()
    assert Path(result.file_path).read_text(encoding="utf-8") == "debug output"
    assert result.file_url == "https://example.test/files/test-debug.txt"
    assert result.url_error_message is None


@pytest.mark.asyncio
async def test_write_content_returns_missing_url_error_when_env_not_set(
    debug_file_writer: DebugFileWriter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("IMAGE_GENERATION_URL", raising=False)

    result = await debug_file_writer.write_content(
        content="debug output",
        output_folder=str(tmp_path),
        filename="test-debug-no-url.txt",
    )

    assert result.file_path is not None
    assert result.file_url is None
    assert (
        result.url_error_message
        == "Tool output file URL could not be generated due to missing IMAGE_GENERATION_URL environment variable."
    )


@pytest.mark.asyncio
async def test_write_content_returns_none_path_when_content_is_empty(
    debug_file_writer: DebugFileWriter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGE_GENERATION_URL", "https://example.test/files")

    result = await debug_file_writer.write_content(
        content="",
        output_folder=str(tmp_path),
        filename="test-empty.txt",
    )

    assert result.file_path is None
    assert result.file_url is None
    assert result.url_error_message is None
