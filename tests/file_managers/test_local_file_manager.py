from pathlib import Path

import pytest
from fastapi import HTTPException

from languagemodelcommon.file_managers.file_manager import FileManager
from languagemodelcommon.file_managers.local_file_manager import LocalFileManager


@pytest.mark.asyncio
async def test_save_and_read_file_async_round_trip(tmp_path: Path) -> None:
    local_file_manager = LocalFileManager()
    file_data = b"safe-content"

    saved_file_path = await local_file_manager.save_file_async(
        file_data=file_data,
        folder=str(tmp_path),
        filename="safe.txt",
        content_type="text/plain",
    )

    assert saved_file_path == str(tmp_path / "safe.txt")

    response = await local_file_manager.read_file_async(
        folder=str(tmp_path), file_path="safe.txt"
    )
    assert response.status_code == 200

    extracted_content = await FileManager.extract_content(response)
    assert extracted_content == "safe-content"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malicious_filename",
    [
        "../secrets.txt",
        "../../etc/passwd",
        "/etc/passwd",
    ],
)
async def test_save_file_async_rejects_path_traversal(
    tmp_path: Path, malicious_filename: str
) -> None:
    local_file_manager = LocalFileManager()

    with pytest.raises(ValueError, match="Invalid file path"):
        await local_file_manager.save_file_async(
            file_data=b"blocked",
            folder=str(tmp_path),
            filename=malicious_filename,
            content_type="text/plain",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malicious_path",
    [
        "../secrets.txt",
        "../../etc/passwd",
        "/etc/passwd",
    ],
)
async def test_read_file_async_rejects_path_traversal(
    tmp_path: Path, malicious_path: str
) -> None:
    local_file_manager = LocalFileManager()

    with pytest.raises(HTTPException) as error:
        await local_file_manager.read_file_async(
            folder=str(tmp_path),
            file_path=malicious_path,
        )

    assert error.value.status_code == 400
    assert error.value.detail == "Invalid file path"


@pytest.mark.asyncio
async def test_read_file_async_not_found_hides_internal_path(tmp_path: Path) -> None:
    local_file_manager = LocalFileManager()

    with pytest.raises(HTTPException) as error:
        await local_file_manager.read_file_async(
            folder=str(tmp_path),
            file_path="missing-file.txt",
        )

    assert error.value.status_code == 404
    assert error.value.detail == "File not found"
