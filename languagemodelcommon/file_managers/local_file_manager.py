import logging
import mimetypes
import os
from os import makedirs
from pathlib import Path
from typing import Optional, AsyncGenerator, override

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from languagemodelcommon.file_managers.file_manager import FileManager
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.FILES)


class LocalFileManager(FileManager):
    @staticmethod
    def _resolve_safe_path(
        *, folder: str, relative_path: str, create_folder: bool = False
    ) -> Path:
        base_path = Path(folder).resolve()
        if create_folder:
            makedirs(base_path, exist_ok=True)

        resolved_path = (base_path / relative_path).resolve()
        try:
            resolved_path.relative_to(base_path)
        except ValueError as ex:
            raise ValueError("Invalid file path") from ex

        return resolved_path

    @override
    async def save_file_async(
        self,
        *,
        file_data: bytes,
        folder: str,
        filename: str,
        content_type: str,
    ) -> Optional[str]:
        """Save the generated image to a file"""
        file_path: str = self.get_full_path(filename=filename, folder=folder)
        if file_data:
            with open(file_path, "wb") as f:
                f.write(file_data)
            logger.info(f"File saved as {file_path}")
            return str(file_path)
        else:
            logger.error("No file to save")
            return None

    @override
    def get_full_path(self, *, filename: str, folder: str) -> str:
        file_path = self._resolve_safe_path(
            folder=folder, relative_path=filename, create_folder=True
        )
        return str(file_path)

    # @override
    # async def save_image_async(self, image_data: bytes, filename: Path) -> None:
    #     """Save the generated image to a file asynchronously"""
    #     if not image_data:
    #         logger.warning("No image data to save")
    #         return
    #
    #     try:
    #         # Use aiofiles for async file operations
    #         async with aiofiles.open(filename, mode='wb') as f:
    #             await f.write(image_data)
    #         logger.info(f"Image saved as {filename}")
    #
    #     except Exception as e:
    #         logger.error(f"Error saving image to {filename}: {str(e)}")
    #         raise

    @override
    async def read_file_async(
        self, *, folder: str, file_path: str
    ) -> StreamingResponse:
        try:
            full_path = self._resolve_safe_path(folder=folder, relative_path=file_path)
            # Determine file size and MIME type
            file_size = os.path.getsize(full_path)
            mime_type, _ = mimetypes.guess_type(full_path)
            mime_type = mime_type or "application/octet-stream"

            # Open file as a generator to stream content
            async def file_iterator() -> AsyncGenerator[bytes, None]:
                with open(full_path, "rb") as file:
                    while chunk := file.read(4096):  # Read in 4KB chunks
                        yield chunk

            return StreamingResponse(
                file_iterator(),
                media_type=mime_type,
                headers={
                    "Content-Length": str(file_size),
                    "Content-Disposition": f'inline; filename="{os.path.basename(full_path)}"',
                },
            )
        except ValueError:
            logger.warning("Invalid file path requested")
            raise HTTPException(status_code=400, detail="Invalid file path")
        except FileNotFoundError:
            logger.error("File not found")
            raise HTTPException(status_code=404, detail="File not found")
        except PermissionError:
            logger.error("Access forbidden")
            raise HTTPException(status_code=403, detail="Access forbidden")
