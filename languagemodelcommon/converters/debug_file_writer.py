import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from languagemodelcommon.file_managers.file_manager_factory import FileManagerFactory
from languagemodelcommon.utilities.url_parser import UrlParser


@dataclass(frozen=True)
class DebugFileWriteResult:
    file_path: Optional[str]
    file_url: Optional[str]
    url_error_message: Optional[str]


class DebugFileWriter:
    def __init__(self, *, file_manager_factory: FileManagerFactory) -> None:
        self.file_manager_factory = file_manager_factory
        if self.file_manager_factory is None:
            raise ValueError("file_manager_factory must not be None")
        if not isinstance(self.file_manager_factory, FileManagerFactory):
            raise TypeError(
                "file_manager_factory must be an instance of FileManagerFactory"
            )

    @staticmethod
    def generate_secure_filename(
        *,
        tool_name: Optional[str],
        user_id: Optional[str],
    ) -> str:
        """
        Generate a secure, non-guessable filename for tool output files.

        Args:
            tool_name: The name of the tool that generated the output
            user_id: The user identifier (currently not embedded in the filename)

        Returns:
            A secure filename string
        """
        random_token = secrets.token_urlsafe(16)

        # We do not embed user identifiers in file names to avoid linkability.
        _ = user_id

        base_tool_name = tool_name or "unknown"
        safe_tool_name = re.sub(r"[^A-Za-z0-9._-]", "_", base_tool_name)
        safe_tool_name = re.sub(r"_+", "_", safe_tool_name).strip("_")
        if not safe_tool_name:
            safe_tool_name = "unknown"

        max_tool_name_length = 50
        safe_tool_name = safe_tool_name[:max_tool_name_length]

        timestamp = int(time.time())
        return f"{safe_tool_name}_{timestamp}_{random_token}.txt"

    async def write_content(
        self,
        *,
        content: str,
        output_folder: str,
        filename: str,
        content_type: str = "text/plain",
    ) -> DebugFileWriteResult:
        file_manager = self.file_manager_factory.get_file_manager(folder=output_folder)
        file_path: Optional[str] = await file_manager.save_file_async(
            file_data=content.encode("utf-8"),
            folder=output_folder,
            filename=filename,
            content_type=content_type,
        )
        if not file_path:
            return DebugFileWriteResult(
                file_path=None,
                file_url=None,
                url_error_message=None,
            )

        try:
            file_url = UrlParser.get_url_for_file_name(filename)
            if file_url is None:
                return DebugFileWriteResult(
                    file_path=file_path,
                    file_url=None,
                    url_error_message="Tool output file URL could not be generated.",
                )
            return DebugFileWriteResult(
                file_path=file_path,
                file_url=file_url,
                url_error_message=None,
            )
        except KeyError:
            return DebugFileWriteResult(
                file_path=file_path,
                file_url=None,
                url_error_message=(
                    "Tool output file URL could not be generated due to missing "
                    "IMAGE_GENERATION_URL environment variable."
                ),
            )
