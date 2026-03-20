import asyncio
import logging

import httpx
import json
import os
import tempfile
import zipfile

from typing import List, Optional

from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig
from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS["CONFIG"])


def _mask_token(token: Optional[str]) -> str:
    """Mask sensitive token for logging"""
    if not token:
        return "[REDACTED]"
    if len(token) > 8:
        return f"{token[:4]}...{token[-4:]}"
    return "[REDACTED]"


class GitHubConfigZipDownloader:
    def __init__(
        self,
        github_token: Optional[str] = None,
        max_retries: int = 3,
        base_delay: int = 1,
    ) -> None:
        """
        Initialize GitHub configuration downloader
        Args:
            github_token: Optional GitHub API token
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
        """
        self.github_token: Optional[str] = github_token or os.environ.get(
            "GITHUB_TOKEN"
        )
        self.max_retries: int = max_retries
        self.base_delay: int = base_delay
        self.timeout: int = int(os.environ.get("GITHUB_TIMEOUT", 3600))

    async def download_zip(
        self, zip_url: str, target_path: Optional[str] = None
    ) -> str:
        """
        Download ZIP file from given URL
        Args:
            zip_url: Full URL to the ZIP file
            target_path: Optional target directory for extraction
        Returns:
            Path to the extracted repository
        """
        # Create a temporary directory if no target path is provided
        if target_path is None:
            target_path = tempfile.mkdtemp(prefix="github_config_")

        # Ensure target path exists
        os.makedirs(target_path, exist_ok=True)

        async def download_with_retry(url: str) -> bytes:
            """
            Download with exponential backoff and retry logic
            Args:
                url: Download URL
            Returns:
                Downloaded content as bytes
            """
            headers = {}
            if self.github_token:
                headers["Authorization"] = f"Bearer {self.github_token}"

            headers["X-GitHub-Api-Version"] = "2022-11-28"
            headers["Accept"] = "application/vnd.github+json"

            for attempt in range(self.max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            url,
                            headers=headers,
                            follow_redirects=True,
                            timeout=httpx.Timeout(self.timeout),
                        )
                        response.raise_for_status()
                        return response.content
                except Exception as e1:
                    logger.error(
                        "Download attempt %s failed URL: %s, token: %s %s: %s",
                        attempt + 1,
                        url,
                        _mask_token(self.github_token),
                        type(e1),
                        e1,
                    )

                    # Exponential backoff
                    await asyncio.sleep(self.base_delay * (2**attempt))

            raise RuntimeError(
                "Failed to download ZIP after %s attempts URL: %s, token: %s"
                % (self.max_retries, url, _mask_token(self.github_token))
            )

        try:
            # Download ZIP archive
            logger.info("Downloading ZIP from: %s", zip_url)
            zip_content = await download_with_retry(zip_url)
            logger.info("Downloaded ZIP from %s", zip_url)

            # Create a temporary file to save the ZIP
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
                temp_zip.write(zip_content)
                temp_zip_path = temp_zip.name

            # Extract ZIP archive
            logger.info("Extracting ZIP to: %s", target_path)
            with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                # Validate all members for path traversal before extraction
                safe_target = os.path.abspath(target_path)
                for member in zip_ref.namelist():
                    member_path = os.path.abspath(os.path.join(target_path, member))
                    # Ensure member path is within target directory
                    if not member_path.startswith(safe_target + os.sep):
                        if member_path != safe_target:
                            raise ValueError(
                                f"ZIP member {member} attempts path traversal outside target directory"
                            )
                # List all contents to find the root directory
                all_contents = zip_ref.namelist()
                root_dir = all_contents[0].split("/")[0] if all_contents else None

                if not root_dir:
                    raise ValueError("Could not find root directory in ZIP archive")

                # Extract all contents
                zip_ref.extractall(path=target_path)

            # Remove temporary ZIP file
            os.unlink(temp_zip_path)

            # Return the full path to the extracted repository
            extracted_path = os.path.join(target_path, root_dir)
            return extracted_path

        except Exception as e:
            logger.error("Error downloading ZIP: %s", e)
            raise

    @staticmethod
    def _find_json_configs(
        repo_path: str, config_dir: Optional[str] = None
    ) -> List[ChatModelConfig]:
        """
        Find and parse JSON configuration files in the repository
        Args:
            repo_path: Path to the extracted repository
            config_dir: Optional subdirectory to search for configs
        Returns:
            List of parsed JSON configurations
        """
        configs: List[ChatModelConfig] = []

        repo_root = os.path.realpath(repo_path)
        search_path = (
            os.path.realpath(os.path.join(repo_root, config_dir))
            if config_dir
            else repo_root
        )

        try:
            if os.path.commonpath([repo_root, search_path]) != repo_root:
                logger.warning("Skipping config directory outside repository root")
                return configs
        except ValueError:
            logger.warning("Skipping invalid config directory path")
            return configs

        if not os.path.isdir(search_path):
            return configs

        # Walk through directory
        for root, _, files in os.walk(search_path):
            for file in files:
                if file.endswith(".json"):
                    try:
                        file_path = os.path.join(root, file)
                        resolved_file_path = os.path.realpath(file_path)
                        if (
                            os.path.commonpath([search_path, resolved_file_path])
                            != search_path
                        ):
                            logger.warning(
                                "Skipping config file outside allowed directory"
                            )
                            continue
                        with open(resolved_file_path, "r", encoding="utf-8") as f:
                            config = substitute_env_vars(json.load(f))
                            configs.append(ChatModelConfig(**config))
                    except json.JSONDecodeError as e:
                        logger.error("Error parsing JSON from %s: %s", file, e)
                    except Exception as e:
                        logger.error("Unexpected error processing %s: %s", file, e)

        # sort the configs by name
        configs.sort(key=lambda x: x.name)

        return configs

    async def read_model_configs(
        self,
        *,
        github_url: str,
        models_official_path: str,
        models_testing_path: Optional[str],
    ) -> List[ChatModelConfig]:
        """
        Comprehensive method to download ZIP and extract configs
        Returns:
            List of model configurations
        """
        try:
            # Download and extract ZIP
            repo_path: str = await self.download_zip(zip_url=github_url)

            # Find and parse JSON configs
            configs: List[ChatModelConfig] = self._find_json_configs(
                repo_path=repo_path, config_dir=models_official_path
            )

            if models_testing_path:
                test_configs: List[ChatModelConfig] = self._find_json_configs(
                    repo_path=repo_path, config_dir="configs/chat_completions/testing"
                )

                if test_configs and len(test_configs) > 0:
                    configs.append(
                        ChatModelConfig(
                            id="testing",
                            name="----- Models in Testing -----",
                            description="",
                        )
                    )
                    configs.extend(test_configs)

            return configs

        except Exception as e:
            logger.error("Error retrieving model configs: %s", e)
            return []
