from typing import Mapping

from languagemodelcommon.utilities.environment.language_model_common_environment_variables import (
    LanguageModelCommonEnvironmentVariables,
)
from languagemodelcommon.utilities.header_reader.header_reader import HeaderReader


class DebugConfiguration:
    @staticmethod
    def is_request_enabled_for_debug_logging(
        *,
        environment_variables: LanguageModelCommonEnvironmentVariables,
        headers: Mapping[str, str],
    ) -> bool:

        client_id: str | None = HeaderReader.get_client_id_from_headers(headers=headers)

        # if headers includes "Debug-Mode" with value "true", enable debug logging regardless of client ID allowlist
        if HeaderReader.has_debug_mode_enabled(headers=headers):
            return True

        if client_id is None:
            return False

        enable_debug_logging: bool = (
            client_id.lower()
            in [c.lower() for c in environment_variables.client_ids_for_debug_output]
            if client_id and environment_variables.client_ids_for_debug_output
            else False
        )
        return enable_debug_logging
