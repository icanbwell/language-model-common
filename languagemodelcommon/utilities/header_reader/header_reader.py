from typing import Mapping


class HeaderReader:
    @staticmethod
    def get_client_id_from_headers(headers: Mapping[str, str]) -> str | None:
        """
        Extract the client ID from HTTP headers.
        """
        lower_case_headers: Mapping[str, str] = {
            k.lower(): v for k, v in headers.items()
        }
        client_id: str | None = (
            lower_case_headers.get("bwell-managing-organization".lower())
            or lower_case_headers.get("x-client-id".lower())
            or lower_case_headers.get("client-id".lower())
        )
        return client_id

    @staticmethod
    def has_debug_mode_enabled(headers: Mapping[str, str]) -> bool:
        """
        Check if debug mode is enabled via the "Debug-Mode" header.

        Only explicit truthy values (e.g., "true" or "1", case-insensitive) are
        treated as enabling debug mode. Mere presence of the header without a
        supported truthy value does not enable debug behavior.
        """
        lower_case_headers: Mapping[str, str] = {
            k.lower(): v for k, v in headers.items()
        }
        debug_mode_header_value: str | None = lower_case_headers.get("debug-mode")
        if debug_mode_header_value is None:
            return False

        normalized_value: str = debug_mode_header_value.strip().lower()
        return normalized_value in {"true", "1"}
