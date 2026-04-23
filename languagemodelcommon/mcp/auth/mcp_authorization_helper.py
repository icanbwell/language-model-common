import re

from httpx import Headers


class McpAuthorizationHelper:
    """
    Helper class for MCP authorization.
    """

    @staticmethod
    def extract_resource_metadata_from_www_auth(*, headers: Headers) -> str | None:
        """
        Extract protected resource metadata URL from WWW-Authenticate header as per RFC9728.

        Returns:
            Resource metadata URL if found in WWW-Authenticate header, None otherwise
        """
        www_auth_header = headers.get("WWW-Authenticate")
        if not www_auth_header:
            return None

        # Pattern matches: resource_metadata="url" or resource_metadata=url (unquoted)
        pattern = r'resource_metadata=(?:"([^"]+)"|([^\s,]+))'
        match = re.search(pattern, www_auth_header)

        if match:
            # Return quoted value if present, otherwise unquoted value
            return match.group(1) or match.group(2)

        return None

    @staticmethod
    def build_www_authenticate_login_message(
        *, resource_metadata_url: str | None, tool_url: str
    ) -> str:
        """Build a user-facing login message for MCP tools that return
        WWW-Authenticate headers (RFC 9728).

        This is the single place where WWW-Authenticate-based login
        messages are constructed.
        """
        if resource_metadata_url:
            return (
                f"Please [login]({resource_metadata_url}) to access "
                f"the MCP tool from {tool_url}."
            )
        return f"Authentication is required to access the MCP tool from {tool_url}."
