from languagemodelcommon.mcp.exceptions.mcp_tool_exception import (
    McpToolException,
)


class McpToolNotFoundException(McpToolException):
    """
    Exception raised when a tool is not found.
    """

    pass
