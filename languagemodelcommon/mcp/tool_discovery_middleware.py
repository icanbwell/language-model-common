from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    ExtendedModelResponse,
)
from langchain.messages import SystemMessage
from typing import Callable, Any, Awaitable, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from languagemodelcommon.mcp.tool_catalog import ToolCatalog


class ToolDiscoveryMiddleware(AgentMiddleware):
    """Middleware that injects tool discovery category descriptions into the system prompt.

    Analogous to SkillMiddleware but for MCP tool categories. Tells the LLM
    what categories of tools are available via the discovery system and
    instructs it to use search_tools / call_tool to find and invoke them.
    """

    _TOOLS_BLOCK_MARKER = "<available_tool_categories>"

    def __init__(self, catalog: ToolCatalog) -> None:
        if catalog is None:
            raise ValueError("catalog must not be None")
        self._catalog = catalog

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Inject tool category descriptions into the system prompt."""
        existing_messages: list[AnyMessage] = list(request.messages or ())

        if self._request_has_tools_message(existing_messages):
            return await handler(request)

        categories = self._catalog.get_categories()
        if not categories:
            return await handler(request)

        tools_block_text = self._build_tools_prompt(categories)
        tools_message = SystemMessage(content=tools_block_text)

        insertion_index = 0
        for idx, message in enumerate(existing_messages):
            if isinstance(message, SystemMessage):
                insertion_index = idx + 1
                break

        existing_messages.insert(insertion_index, tools_message)
        modified_request = request.override(messages=list(existing_messages))
        return await handler(modified_request)

    @classmethod
    def _build_tools_prompt(cls, categories: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in categories:
            auth_tag = ""
            if c.get("requires_auth"):
                auth_tag = "\n    <requires_auth>true</requires_auth>"
            parts.append(
                f"  <category>\n"
                f"    <name>{c['name']}</name>\n"
                f"    <description>{c['description']}</description>"
                f"{auth_tag}\n"
                f"  </category>"
            )
        category_lines = "\n".join(parts)

        return (
            "You have access to a tool discovery system with the following "
            "categories of tools:\n\n"
            f"{cls._TOOLS_BLOCK_MARKER}\n"
            f"{category_lines}\n"
            "</available_tool_categories>\n\n"
            "When a task requires tools from one of these categories:\n"
            '1. Call search_tools(query="your search query") to discover '
            "available tools. You can optionally filter by category.\n"
            "2. Review the returned tool names, descriptions, and parameter schemas.\n"
            '3. Call call_tool(name="tool_name", arguments={...}) to invoke '
            "the tool.\n\n"
            "Always search for tools before trying to call them directly."
        )

    @classmethod
    def _request_has_tools_message(cls, messages: Sequence[AnyMessage]) -> bool:
        for message in messages:
            if not isinstance(message, SystemMessage):
                continue
            if cls._content_contains_tools_marker(message.content):
                return True
        return False

    @classmethod
    def _content_contains_tools_marker(cls, content: object) -> bool:
        if isinstance(content, str):
            return cls._TOOLS_BLOCK_MARKER in content
        if isinstance(content, (list, tuple)):
            return any(cls._content_contains_tools_marker(item) for item in content)
        if isinstance(content, dict):
            return any(
                cls._content_contains_tools_marker(item) for item in content.values()
            )
        return False
