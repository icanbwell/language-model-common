from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    AgentMiddleware,
    ExtendedModelResponse,
)
from langchain.messages import SystemMessage
from typing import Callable, Any, Awaitable, Sequence

from langchain_core.messages import AIMessage, AnyMessage

from languagemodelcommon.mcp.resource_catalog import ResourceCatalog
from languagemodelcommon.mcp.tool_catalog import ToolCatalog


class ToolDiscoveryMiddleware(AgentMiddleware):
    """Middleware that injects tool and resource discovery category descriptions into the system prompt.

    Tells the LLM what categories of tools and resources are available via
    the discovery system and instructs it to use search_tools / call_tool
    and search_resources / read_resource to find and access them.
    """

    _TOOLS_BLOCK_MARKER = "<available_tool_categories>"

    def __init__(
        self,
        catalog: ToolCatalog,
        resource_catalog: ResourceCatalog | None = None,
    ) -> None:
        if catalog is None:
            raise ValueError("catalog must not be None")
        self._catalog = catalog
        self._resource_catalog = resource_catalog

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Inject tool and resource category descriptions into the system prompt."""
        existing_messages: list[AnyMessage] = list(request.messages or ())

        if self._request_has_tools_message(existing_messages):
            return await handler(request)

        tool_categories = self._catalog.get_categories()
        resource_categories = (
            self._resource_catalog.get_categories()
            if self._resource_catalog is not None
            else []
        )

        if not tool_categories and not resource_categories:
            return await handler(request)

        prompt_text = self._build_discovery_prompt(
            tool_categories=tool_categories,
            resource_categories=resource_categories,
        )
        discovery_message = SystemMessage(content=prompt_text)

        insertion_index = 0
        for idx, message in enumerate(existing_messages):
            if isinstance(message, SystemMessage):
                insertion_index = idx + 1
                break

        existing_messages.insert(insertion_index, discovery_message)
        modified_request = request.override(messages=list(existing_messages))
        return await handler(modified_request)

    @classmethod
    def _build_discovery_prompt(
        cls,
        *,
        tool_categories: list[dict[str, Any]],
        resource_categories: list[dict[str, Any]],
    ) -> str:
        parts: list[str] = []

        if tool_categories:
            category_lines = "\n".join(
                f"  <category>\n"
                f"    <name>{c['name']}</name>\n"
                f"    <description>{c['description']}</description>\n"
                f"  </category>"
                for c in tool_categories
            )

            parts.append(
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

        if resource_categories:
            resource_lines = "\n".join(
                f"  <category>\n"
                f"    <name>{c['name']}</name>\n"
                f"    <description>{c['description']}</description>\n"
                f"  </category>"
                for c in resource_categories
            )

            parts.append(
                "\n\nYou also have access to a resource discovery system with the "
                "following categories of resources:\n\n"
                "<available_resource_categories>\n"
                f"{resource_lines}\n"
                "</available_resource_categories>\n\n"
                "When you need data or content from one of these categories:\n"
                '1. Call search_resources(query="your search query") to discover '
                "available resources. You can optionally filter by category.\n"
                "2. Review the returned resource names, URIs, descriptions, and MIME types.\n"
                '3. Call read_resource(uri="resource_uri") to read the resource content.\n\n'
                "Always search for resources before trying to read them directly."
            )

        return "\n".join(parts)

    # Keep backward compatibility
    @classmethod
    def _build_tools_prompt(cls, categories: list[dict[str, Any]]) -> str:
        return cls._build_discovery_prompt(
            tool_categories=categories, resource_categories=[]
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
