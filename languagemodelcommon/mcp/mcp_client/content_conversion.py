"""Content conversion between MCP and LangChain content block formats."""

from __future__ import annotations

from langchain_core.messages.content import (
    FileContentBlock,
    ImageContentBlock,
    TextContentBlock,
    create_file_block,
    create_image_block,
    create_text_block,
)
from langchain_core.tools import ToolException
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

ToolMessageContentBlock = TextContentBlock | ImageContentBlock | FileContentBlock


def convert_mcp_content_to_lc_block(
    content: ContentBlock,
) -> ToolMessageContentBlock:
    """Convert an MCP content block to a LangChain content block."""
    if isinstance(content, TextContent):
        return create_text_block(text=content.text)

    if isinstance(content, ImageContent):
        return create_image_block(base64=content.data, mime_type=content.mimeType)

    if isinstance(content, AudioContent):
        raise NotImplementedError(
            f"AudioContent conversion not supported. Mime type: {content.mimeType}"
        )

    if isinstance(content, ResourceLink):
        mime_type = content.mimeType or None
        if mime_type and mime_type.startswith("image/"):
            return create_image_block(url=str(content.uri), mime_type=mime_type)
        return create_file_block(url=str(content.uri), mime_type=mime_type)

    if isinstance(content, EmbeddedResource):
        resource = content.resource
        if isinstance(resource, TextResourceContents):
            return create_text_block(text=resource.text)
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mimeType or None
            if mime_type and mime_type.startswith("image/"):
                return create_image_block(base64=resource.blob, mime_type=mime_type)
            return create_file_block(base64=resource.blob, mime_type=mime_type)
        raise ValueError(f"Unknown embedded resource type: {type(resource).__name__}")

    raise ValueError(f"Unknown MCP content type: {type(content).__name__}")


def convert_call_tool_result(
    result: CallToolResult,
) -> list[ToolMessageContentBlock]:
    """Convert a CallToolResult to LangChain content blocks.

    Raises ToolException if the result indicates an error.
    """
    tool_content: list[ToolMessageContentBlock] = [
        convert_mcp_content_to_lc_block(c) for c in result.content
    ]

    if result.isError:
        error_parts = [
            block.text for block in result.content if isinstance(block, TextContent)
        ]
        raise ToolException(
            "\n".join(error_parts) if error_parts else str(tool_content)
        )

    return tool_content
