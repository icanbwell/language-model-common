"""MCP App UI resource detection and fetching.

Supports the MCP Apps spec where tools declare a ``ui://`` resource URI
in their metadata.  After a tool call, the UI resource can be fetched
and sent to the client as an HTML embed rendered in a sandboxed iframe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.types import Tool as MCPTool

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class McpAppEmbed:
    """An MCP app HTML embed ready to be sent to the client."""

    html: str
    title: str | None = None
    tool_name: str | None = None


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an MCP SDK model or dict to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return {}


def extract_ui_resource_uri(tool: MCPTool) -> str | None:
    """Extract a ``ui://`` resource URI from tool metadata, if present.

    Checks ``tool.meta.ui.resourceUri`` (MCP Apps spec) and the flat
    ``tool.meta["ui/resourceUri"]`` fallback.
    """
    meta = getattr(tool, "meta", None)
    if meta is None:
        return None

    meta_dict = _to_dict(meta)
    if not meta_dict:
        return None

    # Nested: meta.ui.resourceUri
    ui_meta = meta_dict.get("ui", {})
    if isinstance(ui_meta, dict):
        uri = ui_meta.get("resourceUri", "")
        if isinstance(uri, str) and uri.startswith("ui://"):
            return uri

    # Flat key fallback: meta["ui/resourceUri"]
    flat_uri = meta_dict.get("ui/resourceUri", "")
    if isinstance(flat_uri, str) and flat_uri.startswith("ui://"):
        return flat_uri

    return None


async def fetch_ui_resource(
    session: ClientSession,
    uri: str,
) -> str | None:
    """Fetch HTML content from a ``ui://`` resource URI.

    Returns the HTML string, or ``None`` if the resource is empty or
    the fetch fails.
    """
    try:
        result = await session.read_resource(uri)
    except Exception as e:
        logger.warning("Failed to fetch UI resource %s: %s", uri, e)
        return None

    if result and getattr(result, "contents", None):
        for item in result.contents:
            text = getattr(item, "text", None)
            if text:
                return text

    return None


def inject_tool_data_into_html(
    html: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result_text: str,
) -> str:
    """Inject MCP tool result data and an AppBridge shim into HTML.

    Adds:
    - ``window.__MCP_TOOL_RESULT__`` / ``__MCP_TOOL_ARGS__`` / ``__MCP_TOOL_NAME__``
    - An AppBridge-compatible shim that dispatches ``ui/notifications/tool-result``
    - An auto-height reporter via ``postMessage({type: 'iframe:height', height})``
    """
    import json

    data_script = (
        "<script>\n"
        f" window.__MCP_TOOL_RESULT__ = {json.dumps(tool_result_text)};\n"
        f" window.__MCP_TOOL_ARGS__ = {json.dumps(tool_args, ensure_ascii=False)};\n"
        f" window.__MCP_TOOL_NAME__ = {json.dumps(tool_name)};\n"
        "</script>\n"
    )

    appbridge_shim = (
        "<script>\n"
        "(function(){\n"
        f" var _result = {json.dumps(tool_result_text)};\n"
        " var _notification = {\n"
        "   jsonrpc: '2.0',\n"
        "   method: 'ui/notifications/tool-result',\n"
        "   params: { content: [{ type: 'text', text: _result }] }\n"
        " };\n"
        " try {\n"
        "   var _parsed = JSON.parse(_result);\n"
        "   if (_parsed && typeof _parsed === 'object')\n"
        "     _notification.params.structuredContent = _parsed;\n"
        " } catch(e) {}\n"
        " function _dispatch() {\n"
        "   window.dispatchEvent(new MessageEvent('message', {\n"
        "     data: _notification,\n"
        "     origin: window.location.origin,\n"
        "     source: window.parent\n"
        "   }));\n"
        " }\n"
        " if (document.readyState === 'complete' || document.readyState === 'interactive')\n"
        "   setTimeout(_dispatch, 50);\n"
        " else\n"
        "   window.addEventListener('DOMContentLoaded', function(){ setTimeout(_dispatch, 50); });\n"
        "})();\n"
        "</script>\n"
    )

    height_script = (
        "<script>\n"
        "function reportHeight(){\n"
        "  var h = document.documentElement.scrollHeight;\n"
        "  window.parent.postMessage({type:'iframe:height', height:h}, '*');\n"
        "}\n"
        "window.addEventListener('load', function(){ reportHeight(); setTimeout(reportHeight, 200); });\n"
        "new MutationObserver(reportHeight).observe(document.body, {childList:true, subtree:true});\n"
        "window.addEventListener('resize', reportHeight);\n"
        "</script>\n"
    )

    injection = data_script + appbridge_shim + height_script

    if "<head>" in html:
        return html.replace("<head>", "<head>\n" + injection, 1)
    if "<html>" in html:
        return html.replace("<html>", "<html>\n<head>" + injection + "</head>", 1)
    return "<head>" + injection + "</head>\n" + html
