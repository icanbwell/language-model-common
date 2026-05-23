"""MCP App UI resource detection and fetching.

Supports the MCP Apps spec where tools declare a ``ui://`` resource URI
in their metadata.  After a tool call, the UI resource can be fetched
and sent to the client as an HTML embed rendered in a sandboxed iframe.
"""

import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.types import Tool as MCPTool
from pydantic import AnyUrl

from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.MCP)


@dataclass
class McpAppUiMeta:
    """CSP, permissions, and display metadata from a UI resource response."""

    csp: dict[str, Any] | None = None
    permissions: dict[str, Any] | None = None
    prefers_border: bool | None = None
    display_mode: str | None = None


@dataclass
class McpAppEmbed:
    """An MCP app HTML embed ready to be sent to the client."""

    html: str
    title: str | None = None
    tool_name: str | None = None
    ui_meta: McpAppUiMeta | None = None


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an MCP SDK model or dict to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump(mode="json"))
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


def is_tool_visible_to_model(tool: MCPTool) -> bool:
    """Check whether a tool should be exposed to the LLM.

    Per MCP Apps spec, tools with ``_meta.ui.visibility: ["app"]`` are
    only callable by the rendered UI iframe, not by the model.
    Default visibility (when unspecified) is ``["model", "app"]``.
    """
    meta = getattr(tool, "meta", None)
    if meta is None:
        return True

    meta_dict = _to_dict(meta)
    if not meta_dict:
        return True

    ui_meta = meta_dict.get("ui", {})
    if not isinstance(ui_meta, dict):
        return True

    visibility = ui_meta.get("visibility")
    if visibility is None:
        return True

    if isinstance(visibility, list):
        return "model" in visibility

    return True


@dataclass
class UiResourceFetchResult:
    """Result of fetching a UI resource, including HTML and metadata."""

    html: str
    ui_meta: McpAppUiMeta | None = None


async def fetch_ui_resource(
    session: ClientSession,
    uri: str,
) -> UiResourceFetchResult | None:
    """Fetch HTML content and metadata from a ``ui://`` resource URI.

    Returns a ``UiResourceFetchResult`` with HTML and any CSP/permissions
    metadata, or ``None`` if the resource is empty or the fetch fails.
    """
    try:
        result = await session.read_resource(AnyUrl(uri))
    except Exception as e:
        logger.warning("Failed to fetch UI resource %s: %s", uri, e)
        return None

    if result and getattr(result, "contents", None):
        for item in result.contents:
            text = getattr(item, "text", None)
            if not text:
                continue

            ui_meta = _extract_ui_meta_from_content(item)
            return UiResourceFetchResult(html=str(text), ui_meta=ui_meta)

    return None


def _extract_ui_meta_from_content(content_item: Any) -> McpAppUiMeta | None:
    """Extract CSP and permissions from a resource content item's _meta.ui."""
    meta = getattr(content_item, "_meta", None) or getattr(content_item, "meta", None)
    if meta is None:
        return None

    meta_dict = _to_dict(meta)
    ui = meta_dict.get("ui")
    if not isinstance(ui, dict):
        return None

    csp = ui.get("csp")
    permissions = ui.get("permissions")
    prefers_border = ui.get("prefersBorder")
    display_mode = ui.get("displayMode")

    if (
        csp is None
        and permissions is None
        and prefers_border is None
        and display_mode is None
    ):
        return None

    return McpAppUiMeta(
        csp=csp if isinstance(csp, dict) else None,
        permissions=permissions if isinstance(permissions, dict) else None,
        prefers_border=bool(prefers_border) if prefers_border is not None else None,
        display_mode=str(display_mode) if display_mode else None,
    )


def inject_tool_data_into_html(
    html: str,
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result_text: str,
    proxy_base_url: str | None = None,
    session_token: str | None = None,
) -> str:
    """Inject a spec-compliant MCP Apps bridge into HTML.

    Implements the MCP Apps ``ui/initialize`` handshake protocol.  The injected
    bridge intercepts JSON-RPC ``postMessage`` calls the app sends to
    ``window.parent`` and handles them according to the spec:

    Phase 2 — Bidirectional communication:
      - Responds to ``ui/initialize`` with host capabilities and context
      - Sends ``ui/notifications/tool-input`` (complete tool args)
      - Sends ``ui/notifications/tool-result`` (tool result text)
      - Handles ``ui/open-link`` (opens URL in new tab)
      - Handles ``ui/message`` (forwards to real parent for chat injection)
      - Handles ``ui/request-display-mode``

    Phase 3 — Proxy:
      - Proxies ``tools/call`` and ``resources/read`` via fetch to backend
      - Reports container dimensions and display mode

    Also injects legacy globals (``__MCP_TOOL_RESULT__``, etc.) for backwards
    compatibility with apps that don't use the JSON-RPC protocol.
    """
    import json

    bridge_config = json.dumps(
        {
            "toolName": tool_name,
            "toolArgs": tool_args,
            "toolResult": tool_result_text,
            "proxyBaseUrl": proxy_base_url,
            "sessionToken": session_token,
        },
        ensure_ascii=False,
    )

    bridge_script = (
        "<script>\n"
        f"window.__MCP_TOOL_RESULT__ = {json.dumps(tool_result_text)};\n"
        f"window.__MCP_TOOL_ARGS__ = {json.dumps(tool_args, ensure_ascii=False)};\n"
        f"window.__MCP_TOOL_NAME__ = {json.dumps(tool_name)};\n"
        f"window.__MCP_BRIDGE_CONFIG__ = {bridge_config};\n"
        "</script>\n"
        "<script>\n" + _MCP_APP_BRIDGE_JS + "\n</script>\n"
    )

    if "<head>" in html:
        return html.replace("<head>", "<head>\n" + bridge_script, 1)
    if "<html>" in html:
        return html.replace("<html>", "<html>\n<head>" + bridge_script + "</head>", 1)
    return "<head>" + bridge_script + "</head>\n" + html


_MCP_APP_BRIDGE_JS = """\
(function() {
  'use strict';
  var cfg = window.__MCP_BRIDGE_CONFIG__ || {};
  var toolName = cfg.toolName || '';
  var toolArgs = cfg.toolArgs || {};
  var toolResult = cfg.toolResult || '';
  var proxyBaseUrl = cfg.proxyBaseUrl || null;
  var sessionToken = cfg.sessionToken || null;

  var _realParentPostMessage = window.parent !== window
    ? window.parent.postMessage.bind(window.parent)
    : function() {};

  var _pendingRequests = {};
  var _nextId = 1;
  var _initialized = false;
  var _displayMode = 'inline';

  function _sendToApp(msg) {
    window.dispatchEvent(new MessageEvent('message', {
      data: msg,
      origin: window.location.origin,
      source: window.parent
    }));
  }

  function _respondToApp(id, result) {
    _sendToApp({ jsonrpc: '2.0', id: id, result: result });
  }

  function _errorToApp(id, code, message) {
    _sendToApp({ jsonrpc: '2.0', id: id, error: { code: code, message: message } });
  }

  function _notifyApp(method, params) {
    _sendToApp({ jsonrpc: '2.0', method: method, params: params || {} });
  }

  function _reportHeight() {
    var h = document.documentElement.scrollHeight;
    _realParentPostMessage({ type: 'iframe:height', height: h }, '*');
  }

  function _getContainerDimensions() {
    return {
      width: window.innerWidth,
      height: window.innerHeight,
      maxHeight: window.screen.availHeight
    };
  }

  function _handleInitialize(id) {
    var hostCapabilities = {
      ui: { openLink: true, message: true, requestDisplayMode: true },
      notifications: { toolInput: true, toolResult: true },
      proxy: { toolsCall: !!proxyBaseUrl, resourcesRead: !!proxyBaseUrl }
    };
    var hostContext = {
      containerDimensions: _getContainerDimensions(),
      displayMode: _displayMode
    };
    _respondToApp(id, {
      protocolVersion: '2025-06-18',
      hostCapabilities: hostCapabilities,
      hostContext: hostContext
    });
    _initialized = true;

    setTimeout(function() {
      _notifyApp('ui/notifications/tool-input', {
        toolName: toolName,
        arguments: toolArgs
      });

      var resultContent = [{ type: 'text', text: toolResult }];
      var structuredContent = null;
      try {
        var parsed = JSON.parse(toolResult);
        if (parsed && typeof parsed === 'object') structuredContent = parsed;
      } catch(e) {}
      var resultParams = { content: resultContent };
      if (structuredContent) resultParams.structuredContent = structuredContent;
      _notifyApp('ui/notifications/tool-result', resultParams);
    }, 0);
  }

  function _handleOpenLink(id, params) {
    var url = params && params.url;
    if (!url) {
      _errorToApp(id, -32602, 'Missing url parameter');
      return;
    }
    try {
      window.open(url, '_blank', 'noopener,noreferrer');
      _respondToApp(id, {});
    } catch(e) {
      _errorToApp(id, -32000, 'Failed to open link: ' + e.message);
    }
  }

  function _handleMessage(id, params) {
    var message = params && params.message;
    if (!message) {
      _errorToApp(id, -32602, 'Missing message parameter');
      return;
    }
    _realParentPostMessage({
      type: 'mcp:ui:message',
      message: message,
      toolName: toolName
    }, '*');
    _respondToApp(id, {});
  }

  function _handleRequestDisplayMode(id, params) {
    var mode = params && params.mode;
    if (!mode || (mode !== 'inline' && mode !== 'fullscreen' && mode !== 'pip')) {
      _errorToApp(id, -32602, 'Invalid display mode');
      return;
    }
    _displayMode = mode;
    _realParentPostMessage({
      type: 'mcp:ui:display-mode',
      mode: mode,
      toolName: toolName
    }, '*');
    _respondToApp(id, { displayMode: mode });
  }

  async function _handleToolsCall(id, params) {
    if (!proxyBaseUrl) {
      _errorToApp(id, -32601, 'Proxy not available');
      return;
    }
    try {
      var resp = await fetch(proxyBaseUrl + '/mcp-proxy/tools/call', {
        method: 'POST',
        headers: _proxyHeaders(),
        body: JSON.stringify({ name: params.name, arguments: params.arguments || {} })
      });
      if (!resp.ok) {
        _errorToApp(id, -32000, 'Proxy error: HTTP ' + resp.status);
        return;
      }
      var result = await resp.json();
      _respondToApp(id, result);
    } catch(e) {
      _errorToApp(id, -32000, 'Proxy fetch failed: ' + e.message);
    }
  }

  async function _handleResourcesRead(id, params) {
    if (!proxyBaseUrl) {
      _errorToApp(id, -32601, 'Proxy not available');
      return;
    }
    try {
      var resp = await fetch(proxyBaseUrl + '/mcp-proxy/resources/read', {
        method: 'POST',
        headers: _proxyHeaders(),
        body: JSON.stringify({ uri: params.uri })
      });
      if (!resp.ok) {
        _errorToApp(id, -32000, 'Proxy error: HTTP ' + resp.status);
        return;
      }
      var result = await resp.json();
      _respondToApp(id, result);
    } catch(e) {
      _errorToApp(id, -32000, 'Proxy fetch failed: ' + e.message);
    }
  }

  function _proxyHeaders() {
    var h = { 'Content-Type': 'application/json' };
    if (sessionToken) h['Authorization'] = 'Bearer ' + sessionToken;
    return h;
  }

  function _handleJsonRpcMessage(msg) {
    if (!msg || typeof msg !== 'object' || msg.jsonrpc !== '2.0') return false;
    if (!msg.method) return false;

    var method = msg.method;
    var id = msg.id;
    var params = msg.params || {};

    switch (method) {
      case 'ui/initialize':
        _handleInitialize(id);
        break;
      case 'ui/notifications/initialized':
        break;
      case 'ui/open-link':
        _handleOpenLink(id, params);
        break;
      case 'ui/message':
        _handleMessage(id, params);
        break;
      case 'ui/request-display-mode':
        _handleRequestDisplayMode(id, params);
        break;
      case 'tools/call':
        _handleToolsCall(id, params);
        break;
      case 'resources/read':
        _handleResourcesRead(id, params);
        break;
      default:
        if (id !== undefined) {
          _errorToApp(id, -32601, 'Method not found: ' + method);
        }
    }
    return true;
  }

  var _origPostMessage = window.parent.postMessage;
  if (window.parent !== window) {
    try {
      Object.defineProperty(window, '__mcpParent', {
        value: new Proxy(window.parent, {
          get: function(target, prop) {
            if (prop === 'postMessage') {
              return function(msg, origin) {
                if (msg && typeof msg === 'object' && msg.jsonrpc === '2.0' && msg.method) {
                  _handleJsonRpcMessage(msg);
                } else {
                  _realParentPostMessage(msg, origin || '*');
                }
              };
            }
            return target[prop];
          }
        })
      });

      Object.defineProperty(window, 'parent', {
        get: function() { return window.__mcpParent; },
        configurable: true
      });
    } catch(e) {
      // Fallback: intercept via message event listener for apps that
      // use window.addEventListener('message', ...) directly
    }
  }

  // Height reporter
  function _setupHeightReporter() {
    window.addEventListener('load', function() {
      _reportHeight();
      setTimeout(_reportHeight, 200);
    });
    if (document.body) {
      new MutationObserver(_reportHeight).observe(document.body, {
        childList: true, subtree: true
      });
    } else {
      document.addEventListener('DOMContentLoaded', function() {
        new MutationObserver(_reportHeight).observe(document.body, {
          childList: true, subtree: true
        });
      });
    }
    window.addEventListener('resize', function() {
      _reportHeight();
      if (_initialized) {
        _notifyApp('ui/notifications/container-resized', {
          containerDimensions: _getContainerDimensions()
        });
      }
    });
  }
  _setupHeightReporter();

  // Legacy fallback: if the app doesn't send ui/initialize within 500ms,
  // dispatch tool-result notification directly (backwards compat)
  setTimeout(function() {
    if (!_initialized) {
      var resultContent = [{ type: 'text', text: toolResult }];
      var notification = {
        jsonrpc: '2.0',
        method: 'ui/notifications/tool-result',
        params: { content: resultContent }
      };
      try {
        var parsed = JSON.parse(toolResult);
        if (parsed && typeof parsed === 'object')
          notification.params.structuredContent = parsed;
      } catch(e) {}
      _sendToApp(notification);
    }
  }, 500);
})();
"""
