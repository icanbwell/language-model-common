# MCP Apps Support

This document describes the MCP Apps integration in `languagemodelcommon`, which allows MCP tools to declare interactive HTML UIs that are rendered in the client (OpenWebUI) as sandboxed iframes.

---

## Overview

The [MCP Apps spec](https://modelcontextprotocol.io) allows MCP tools to declare a `ui://` resource URI in their metadata. After a tool call completes, the framework fetches the HTML from this resource, injects the tool result data, and streams it to the client as a custom SSE event. The client then renders the HTML in a sandboxed iframe.

**Data flow:**

```
MCP Server (tool with ui:// resource)
    → CallToolTool._arun()
        → MCPToolProvider.execute_mcp_tool()        # call the tool
        → MCPToolProvider.fetch_mcp_app_embed()     # fetch ui:// resource HTML
            → inject_tool_data_into_html()          # inject result + AppBridge shim
    → ToolMessage.artifact = {"mcp_app_embed": McpAppEmbed(...)}
        → LangGraphStreamingManager._handle_on_tool_end()
            → chat_request_wrapper.create_mcp_app_sse_event()
                → SSE: "event: mcp_app\ndata: {"html": "...", "title": "..."}\n\n"
    → Pipe (language-model-gateway) parses named SSE event
        → __event_emitter__({"type": "embeds", "data": {"embeds": [html]}})
    → OpenWebUI renders in sandboxed iframe via FullHeightIframe.svelte
```

---

## MCP Server Requirements

An MCP tool declares a UI resource by setting `meta.ui.resourceUri` in its tool definition:

```json
{
  "name": "my_tool",
  "description": "A tool with a UI",
  "inputSchema": { ... },
  "meta": {
    "ui": {
      "resourceUri": "ui://my-server/my-tool-ui"
    }
  }
}
```

The server must also implement `resources/read` for the declared `ui://` URI, returning HTML content.

**Alternative flat key format** (also supported):

```json
{
  "meta": {
    "ui/resourceUri": "ui://my-server/my-tool-ui"
  }
}
```

---

## Architecture Details

### 1. UI Resource Detection and Fetching (`mcp_client/ui_resource.py`)

| Function | Purpose |
|---|---|
| `extract_ui_resource_uri(tool)` | Extracts the `ui://` URI from tool metadata (nested or flat format) |
| `fetch_ui_resource(session, uri)` | Calls `session.read_resource(uri)` to get the HTML content |
| `inject_tool_data_into_html(html, tool_name, tool_args, tool_result_text)` | Injects JavaScript globals and AppBridge shim into the HTML |
| `McpAppEmbed` | Dataclass holding `html`, `title`, and `tool_name` |

### 2. Tool Provider Integration (`mcp/mcp_tool_provider.py`)

`MCPToolProvider.fetch_mcp_app_embed()` orchestrates the fetch:

1. Calls `extract_ui_resource_uri(tool)` — returns `None` if no UI declared
2. Gets a session from the pool (or creates a one-shot session)
3. Calls `fetch_ui_resource(session, uri)` to get raw HTML
4. Calls `inject_tool_data_into_html()` to add tool result data
5. Returns `McpAppEmbed(html=..., title=..., tool_name=...)`

This is **best-effort** — exceptions are caught and logged, returning `None` so the tool call itself is never disrupted by a UI fetch failure.

### 3. CallToolTool Artifact (`tools/mcp/call_tool_tool.py`)

`CallToolTool` uses LangChain's `content_and_artifact` response format. After executing the tool:

```python
result = await self.mcp_tool_provider.execute_mcp_tool(...)
app_embed = await self.mcp_tool_provider.fetch_mcp_app_embed(...)

artifact = {"mcp_app_embed": app_embed} if app_embed else None
return text, artifact
```

The artifact is stored in `ToolMessage.artifact` and carried through the LangGraph event stream.

### 4. SSE Emission (`converters/streaming_manager.py`)

In `_handle_on_tool_end`, the streaming manager checks:

```python
if isinstance(artifact, dict) and "mcp_app_embed" in artifact:
    mcp_app_event = chat_request_wrapper.create_mcp_app_sse_event(
        html=artifact["mcp_app_embed"].html,
        title=artifact["mcp_app_embed"].title,
    )
    if mcp_app_event:
        yield mcp_app_event
```

### 5. SSE Format (`structures/openai/request/`)

Both `ChatCompletionApiRequestWrapper` and `ResponsesApiRequestWrapper` emit the same custom SSE event:

```
event: mcp_app
data: {"html": "<html>...</html>", "title": "My App"}

```

This uses the named-event SSE format (`event:` line + `data:` line), which is distinct from the standard `data:` only chunks used for text deltas.

### 6. Session Pool Reuse

`fetch_mcp_app_embed` accepts an optional `McpSessionPool`. When provided (the normal case during agent execution), the UI resource fetch reuses the same pooled TCP+TLS connection that was used for the tool call itself — no extra connection overhead.

---

## Injected JavaScript

The `inject_tool_data_into_html()` function adds three script blocks to the HTML `<head>`:

### Data Globals

```javascript
window.__MCP_TOOL_RESULT__ = "...";   // Tool result text
window.__MCP_TOOL_ARGS__ = {...};      // Tool arguments object
window.__MCP_TOOL_NAME__ = "...";      // Tool name
```

### AppBridge Shim

Dispatches a `ui/notifications/tool-result` MessageEvent on `window` after DOM ready, compatible with MCP Apps SDK clients:

```javascript
window.dispatchEvent(new MessageEvent('message', {
    data: {
        jsonrpc: '2.0',
        method: 'ui/notifications/tool-result',
        params: { content: [{ type: 'text', text: result }] }
    }
}));
```

If the result is valid JSON, `params.structuredContent` is also populated.

### Auto-Height Reporter

Reports iframe content height to the parent via `postMessage`, enabling the client to dynamically resize the iframe:

```javascript
window.parent.postMessage({type: 'iframe:height', height: scrollHeight}, '*');
```

---

## Client-Side Integration (OpenWebUI Pipe)

The pipe in `language-model-gateway` (`language_model_gateway_pipe.py`) handles the `event: mcp_app` SSE event:

1. Parses named SSE events using a spec-compliant SSE parser
2. On `mcp_app` event, extracts `html` from the JSON payload
3. Emits to OpenWebUI via `__event_emitter__`:

```python
await self.__event_emitter__({
    "type": "embeds",
    "data": {"embeds": [html]}
})
```

OpenWebUI renders each embed in a sandboxed `<iframe>` via `FullHeightIframe.svelte`, which listens for `iframe:height` postMessages to auto-size.

---

## Configuration

| Valve / Environment Variable | Description | Default |
|---|---|---|
| `mcp_app_event_name` (pipe valve) | SSE event name to listen for | `mcp_app` |

No additional configuration is needed on the `language-model-common` side. MCP Apps are automatically detected from tool metadata when present.

---

## Error Handling

- If the MCP server doesn't implement `resources/read` for the URI, the fetch fails silently (logged as warning)
- If the HTML is empty, no embed is emitted
- If the session pool is unavailable, a one-shot session is created as fallback
- Tool execution is never blocked or failed by UI resource fetch errors
