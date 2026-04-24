# MCP Tool Discovery

This document describes how the language-model-gateway discovers, registers, and exposes MCP tools to the LLM at runtime.

---

## Overview

MCP server definitions live in `.mcp.json` files inside marketplace plugin directories. At startup, the gateway scans these files, merges server entries, and resolves them onto model configs. At request time, the LLM discovers tools lazily through two meta-tools (`search_tools` and `call_tool`) rather than loading all tools upfront.

**Data flow:**

```
marketplace/plugins/*/.mcp.json           ← MCP server definitions
    → McpJsonReader.read_mcp_json()       ← scan & merge
    → resolve_mcp_servers()               ← expand wildcards, populate AgentConfig
    → ChatModelConfig.tools               ← resolved model configs (cached)

[per chat request]
    → LangChainCompletionsProvider         ← builds tool list
    → ToolCatalog + SearchToolsTool + CallToolTool
    → LangGraph agent with ToolDiscoveryMiddleware

[per LLM turn]
    → search_tools(category, query)        ← lazy connect + BM25 search
    → call_tool(name, arguments)           ← pooled MCP session + interceptors
```

---

## Phase 1: Configuration Loading

### 1.1 McpJsonReader scans marketplace plugins

`languagemodelcommon/configs/config_reader/mcp_json_reader.py`

`McpJsonReader.read_mcp_json()` reads the `PLUGINS_MARKETPLACE` environment variable and globs `{PLUGINS_MARKETPLACE}/plugins/*/.mcp.json`. Each file is parsed, environment variable substitution is applied (e.g. `${GROUNDCOVER_API_KEY}`), and all `mcpServers` entries are merged into a single `McpJsonConfig`.

```
marketplace/
  plugins/
    all-employees/
      .mcp.json          ← atlassian, github, skills, google-drive, ...
    team-specific/
      .mcp.json          ← team-only servers
```

If multiple plugins define the same server key, later files (sorted alphabetically by plugin name) overwrite earlier ones.

### 1.2 FileConfigReader loads model configs

`languagemodelcommon/configs/config_reader/file_config_reader.py`

`FileConfigReader.read_model_configs()` reads all `.json` files recursively from `MODELS_OFFICIAL_PATH`, parses them into `ChatModelConfig` objects, then calls `resolve_mcp_servers(configs, mcp_config)`.

### 1.3 resolve_mcp_servers expands wildcards and populates fields

`languagemodelcommon/configs/config_reader/mcp_json_reader.py`

For each model config:

1. **Wildcard expansion**: Any `AgentConfig` with `mcp_server: "*"` is replaced with one `AgentConfig` per `.mcp.json` server entry.
2. **Field population**: For each agent with a `mcp_server` key, the matching `.mcp.json` entry populates `url`, `headers`, `auth`, `oauth`, `display_name`, `description`, `auth_providers`, and `issuers`.
3. **Auto-auth**: OAuth config sets `auth: "jwt_token"`. An `Authorization` header sets `auth: "headers"`.
4. **Scope union**: A second pass merges OAuth scopes for agents sharing the same provider key, so the user gets a single consent prompt.

### 1.4 ConfigReader orchestrates with caching

`languagemodelcommon/configs/config_reader/config_reader.py`

`ConfigReader.read_model_configs_async()` checks three layers:

1. In-memory `ConfigExpiringCache` (TTL from `CONFIG_CACHE_TIMEOUT_SECONDS`)
2. MongoDB snapshot cache (survives restarts)
3. Filesystem / GitHub download (source of truth)

After loading, it resolves system prompt references via `PromptLibraryManager` (e.g. `name: "skills"` loads `prompts/skills.md`).

---

## Phase 2: Chat Request Setup

`language_model_gateway/gateway/providers/langchain_chat_completions_provider.py`

When a chat request arrives, the provider extracts resolved `AgentConfig` entries from the model config:

```python
mcp_tool_configs = model_config.get_agents()
```

### Tool Discovery mode (`use_tool_discovery: true`)

This is the default for models like `general_purpose`. The provider:

1. Calls `mcp_tool_provider.discover_tool_catalog(tools=mcp_tool_configs)` which registers server metadata (name, URL, category) in a `ToolCatalog` **without connecting to any MCP server**.
2. Creates two meta-tools:
   - `SearchToolsTool(catalog, resolver)` — LLM-callable tool for discovering available tools
   - `CallToolTool(catalog, mcp_tool_provider)` — LLM-callable tool for invoking a discovered tool
3. Passes these meta-tools (plus any non-MCP tools) to `GraphBuilder.create_graph_for_llm_async()`.

### Direct mode (`use_tool_discovery: false`)

The provider calls `mcp_tool_provider.get_tools_async()` which connects to **every** MCP server immediately and converts each remote tool into a LangChain tool. This loads all tools into the LLM context upfront.

### Graph creation

`languagemodelcommon/graph/graph_builder.py`

`GraphBuilder` creates a LangGraph `create_react_agent()` with the tool list. When a `ToolCatalog` is present, it adds `ToolDiscoveryMiddleware` which injects available categories into the system prompt as an XML block:

```xml
<available_tool_categories>
  <category>
    <name>skills</name>
    <description>Domain-specific skills and knowledge</description>
    <requires_auth>true</requires_auth>
  </category>
  <category>
    <name>github</name>
    <description>GitHub repository management</description>
  </category>
</available_tool_categories>
```

---

## Phase 3: LLM Conversation

The LLM receives two pieces of guidance for tool discovery:

1. **`skills.md` system prompt** — static instructions loaded from `PROMPT_LIBRARY_PATH/skills.md`
2. **ToolDiscoveryMiddleware injection** — dynamic list of available tool categories

### 3.1 search_tools

`languagemodelcommon/tools/mcp/search_tools_tool.py`

```
LLM → search_tools(category="skills", query="FHIR query builder")
```

1. Checks if the category's servers have been resolved. On first call, triggers **lazy resolution**:
   - Connects to the MCP server via `create_mcp_session()`
   - Calls `list_tools()` to fetch tool metadata
   - Caches metadata with TTL (`MCP_TOOLS_METADATA_CACHE_TIMEOUT_SECONDS`, default 3600s)
   - Indexes tools in the `ToolCatalog`'s BM25 search index
2. Performs BM25 Okapi-ranked search over tool name, description, and parameter schemas.
3. Returns matching tool schemas as JSON (name, description, input schema).

### 3.2 call_tool

`languagemodelcommon/tools/mcp/call_tool_tool.py`

```
LLM → call_tool(name="load_skill", arguments={"skill_name": "fhir_query_builder"})
```

1. Looks up the tool by name in `ToolCatalog` to find the owning server's `AgentConfig`.
2. Calls `mcp_tool_provider.execute_mcp_tool()` using a pooled MCP session (`McpSessionPool` reuses TCP+TLS connections within a request).
3. Applies interceptor chain: authentication, tracing, output truncation.
4. Returns the tool result (text, images, embedded resources) to the LLM.

---

## Visual Summary

```
┌─────────────────────────────────────────────────────┐
│              Marketplace Plugin Directory             │
│                                                       │
│  plugins/all-employees/.mcp.json                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │ mcpServers:                                     │ │
│  │   skills:      { url, oauth }                   │ │
│  │   github:      { url, headers }                 │ │
│  │   atlassian:   { url, oauth }                   │ │
│  │   google-drive: { url, oauth }                  │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
            McpJsonReader.read_mcp_json()
            scans plugins/*/.mcp.json
                       │
                       ▼
            ┌─────────────────────┐
            │  McpJsonConfig      │
            │  (merged servers)   │
            └──────────┬──────────┘
                       │
            resolve_mcp_servers()
            expands mcp_server:"*"
            populates url, auth, oauth
                       │
                       ▼
  ┌─────────────────────────────────────────────────┐
  │  ChatModelConfig (general_purpose)               │
  │                                                   │
  │  tools: [                                         │
  │    AgentConfig(name="skills", url=..., oauth=..) │
  │    AgentConfig(name="github", url=..., auth=..)  │
  │    AgentConfig(name="atlassian", url=..., ...)   │
  │  ]                                                │
  │  system_prompts: ["general_purpose", "skills"]   │
  │  use_tool_discovery: true                         │
  └──────────────────────┬────────────────────────────┘
                         │
                    per chat request
                         │
                         ▼
           ┌─────────────────────────────┐
           │  ToolCatalog                 │
           │  (server metadata only,     │
           │   no MCP connections yet)   │
           └──────────┬──────────────────┘
                      │
                      ▼
           ┌──────────────────────────┐
           │  LangGraph Agent          │
           │                           │
           │  System prompts:          │
           │    skills.md              │
           │    + <tool_categories>    │
           │                           │
           │  Tools:                   │
           │    search_tools           │
           │    call_tool              │
           └──────────┬───────────────┘
                      │
                 LLM decides
                      │
    ┌─────────────────┴──────────────────┐
    │                                     │
    ▼                                     ▼
search_tools(                      call_tool(
  category="skills",                 name="load_skill",
  query="FHIR")                      arguments={...})
    │                                     │
    │  lazy connect to                    │  pooled MCP session
    │  MCP server                         │  auth interceptors
    │  BM25 search                        │
    │                                     │
    ▼                                     ▼
[{name: "load_skill",              tool result text
  description: "...",              (returned to LLM)
  parameters: {...}}]
(returned to LLM)
```

---

## Configuration

### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `PLUGINS_MARKETPLACE` | Root directory for marketplace plugins | `/configs/marketplace` |
| `MODELS_OFFICIAL_PATH` | Directory containing model config JSON files | `/configs/chat_completions/official` |
| `PROMPT_LIBRARY_PATH` | Directory containing system prompt files | `/configs/chat_completions/prompts` |
| `CONFIG_CACHE_TIMEOUT_SECONDS` | TTL for in-memory config cache | `10` |
| `MCP_TOOLS_METADATA_CACHE_TIMEOUT_SECONDS` | TTL for cached MCP tool metadata | `3600` |

### Docker Compose Volumes

```yaml
volumes:
  # Config files (from local checkout or GitHub download)
  - ./language-model-gateway-configs/chat_completions:/configs/chat_completions
  - ./language-model-gateway-configs/marketplace:/configs/marketplace
```

### Key Files

| File | Purpose |
|---|---|
| `marketplace/plugins/all-employees/.mcp.json` | MCP server definitions (URLs, auth, display names) |
| `chat_completions/official/general_purpose.json` | Model config with `mcp_server: "*"` wildcard |
| `chat_completions/prompts/skills.md` | System prompt instructing LLM to use `search_tools`/`call_tool` |

---

## Adding a New MCP Server

1. Add an entry to `marketplace/plugins/{plugin-name}/.mcp.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "url": "https://my-server.example.com/mcp",
      "displayName": "My Server",
      "description": "Does useful things",
      "oauth": {
        "clientId": "abc123",
        "authServerMetadataUrl": "https://idp.example.com/.well-known/openid-configuration"
      }
    }
  }
}
```

2. Any model config with `mcp_server: "*"` will automatically pick it up on the next config refresh.

3. To restrict a server to a specific model, set `mcp_server: "my-server"` on an `AgentConfig` in the model's JSON instead of using the wildcard.

---

## Design Decisions

**Lazy discovery over eager loading.** Tool discovery connects to MCP servers only when the LLM searches for tools in a category. This keeps the initial context window small and avoids connections to servers that won't be used in a given conversation.

**BM25 search over tool metadata.** The `ToolCatalog` indexes tool name, description, and parameter schemas. This lets the LLM find relevant tools without loading the full tool list into its context.

**Per-request session pooling.** `McpSessionPool` reuses TCP+TLS connections within a single chat request. Multiple `call_tool` invocations to the same server share one connection.

**Marketplace plugin convention.** `.mcp.json` files live inside plugin directories (`plugins/{name}/.mcp.json`) rather than a single global file. This allows different plugin bundles to contribute servers independently. The same convention is used by `mcp-server-gateway`.

**Two-pass OAuth scope union.** Agents sharing the same OAuth provider key get their scopes merged in a second pass. This produces a single consent prompt covering all tools from that provider.
