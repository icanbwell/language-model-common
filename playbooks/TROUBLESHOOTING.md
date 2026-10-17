# Troubleshooting

Last verified: 2026-10-16 against 6245270 by Claude

First checks for symptoms in this repo. This is a library: a symptom is observed in a consumer
(baileyai, baileyai-skills-service, mcp-fhir-agent), and the first job is
to decide whether this library is involved. Where to look is in [MONITORING.md](MONITORING.md).
Method, not numbers.

## Reading signals correctly

- Identify the library version the consumer actually ran before reading this repo's code: read the consumer's lock file at the deployed commit. `main` here may be ahead of what the consumer pinned (see [DEPLOYMENT.md](DEPLOYMENT.md)).
- A WARNING from `languagemodelcommon.mcp.mcp_client.*` describes the library's own retry or fallback; whether the request then failed is decided by the consumer's per-request line, not by this WARNING.
- Library log lines carry no request id of their own (`unverified`; no request-id field set in `languagemodelcommon/utilities/logger/`). Correlate by time window and the consumer's trace id.
- Wide freetext log searches run in 3h or smaller chunks.

## Come here if you're seeing the model invent tool results or list records it cannot have

**Signal:** assistant text naming records, medications, care team members or tool-call markup (`<call_tool>`, `<search_tools>`) with no tool message in the conversation.

This library does not bind record tools and holds no tool catalog (tool-catalog code moved to baileyai-skills-service; evidence: `languagemodelcommon/mcp/` has only generic client, session and interceptor code). It is involved only through the request wrapper that turns a request's `tools` into `AgentConfig` objects. The diagnosis belongs to baileyai: follow baileyai's `playbooks/TROUBLESHOOTING.md` (`unverified` until that playbook merges) and baileyai-skills-service's.

**First checks that touch this repo:**

1. Which tools were in the request: `extract_mcp_agent_configs` in `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py` builds one `AgentConfig` per `type: "mcp"` entry. An entry with `server_url` becomes a URL tool; one with only `server_label` becomes a label reference (`mcp_server=server_label`).
2. For a URL tool, `auth="headers"` is set only when the entry supplied `headers`; a headerless URL tool leaves `auth` unset. `AgentConfig.tool_argument_allowlist` is required for a `headers` server's tools to be discoverable or callable (`languagemodelcommon/configs/schemas/config_schema.py`); the downstream error text `has no approved argument allowlist` is raised outside this repo (`unverified` where).
3. Whether the MCP connection itself failed: look for `MCP session establishment attempt N/M failed` ([MONITORING.md](MONITORING.md)).

**Likely causes (suspected, not verified here):** tools not bound for the turn upstream of this library; a URL tool filtered because its allowlist is missing.
**Action:** hand off to the baileyai playbook for Groundcover phrases, the `pseudo_tool_call_detected_total` metric, the degraded-notice step and the Mongo conversation lookup; do not repeat them here.
**Do not:** copy message content into a ticket or file. Query counts, roles and timestamps only.

## Come here if you're seeing MCP tool calls fail or stall in a consumer

**Signal:** tool errors or long gaps during a tool call.
**First checks:**

1. `MCP session establishment attempt N/M failed` WARNINGs: transient connect failure being retried by `open_initialized_mcp_session` (`languagemodelcommon/mcp/mcp_client/session.py`); the tool call itself is never retried (comment in `tool_invocation.py`).
2. `Task protocol failed for ... falling back to call_tool` (`tool_invocation.py`): a task-protocol call degraded to a plain call.
3. Timeouts: `TOOL_CALL_TIMEOUT_SECONDS` (default 600) and `MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS` (`language_model_common_environment_variables.py`); the consumer's own timeout may be shorter.
4. Protocol negotiation: `MCP_PROTOCOL_NEGOTIATION_MODE` (`legacy` or `auto`; evidence: `adrs/0003-configurable-mcp-protocol-negotiation-mode.md`).

**Likely causes:** server unreachable or slow (suspected); negotiation mode mismatch with the server (suspected).
**Action:** check the server's own logs in its repo; if the library misbehaves, reproduce with a test under `tests/mcp/`.

## Come here if you're seeing stale tool lists, prompts or model configs after a change upstream

**First checks:** caches have TTLs. MCP tool list: `MCP_TOOLS_METADATA_CACHE_TTL_SECONDS`; prompt store: `PROMPT_STORE_TTL_SECONDS` (default 3600); model config cache: `MODEL_CONFIG_CACHE_TTL_SECONDS` (default 3600); `CONFIG_CACHE_TIMEOUT_SECONDS` is defined (code default 3600, `language_model_common_environment_variables.py`) but nothing reads it, so it controls nothing; the README's `120` is stale. Mongo-backed entries are in the collections in [MONITORING.md](MONITORING.md).
**Action:** wait out the TTL or clear the entry with the owning consumer's process; confirm which consumer owns the data first.

## Come here if you're seeing a consumer that cannot pick up a fix released here

**First checks:** the fix's release exists on this repo's Releases page; the consumer's `pyproject.toml` floor and lock file; for `baileyai` and `baileyai-skills-service`, the `exclude-newer-package` override. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Escalation

Owning team and Slack channel for this repo are `unverified` (`CODEOWNERS` names only `@icanbwell/enterprise-architecture`, for `AGENTS.md` and `CLAUDE.md`). Bring: the consumer, its library version, time window, trace id; never PHI or message content.
