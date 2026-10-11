# Monitoring

Last verified: 2026-10-11 against 8bc128d by Claude

Where to look when investigating this repo. This repo is a library: it has no workload, namespace,
dashboard or monitor of its own. Its code runs inside each consumer's process and logs through the
consumer's logger. Locations, query shapes and the meaning of log lines only; no findings.

## Groundcover

No namespace of its own (library; evidence: no `.helm/` directory and no deployment workflow on `origin/main`). Look in the consuming service's namespace and workload, for example workload `baileyai` for baileyai. Namespace convention per consumer is in that repo's `playbooks/MONITORING.md`, where it exists (`unverified` which consumers have adopted playbooks).

Query shape: filter the consumer's workload, then free-text on the logger name. Loggers are `logging.getLogger(__name__)`, so the module path appears as the logger name, for example `languagemodelcommon.mcp.mcp_client.session` (evidence: `languagemodelcommon/mcp/mcp_client/session.py`). Whether Groundcover exposes the logger name as a filterable field is `unverified`; search the module path as text. Run wide free-text searches in 3h or smaller chunks.

To tell which library version a consumer is running, read the consumer's lock file at the deployed commit, not logs (see [DEPLOYMENT.md](DEPLOYMENT.md)).

## Dashboards and monitors

| Platform | What exists | Where |
|---|---|---|
| Groundcover dashboards | not applicable; library | consumer repos |
| Groundcover monitors | not applicable; library | consumer repos |
| Sigma | not applicable; not a data product | |
| Mixpanel / PostHog | not applicable | |
| Databricks | not applicable | |
| Sentry | none configured in this repo (no Sentry dependency in `pyproject.toml`) | |
| CI | `Build and Test` and `Upload Python Package` workflows | `.github/workflows/build_and_test.yml`, `.github/workflows/python-publish.yml` |

## Log formats to know

The library has no fixed log schema. Default root logging, installed only when the host has not configured handlers, is plain text to stdout: `%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s` (`languagemodelcommon/utilities/logger/log_levels.py`). A consumer that configures its own handlers overrides it. Lines that carry signal:

- `MCP session establishment attempt N/M failed for <url> (retrying in ...)` WARNING, from `languagemodelcommon/mcp/mcp_client/session.py`. Transient MCP connection failure being retried. The final attempt raises instead of logging this line.
- `Task protocol failed for <tool> on <url>, falling back to call_tool` WARNING, `languagemodelcommon/mcp/mcp_client/tool_invocation.py`. Also invalidates the tool-list cache entry.
- WARNINGs in `languagemodelcommon/mcp/mcp_client/mcp_tool_list_store.py`, `tool_list_cache.py`, `session_pool.py`, `server_card_discovery.py`: tool-list cache and session-pool trouble (read the message at the cited line).
- `languagemodelcommon/models/model_factory.py` and `languagemodelcommon/models/sanitizing_bedrock_converse.py`: model construction and Bedrock request sanitizing.
- Per-source log levels: `LOG_LEVEL` plus `<SOURCE>_LOG_LEVEL` for sources `HTTP_TRACING, CONFIG, INITIALIZATION, HTTP, AUTH, TOKEN_EXCHANGE, DATABASE, LLM, FILES, IMAGE_GENERATION, IMAGE_PROCESSING, MCP, AGENTS, ERRORS, ROUTER, SERVICES, CACHE, EVALUATOR, RESPONSES, TOOLS, SSE` (`log_levels.py`). A consumer's level for a source decides whether DEBUG lines from this library exist at all.
- Tracing: MCP tool calls are wrapped in an OpenTelemetry span named `mcp.tool.<tool name>` with attributes `mcp.server_name`, `mcp.tool_name`, `mcp.arguments`, and trace context is injected into MCP request headers (`languagemodelcommon/mcp/interceptors/tracing.py`). `mcp.arguments` carries the call arguments; treat span attributes as sensitive when sharing.

The library does not define a per-request log line; request counts come from the consumer.

## Known noise

None recorded. `unverified` whether any WARNING above is routinely benign.

## Databases

The library reads and writes MongoDB through consumer-supplied configuration; it owns no cluster. Database and collection names are environment-driven (`languagemodelcommon/utilities/environment/language_model_common_environment_variables.py`):

| Purpose | Database env var (default) | Collection env var (default) |
|---|---|---|
| MCP tool-list cache (`McpToolListStore`) | `MCP_TOOL_CACHE_DB_NAME` (falls back to `MONGO_LLM_STORAGE_DB_NAME`, then `language_model_gateway`) | `MCP_TOOL_CACHE_DB_COLLECTION` (`mcp-tool-cache`) |
| Model config cache | `unverified` which database | `MODEL_CONFIG_CACHE_COLLECTION_NAME` (`models`) |
| Prompt store | `PROMPT_STORE_DB_NAME` (falls back to `MONGO_LLM_STORAGE_DB_NAME`) | `PROMPT_STORE_COLLECTION` (`prompts`) |
| LangGraph store / checkpointer (off unless `ENABLE_LLM_STORE` / `ENABLE_LLM_CHECKPOINTER`) | `MONGO_LLM_STORAGE_DB_NAME` (`llm_storage`) | `MONGO_LLM_STORAGE_STORE_COLLECTION_NAME` (`stores`), `MONGO_LLM_STORAGE_CHECKPOINTER_COLLECTION_NAME` (`checkpoints`) |
| OAuth DCR registrations / tokens | consumer-set | `MONGO_DB_DCR_COLLECTION_NAME` (`dcr_registrations`), `MONGO_DB_TOKEN_COLLECTION_NAME` |

Which cluster and what values a consumer sets are in that consumer's Helm values (`unverified` here). Never record connection strings.

Bailey conversation messages are not stored by this library. For the conversation lookup shape (database `bailey-ai`, collection `bailey_messages`), see baileyai's `playbooks/MONITORING.md` (`unverified` until that playbook merges).

## Kubernetes / Deployment

Not applicable; library. See the consumer. Consumers pin this package in their `pyproject.toml` (see [RELATIONSHIPS.md](RELATIONSHIPS.md)).

## Other

- Package releases: GitHub Releases page of this repo and PyPI project `language-model-common` (evidence: `CONTRIBUTING.md`).
- Request-wrapper tool config: a URL request tool (`type: "mcp"` with `server_url`) gets `auth="headers"` only when it supplies headers (`languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py`, `extract_mcp_agent_configs`). Use this when checking how a request's tools were turned into `AgentConfig` objects.
