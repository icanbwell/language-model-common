# Relationships

Last verified: 2026-10-16 against 6245270 by Claude

Who depends on this repo and what it depends on, so a change here (or there) is checked against the
right neighbors. This file is the source of truth for the repo; the bwell-repo-wiki repo page cites
it. Every row names its evidence (a path or config key) or says `unverified`.

## Upstream: who calls or consumes this repo

Consumers import the `languagemodelcommon` package; there is no network contract. The contract is
the package's public Python API and the config schemas it reads.

| Consumer | What it uses | Contract it depends on | If you change... check | Evidence |
|---|---|---|---|---|
| baileyai | request wrappers, MCP client and tool plumbing, streaming converters, history, `AgentConfig` (floor `>=3.0.37`) | public class and function signatures, SSE event shapes, `AgentConfig` fields | any of those: baileyai's tests against the new version | baileyai `pyproject.toml` on `origin/main` |
| baileyai-skills-service | config reader, prompt library, container factory, MCP and `AgentConfig` types (floor `>=3.0.30`) | `ConfigReader`, `PromptLibraryManager`, `AgentConfig.tool_argument_allowlist` and `auth="headers"` semantics | `auth` and allowlist handling: its argument-allowlist gate | `baileyai_skills_service/container/container_factory.py`; `pyproject.toml` |
| mcp-fhir-agent | package dependency (floor `>=3.0.30`); which modules it uses: `unverified` | `unverified` | `unverified` | mcp-fhir-agent `pyproject.toml` |

The bwell-repo-wiki repo page also says complaint-parser builds on this package; no import or pin was found in complaint-parser's `origin/main` (`unverified`).

## Downstream: what this repo calls or depends on

| Dependency | What we use it for | How we reach it (env var names, never values) | Assumptions we make | Evidence |
|---|---|---|---|---|
| MCP servers | tool discovery and invocation over HTTP | `AgentConfig.url` per tool, from consumer config | server speaks MCP; negotiation mode `legacy` or `auto` | `languagemodelcommon/mcp/mcp_client/`; `adrs/0003-configurable-mcp-protocol-negotiation-mode.md` |
| MongoDB | tool-list cache, model config cache, prompt store, optional LangGraph store and checkpointer, DCR and token collections | `MONGO_LLM_STORAGE_URI` and related `MONGO_*` vars | consumer provides a reachable cluster; collections per [MONITORING.md](MONITORING.md) | `language_model_common_environment_variables.py` |
| AWS (S3, Bedrock, Textract) | config and file storage, Bedrock models, OCR | AWS credentials from the consumer's environment; `AWS_BEDROCK_RETRY_MODE`, `AWS_BEDROCK_MAX_ATTEMPTS` | consumer's role permits the calls | `languagemodelcommon/aws/`, `models/`, `ocr/` |
| GitHub | download of config and prompt directories | `GITHUB_TOKEN`, or the GitHub App set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, `GITHUB_APP_INSTALLATION_ID`; cache directory `GITHUB_CONFIG_CACHE_DIR` | token can read the config repo | `languagemodelcommon/configs/config_reader/github_directory_helper.py`; `languagemodelcommon/github/token_provider.py`; `languagemodelcommon/utilities/environment/language_model_common_environment_variables.py` |
| Model providers (OpenAI, Bedrock Anthropic, Google GenAI) | chat and image models | provider credentials supplied by the consumer | `unverified` per provider | `pyproject.toml` (`openai`, `langchain-aws[anthropic]`, `langchain-google-genai`) |
| oidcauthlib | OAuth and OIDC primitives; shared environment variables | consumer's OIDC settings | first-party package, `>=3.0.21`; resolved without a freshness delay | `pyproject.toml`; `CONTRIBUTING.md` |
| LangChain, LangGraph, `mcp` SDK | agent graph, MCP client | n/a | major-version changes in these ripple to every consumer | `pyproject.toml` |
| OpenTelemetry | trace propagation into MCP calls | the host's OTel configuration | host initializes OTel (`unverified`) | `languagemodelcommon/mcp/interceptors/tracing.py` |

## If you change X, check Y

- A public signature, class or SSE event shape: every consumer row above; run the consumer's tests against a build of the change before releasing.
- `AgentConfig` fields or the `auth` value semantics (`auth="headers"`, `pass_through`): baileyai-skills-service's allowlist gate and baileyai's tool resolution (`languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py`, `languagemodelcommon/configs/schemas/config_schema.py`).
- Cache names, TTL defaults or collection names: [MONITORING.md](MONITORING.md) Databases table and the consumers' Helm values (`unverified` where).
- A major-version bump of this package: every consumer floor in the table above.
- Removing code from this package: a consumer on a lower floor still imports it until it bumps.

## Not a relationship

- `docker-compose.yml` is a local dev container only (`dev` service), not a service anyone calls.
- bai-evaluator has no dependency on this package (bwell-repo-wiki `repos/language-model-common.md`). Its HTTP calls to language-model-gateway are not a relationship of this package.
- language-model-gateway is retired: it was a consumer of this package until it was archived on GitHub on 2026-10-16, so it is not listed above. No package pin or workflow reference to it remains in `pyproject.toml` or `.github/`; the one `docs/` mention (`docs/migration-model-factory-and-reasoning.md`) is a historical migration note.
- The `alpine-wheels` index in `pyproject.toml` is a wheel source for builds, not a runtime dependency.

Next: [REVIEW.md](REVIEW.md) (blast radius for the PR).
