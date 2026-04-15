# Migration Guide: languagemodelcommon v1 to v2

This document covers all breaking changes, new capabilities, and bug fixes in `languagemodelcommon` v2. Clients upgrading from v1 must review the breaking changes below and update their code accordingly.

---

## Breaking Changes

### 1. ModelFactory moved to languagemodelcommon

`ModelFactory` has been moved from `language-model-gateway` into `languagemodelcommon`. This centralizes model instantiation logic in the shared library where it belongs, alongside `AwsClientFactory`, `ChatModelConfig`, and other model infrastructure.

#### v1

```python
from language_model_gateway.gateway.models.model_factory import ModelFactory
```

#### v2

```python
from languagemodelcommon.models.model_factory import ModelFactory
```

#### Action required

Search your codebase for the old import path and replace it:

```bash
grep -r "from language_model_gateway.gateway.models.model_factory" .
```

If you subclass `ModelFactory`, update the base class import. The class interface is unchanged — only the module location moved.

### 2. All \_\_init\_\_.py files are now empty

v2 enforces the convention that `__init__.py` files must not contain code or re-exports. In v1, `languagemodelcommon/utilities/cache/__init__.py` re-exported `ExpiringCache` and `ConfigExpiringCache`. This is removed in v2.

#### v1

```python
from languagemodelcommon.utilities.cache import ConfigExpiringCache
from languagemodelcommon.utilities.cache import ExpiringCache
```

#### v2

```python
from languagemodelcommon.utilities.cache.config_expiring_cache import ConfigExpiringCache
from languagemodelcommon.utilities.cache.expiring_cache import ExpiringCache
```

#### Action required

Search for package-level cache imports and update to the explicit module path:

```bash
grep -r "from languagemodelcommon.utilities.cache import" .
```

Going forward, always import from the specific module, not the package.

### 3. OAuth authenticator validates token endpoint origin

`OAuthAuthenticator.login_and_get_oauth_access_token()` now validates that the `token_endpoint` returned by the OpenID well-known configuration shares the same scheme and hostname as the `openid_provider_url`. This closes a chained SSRF vector where a compromised well-known response could redirect token requests (carrying client credentials) to an internal service such as the cloud metadata endpoint.

#### Who is affected

Deployments where the OIDC well-known URL and token endpoint are hosted on different hostnames:

```
well-known:     https://auth.example.com/.well-known/openid-configuration
token_endpoint: https://token.example.com/oauth/token   <-- will now raise ValueError
```

Standard OIDC providers (Keycloak, Auth0, Okta, Azure AD) serve both endpoints from the same hostname and are unaffected.

#### Action required

If you have a legitimate multi-host OIDC provider, either:

- Align your provider URLs so the well-known and token endpoint share a hostname, or
- Override `_validate_same_origin` in a subclass to accommodate your provider's topology

---

## New Capabilities

### 4. Extended thinking support for Anthropic models via Bedrock

`ModelFactory` now supports Anthropic extended thinking (reasoning) for Bedrock models. When enabled, the model performs internal chain-of-thought reasoning before producing a response.

#### Configuration

Add `thinking_budget_tokens` to your model config's `model_parameters`:

```json
{
  "model_parameters": [
    { "key": "thinking_budget_tokens", "value": 4096 }
  ]
}
```

This is translated into `additional_model_request_fields` for `ChatBedrockConverse`. No code changes required — config-only.

#### Debug visibility

When debug mode is enabled (prefix your message with `DEBUG:`), reasoning content is rendered as collapsible `<details>` sections in the streamed response. When debug mode is off, reasoning is silently filtered from the output — the model still benefits from deeper thinking, but the reasoning text is not exposed.

### 5. Reasoning content block handling in streaming

`iter_message_content_text_chunks` now recognizes `reasoning_content` and `reasoning` type blocks from Anthropic extended thinking responses. These blocks are:

- Excluded from the main response text (never visible to end users in normal mode)
- Captured in `non_text_blocks` for callers that need them
- Rendered as collapsible `<details><summary>Reasoning</summary>` sections in debug mode by `LangGraphStreamingManager`

No action required. This is additive and does not change existing behavior for requests that do not use extended thinking.

### 6. MCP Apps — interactive HTML UIs from MCP tools

MCP tools that declare a `ui://` resource URI in their metadata now have their HTML fetched and streamed to the client as a custom `event: mcp_app` SSE event. The client renders the HTML in a sandboxed iframe.

Key components:
- `mcp_client/ui_resource.py` — Detection, fetching, and JavaScript injection helpers
- `MCPToolProvider.fetch_mcp_app_embed()` — Orchestrates UI resource fetch with session pool reuse
- `CallToolTool` — Returns `(text, artifact)` tuples via `content_and_artifact` response format
- `LangGraphStreamingManager._handle_on_tool_end()` — Emits `event: mcp_app` SSE when artifact contains an embed
- `ChatRequestWrapper.create_mcp_app_sse_event()` — Formats the custom SSE frame

No action required for existing tools. MCP Apps are automatically detected when tools declare `meta.ui.resourceUri`. See [docs/mcp-apps.md](mcp-apps.md) for full details.

---

## Bug Fixes

### 6. Responses API content type handling

`convert_responses_api_to_single_message` now handles `content` fields that are lists of structured content blocks, not just strings. Previously, a list content value caused:

```
TypeError: can only concatenate str (not "list") to str
```

The fix extracts text from known content block types (`input_text`, `output_text`, `text`) when content is a list. String content continues to work as before.

### 7. System command manager handles list content

`SystemCommandManager.run_system_commands()` now safely handles messages where `content` is a list of structured content blocks (e.g., multimodal input) rather than a plain string. Previously, this caused:

```
AttributeError: 'list' object has no attribute 'lower'
```

List content is now treated as non-matching for system commands, which is the correct behavior since system commands are always plain text strings.