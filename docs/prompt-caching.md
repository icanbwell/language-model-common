# Prompt Caching

Prompt caching reduces latency and cost by reusing previously computed KV-cache entries for stable prompt prefixes across API calls.

---

## Architecture

Caching is managed at the **prompt layer**, not the model layer. System prompt content blocks declare their own `cache_control` markers, giving explicit control over what is cached.

### How it works

The graph builder constructs a `SystemMessage` with structured content blocks:

```python
SystemMessage(content=[
    {"type": "text", "text": "stable instructions...", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "tenant: acme, user: alice"}  # no cache_control
])
```

- Blocks **with** `cache_control` are cached by the provider (Bedrock/Anthropic)
- Blocks **without** `cache_control` are processed normally on every request
- User messages are **never** cached

### Cache boundary design

```
[System block 1: stable instructions]  ← cache_control: ephemeral (shared across tenants)
[System block 2: tenant/user context]  ← NOT cached (tenant-specific, changes per request)
[Tools]                                 ← cached by provider alongside stable prefix
[Messages]                              ← NOT cached (per-conversation, per-user)
```

This ensures:
- The expensive stable prefix (instructions, few-shot examples) is computed once
- Tenant-specific context never contaminates the shared cache
- No risk of cross-tenant data leakage

### Why not `.bind(cache_control=...)`?

The previous approach used `model.bind(cache_control=...)` which told `langchain-aws` to add cache markers to system, tools, AND the last user message. This:
1. Cached user messages — a potential cross-tenant cache sharing vector
2. Gave no control over which system prompt blocks were cacheable
3. Required `PROMPT_CACHE_ENABLED` / `PROMPT_CACHE_TTL` environment variables

The new approach removes all of that in favor of explicit, per-block cache control.

---

## Provider support

| Provider | Cache mechanism | Applied by |
|----------|----------------|------------|
| ChatAnthropicBedrock | `cache_control` on content blocks | Graph builder |
| ChatBedrockConverse | Not currently supported | — |
| OpenAI | Not applicable | — |

---

## Configuration

No environment variables. Caching behavior is controlled entirely by how system prompts are structured in the agent config. To adjust what gets cached, change which blocks get `cache_control` in the graph builder.

---

## Observability

When cache tokens are present in a response, the token usage metadata includes:

```python
{
    "cache_read": 4523,      # Tokens read from cache
    "cache_creation": 0,     # Tokens written to cache
}
```

A healthy setup shows `cache_creation > 0` on the first turn and `cache_read > 0` on subsequent turns.
