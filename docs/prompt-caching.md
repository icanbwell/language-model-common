# Prompt Caching

Prompt caching reduces latency and cost by reusing previously computed KV-cache entries for stable prompt prefixes across API calls.

---

## Architecture

Caching is managed at the **prompt layer** via per-prompt configuration, not at the model layer. Each system prompt in the chat completion config declares whether it should be cached using the `cache` field.

### How it works

Each `PromptConfig` in the agent's `system_prompts` array has an optional `cache` boolean:

```json
{
  "system_prompts": [
    {"role": "system", "name": "bailey_system_prompt", "cache": true},
    {"role": "system", "name": "skills", "cache": true},
    {"role": "system", "name": "datetime_context_message_format"}
  ]
}
```

The graph builder constructs a `SystemMessage` with structured content blocks:

```python
SystemMessage(content=[
    {"type": "text", "text": "stable instructions...", "cache_control": {"type": "ephemeral"}},
    {"type": "text", "text": "Today is Monday..."}  # no cache_control
])
```

- Prompts with `"cache": true` get `cache_control: {"type": "ephemeral"}`
- Prompts without `cache` (or `cache: false/null`) are processed normally on every request
- User messages are **never** cached
- Default is **opt-in**: prompts are NOT cached unless explicitly marked

### Cache boundary design

```
[System block 1: stable instructions]  ← cache: true → cache_control: ephemeral
[System block 2: skills/tools list]    ← cache: true → cache_control: ephemeral
[System block 3: datetime context]     ← cache: false → NOT cached (changes per request)
[Tools]                                 ← cached by provider alongside stable prefix
[Messages]                              ← NOT cached (per-conversation, per-user)
```

This ensures:
- The expensive stable prefix (instructions, skills) is computed once
- Per-request context (date/time, tenant info) never contaminates the shared cache
- No risk of cross-tenant data leakage
- Cache behavior is explicitly declared in config, not hardcoded

---

## Provider support

| Provider | Cache mechanism | Applied by |
|----------|----------------|------------|
| ChatAnthropicBedrock | `cache_control` on content blocks | Graph builder |
| ChatBedrockConverse | Not currently supported | — |
| OpenAI | Not applicable | — |

---

## Configuration

Cache behavior is controlled per-prompt in the chat completion config JSON via the `cache` field on `PromptConfig`. No environment variables needed.

To cache a prompt, add `"cache": true` to its entry in `system_prompts`. Prompts without this field (or with `cache: false`) are never cached.

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
