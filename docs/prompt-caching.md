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
- User/assistant/tool messages are never cached across turns; within a single multi-round
  tool-calling turn they can be, behind a flag — see "Within-turn history caching" below
- Default is **opt-in**: prompts are NOT cached unless explicitly marked

### Cache boundary design

```
[Tools]                                 ← cached by provider alongside stable prefix
[System block 1: stable instructions]  ← cache: true → cache_control: ephemeral
[System block 2: skills/tools list]    ← cache: true → cache_control: ephemeral
[System block 3: datetime context]     ← cache: false → NOT cached (changes per request)
[Messages]                              ← NOT cached across turns; within-turn caching
                                           available behind ENABLE_HISTORY_PROMPT_CACHING
```

Anthropic renders request content in `tools → system → messages` wire order, and
a `cache_control` breakpoint on a system block caches everything *before* it in
that order — including tool definitions — as part of the same prefix. **Confirmed
empirically** (BAI-706, see `adrs/0001-extend-prompt-caching-to-tools-and-history.md`
Phase 0 step 2): a direct `ChatAnthropicBedrock` call with 44 bound tool schemas and
a cached system prompt showed `cache_creation` covering both — isolating the tool
contribution (a second call with the same system prompt and no tools bound) measured
6733 of 9154 cached tokens as attributable to the tool schemas alone. No separate
`cache_control` on tools is needed or implemented; placing the breakpoint on the last
cached system block is sufficient.

This ensures:
- The expensive stable prefix (tools, instructions, skills) is computed once
- Per-request context (date/time, tenant info) never contaminates the shared cache
- No risk of cross-tenant data leakage
- Cache behavior is explicitly declared in config, not hardcoded

### Within-turn history caching

`HistoryCacheMiddleware` (`languagemodelcommon/converters/history_cache_middleware.py`)
places a `cache_control` breakpoint at the end of the message list before each model
call after the first within a single multi-round tool-calling turn — the pattern used
by agentic coding tools: cache everything-so-far, pay full price only for the newest
delta. A `ToolMessage` in the request's message list is the signal a round-trip
already happened, so the first call in a turn (nothing to read yet) is never tagged.

Unlike the system-prompt mechanism, this does not place `cache_control` on content
blocks directly — it sets `model_settings["cache_control"]` on the `ModelRequest`,
which `ChatAnthropicBedrock` (inherited from `langchain_anthropic`) turns into a
block-level breakpoint on the last eligible block of the last message, recomputed
fresh from the live message list on every call. That recomputation is what makes
re-anchoring after history trimming automatic — there is no fixed offset to get
stale.

**Off by default** (`ENABLE_HISTORY_PROMPT_CACHING=false`). Conversation messages
carry far more PHI risk than system prompts or tool schemas; this stays off until
security/EA confirm Bedrock prompt-cache storage is covered by existing PHI/BAA
commitments (see `adrs/0001-extend-prompt-caching-to-tools-and-history.md`, Open
Question 4). Scoped to within-turn only — a separate turn (new user message) never
shares a cache read with a prior one.

---

## Provider support

| Provider | Cache mechanism | Applied by |
|----------|----------------|------------|
| ChatAnthropicBedrock | `cache_control` on content blocks | Graph builder |
| ChatBedrockConverse | Not currently supported | — |
| OpenAI | Not applicable | — |

---

## Configuration

Cache behavior is controlled per-prompt in the chat completion config JSON via the `cache` field on `PromptConfig`.

To cache a prompt, add `"cache": true` to its entry in `system_prompts`. Prompts without this field (or with `cache: false`) are never cached.

Within-turn history caching is controlled separately via the `ENABLE_HISTORY_PROMPT_CACHING`
environment variable (default `false`), read by `LanguageModelCommonEnvironmentVariables.enable_history_prompt_caching`.

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
