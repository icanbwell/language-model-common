# Prompt Caching

This document describes how the language-model-gateway uses AWS Bedrock prompt caching to reduce latency and cost for Claude model invocations.

---

## Overview

Prompt caching allows Bedrock to reuse previously computed KV-cache entries for identical prompt prefixes across API calls. Since system prompts and tool definitions are stable across requests within a session (and often across sessions), caching these prefixes avoids redundant computation on every turn.

**How it works at a high level:**

```
Request 1: [system prompt + tools + messages]
  → Bedrock processes full prefix, writes to cache (cache_creation tokens)
  → Response generated

Request 2: [system prompt + tools + new messages]
  → Bedrock finds cached prefix match at system+tools boundary
  → Only processes new tokens after cache hit (cache_read tokens)
  → ~10x cheaper for cached portion, lower latency
```

**Cache hierarchy in the Converse API:**

```
tools → system → messages
  ↑         ↑         ↑
  cachePoint  cachePoint  cachePoint (on last message)
```

Each `cachePoint` block marks a boundary where Bedrock can cache the prefix up to that point.

---

## Configuration

Prompt caching is controlled by two environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROMPT_CACHE_ENABLED` | `true` | Master toggle. Set to `false` to disable caching entirely. |
| `PROMPT_CACHE_TTL` | `5m` | Cache time-to-live. Valid values: `5m` or `1h`. |

### When to use each TTL

| TTL | Write cost | Read cost | Best for |
|-----|-----------|-----------|----------|
| `5m` | 1.25x base input | 0.1x base input | Active conversations with frequent turns (< 5 min between messages) |
| `1h` | 2x base input | 0.1x base input | Longer sessions, infrequent turns, or agentic workflows that may pause |

The 5-minute TTL refreshes automatically on each cache hit at no additional cost. The 1-hour TTL costs more to write but survives longer gaps between requests.

### Disabling caching

Set `PROMPT_CACHE_ENABLED=false` in the environment. This returns the model as a plain `BaseChatModel` without the cache binding.

---

## Architecture

### How caching is applied

The `ModelFactory.get_model()` method applies caching as the final step after constructing the model:

```
ModelFactory.get_model(config)
  → _create_converse_bedrock_model(...)  # or _create_anthropic_bedrock_model
  → _apply_prompt_caching(llm, provider)
      → llm.bind(cache_control={"type": "ephemeral", "ttl": "5m"})
      → returns RunnableBinding wrapping the original model
```

The `.bind(cache_control=...)` pattern attaches the cache configuration as a default kwarg on every subsequent `invoke`/`ainvoke`/`stream` call. This is consumed by `ChatBedrockConverse._generate()` which calls `_apply_cache_points()` to insert `cachePoint` blocks into the Bedrock Converse API request.

### What gets cached

On each API call, `_apply_cache_points()` adds a `{"cachePoint": {"type": "default"}}` block to:

1. **System messages** — The static system prompt (stable across all requests using the same model config)
2. **Tool definitions** — MCP tool schemas (stable for the session lifetime)
3. **Last message content** — The conversation prefix up to the current turn (grows each turn, but the prefix is shared with the previous request)

### Return type change

Because `.bind()` returns a `_ChatModelBinding` (which is a `Runnable`, not a `BaseChatModel`), the return type of `get_model()` is `BaseChatModel | Runnable`. Downstream consumers that type-check the model must accept both types.

### Provider scope

Prompt caching only applies to **Bedrock** models. OpenAI and Google providers return an unmodified `BaseChatModel`. Within Bedrock, caching is supported by Anthropic Claude and Amazon Nova models. If the model doesn't support caching, Bedrock silently ignores the `cachePoint` blocks.

---

## Token Economics

### Cost comparison (Claude Sonnet 4 on Bedrock)

| Token type | Cost per MTok | When it occurs |
|-----------|---------------|----------------|
| Base input (uncached) | $3.00 | Tokens after the last cache breakpoint |
| Cache write (5m TTL) | $3.75 | First request that establishes the cache |
| Cache write (1h TTL) | $6.00 | First request with 1h TTL |
| Cache read | $0.30 | Subsequent requests hitting the cache |

### Break-even analysis

A cache write pays for itself after **just 2 cache reads** (for 5m TTL):
- Write: 1.25x = pays 0.25x extra over base
- Each read saves: 0.9x (base - read cost)
- Break-even: 0.25 / 0.9 ≈ 0.28 → first cache read already saves more than the write premium

For multi-turn conversations (the primary use case), caching is net-positive from the second turn onward.

### Minimum cacheable tokens

| Model | Minimum tokens |
|-------|---------------|
| Claude Sonnet 4.6, 4.5, 4 | 1,024 |
| Claude Opus 4.7, 4.6, 4.5 | 4,096 |
| Claude Haiku 4.5 | 4,096 |

If the prefix is shorter than the minimum, Bedrock processes it normally without caching (no error, just no cache activity).

---

## Observability

### Log output

When cache tokens are present in a response, the service logs:

```
INFO  Prompt cache: read=4523 tokens, write=0 tokens, uncached=127 tokens
```

- **read** — Tokens served from cache (cheap)
- **write** — Tokens written to a new cache entry (first occurrence)
- **uncached** — Tokens after the last cache breakpoint (normal price)

### Token usage metadata

The `_extract_token_usage()` method in `BaileyAgentService` returns a dict with:

```python
{
    "prompt": 127,           # Uncached input tokens
    "completion": 342,       # Output tokens
    "total": 469,            # prompt + completion
    "cache_read": 4523,      # Tokens read from cache
    "cache_creation": 0,     # Tokens written to cache
}
```

Total input tokens = `cache_read + cache_creation + prompt`.

### Monitoring cache effectiveness

A healthy caching setup shows:
- `cache_creation > 0` on the **first turn** of a conversation
- `cache_read > 0` on **subsequent turns** (growing as conversation grows)
- `cache_read >> prompt` on most requests (prefix is much larger than new content)

If you see `cache_creation` on every request, the prefix is changing between requests (investigate what's breaking the cache — e.g., timestamps in system prompt, tool definition changes).

---

## Interaction with other features

### Extended thinking

Extended thinking (`thinking_budget_tokens`) and prompt caching work together. Thinking blocks in prior turns are part of the cached conversation prefix on subsequent turns. No special configuration needed.

### Tool discovery mode

When `use_tool_discovery: true` is set, only meta-tools (`search_tools`, `call_tool`) are bound to the model — not the full tool catalog. This means the tool definitions are small and stable, which is ideal for caching. The full tool schemas are loaded dynamically via the meta-tools and appear in message content (which is also cached on subsequent turns).

### Health safety evaluator

The health safety evaluator uses its own `ChatBedrockConverse` instance (not the main agent's model). If that instance is also created via `ModelFactory`, it will also have caching enabled. The evaluator's system prompt is stable across requests, so it benefits similarly.

---

## Limitations

| Constraint | Impact |
|-----------|--------|
| Bedrock only | OpenAI and Google models are unaffected |
| No automatic caching on Bedrock | Must use explicit `cachePoint` blocks (handled automatically by `_apply_cache_points`) |
| Minimum token threshold | Very short system prompts won't be cached |
| Cache isolation per AWS organization | Different AWS accounts don't share cache entries |
| 4 breakpoints max per request | `_apply_cache_points` uses 3 (system, tools, last message) — leaves room for 1 more if needed |
| Cache invalidated by tool changes | Adding/removing tools mid-session breaks the tools cache prefix |

---

## Troubleshooting

**Cache metrics are always zero:**
- Verify `PROMPT_CACHE_ENABLED` is not set to `false`
- Check the model supports caching (Anthropic Claude or Amazon Nova)
- Ensure the system prompt + tools exceed the minimum token threshold (1,024 for Sonnet)

**Cache writes on every request (no reads):**
- Something is changing in the prefix between requests
- Check for timestamps or per-request data in the system prompt
- Verify tool definitions aren't being shuffled or modified

**Model type errors downstream:**
- `get_model()` now returns `BaseChatModel | Runnable`
- Ensure `isinstance` checks accept both types
- The bound model still supports all LangChain interfaces (`invoke`, `ainvoke`, `bind_tools`, etc.)
