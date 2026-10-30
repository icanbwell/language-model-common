# Validate tool output from the shared tool-list cache

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1388
* Date: 2026-10-30

## Context and Problem Statement

`ClientSession.call_tool` (mcp 2.2 and 2.3) calls the public
`validate_tool_result(name, result)` after every successful result. When the tool
name is not in the session's private output-schema map, it first runs
`await self.list_tools()` to learn the schema. A fresh session has an empty map,
and `_make_execute_tool` opens a fresh one-shot session per call (the tool
catalog in baileyai-skills-service does not pool). So each data-tool call pays a
downstream `tools/list` (p50 0.94 s in prod: 149 of 150 sampled data-tool spans
had one) even for a tool that declares no output schema, where the check is a
no-op after the lookup.

The shared `ToolListCache` (Mongo-backed `McpToolListStore` in production, keyed
by server URL, shared across pods) already holds every tool's full definition,
including `output_schema`, and `_tool_supports_tasks` already reads per-tool
data from it.

## Decision Drivers

* Remove the per-call `tools/list` without ever skipping a validation the SDK
  would perform.
* Flag off must be byte-for-byte today's behavior; opt-in per consumer.
* No in-process data cache (multi-worker production; org rule). The Mongo-backed
  cache is the cache.
* Depend on the SDK's public surface where possible; fail loudly if it changes.
* `create_mcp_session` returns a `ClientSession` from either the bare constructor
  or the SDK `Client` (when a negotiation mode is set), so subclassing
  `ClientSession` is not available.

## Considered Options

### Option A: Override the public `validate_tool_result` on the session instance (chosen)

A small `CachedOutputSchemaValidator` replaces the method on the instance. It
reads the schema from a `ToolOutputSchemaSource` (a `ToolListCache` entry), and
falls back to the SDK's own method whenever the cache cannot answer.

* Good: only public API (`validate_tool_result`), documented in the SDK, same
  signature in 2.2 and 2.3.
* Good: works for pooled sessions, one-shot sessions and `Client`-created ones.
* Good: the fallback is the unmodified SDK method, so cache miss, cache error
  and untrusted ad-hoc servers behave exactly as today.
* Bad: re-implements the jsonschema check (about 25 lines), so it can drift from
  the SDK. Mitigated by parity tests that run the real SDK against the same
  inputs with the flag on and off, and a test that fails if the method is renamed.
* Bad: the compiled validator is rebuilt per call instead of cached on the
  session (about 60x validation cost, still small next to a 0.9 s round trip).
  Not cached in-process on purpose.

### Option B: Seed the SDK's private `_tool_output_schemas` from the cache

Pre-populate the session's private dict so the SDK's own lookup finds the tool.

* Good: keeps the SDK's exact validation code, including its compiled-validator
  cache.
* Bad: writes a private attribute (`_tool_output_schemas`, with the derived
  `_tool_output_validators` that must stay consistent with it); a rename or
  reshaping in a future release silently turns the flag into a no-op or, worse,
  skips validation.
* Bad: still needs the same cache read, staleness and refetch logic.

### Option C: Skip validation when the server is trusted

* Bad: silently drops a check the SDK performs today. Rejected.

### Option D: Pool sessions in the catalog so the lookup is paid once per session

* Out of scope here: it is a separate change in baileyai-skills-service
  (session pooling) with its own lifecycle and auth implications. It does not help
  the first call per session, and is not mutually exclusive with this change.

## Decision Outcome

Option A, behind `MCP_VALIDATE_TOOL_OUTPUT_FROM_CACHE` (default off, property
`mcp_validate_tool_output_from_cache`). Consumers pass it as
`validate_output_from_tool_list_cache` to `mcp_tool_to_langchain_tool` and
`call_mcp_tool_raw`. Behavior once on, after the session is obtained and before
`session.call_tool`:

1. Cache has the tool, no output schema: no lookup, no validation (what the SDK
   does after its lookup).
2. Cache has a schema: validate locally with the SDK's rules and messages. On
   failure (including missing structured content, an invalid schema, an
   unresolvable `$ref`), re-list tools once on this session, update the cache
   entry, and revalidate against the fresh schema before raising. A schema that
   changed since it was cached therefore cannot wrongly fail a call.
3. No cache entry, tool absent from the entry, no cache given (ad-hoc servers),
   cache read error, or refetch failure: delegate to the SDK's own method.
   Validation is never skipped.

### Tenancy and trust

`ToolListCache.make_key` returns the server URL only (the auth header argument is
ignored), so a lookup can only read the entry of the server the call is addressed
to, never another server's. Output schemas describe the server's tool, not a
caller; entries are written by the same discovery path that already feeds
`_tool_supports_tasks` and tool binding. Untrusted ad-hoc servers are never
written to the cache, so for them the cache returns nothing and the SDK path runs.

Residual risk: a cached entry that says a tool has no schema, while the server
has since started declaring one, skips validation for that tool until the entry is
replaced (reload, deploy, a failed call that invalidates it, or TTL where set).
The catalog's store has no TTL. Output validation is a conformance check on the
server's own response, not an authorization boundary, and the only way to detect
this case would be the `tools/list` this change removes.

### Version notes

mcp 2.3 treats an explicit `structuredContent: null` as a value to validate;
mcp 2.2 rejects it as missing. The local check follows 2.3 (the version
consumers pin). Both fail closed for schemas that do not allow null.

## Consequences

* One downstream round trip fewer per call on a cache hit, for every tool.
* A new `cached_output_validation` module; `tool_invocation` gains one helper and
  a keyword argument that defaults to the previous behavior.
* The check is a copy of SDK logic: revisit when the minimum mcp version moves.
* Parity tests are pinned to the installed SDK and are skipped loudly (not
  passed) below the versions whose behavior they assert.
