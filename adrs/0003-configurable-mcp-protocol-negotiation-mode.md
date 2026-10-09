# Configurable MCP protocol negotiation mode (`legacy` | `auto`)

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1118
* Date: 2026-10-08

## Context and Problem Statement

Every MCP client session in this library is created by `create_mcp_session`
(`languagemodelcommon/mcp/mcp_client/session.py`), which builds a bare
`mcp.ClientSession` over `streamable_http_client`. Callers then run
`session.initialize()` (directly, or through `open_initialized_mcp_session`,
which `McpSessionPool` and `tool_invocation` use). That sends the legacy
`initialize` request at the SDK's `LATEST_HANDSHAKE_VERSION`, `2025-11-25`
with `mcp` 2.2.0, and accepts any of `2024-11-05` through `2025-11-25` from
the server. The SDK's `server/discover` probe, and with it the modern
protocol (2026-07-28 and later), is never used.

Consequences today:

- Clients on this library (baileyai, baileyai-skills-service,
  mcp-fhir-agent) cannot reach the modern protocol even where the server
  supports it. All three repos are on `mcp` 2.x, and `mcp` 2.2.0's server
  already routes requests with a modern `MCP-Protocol-Version` header to a
  separate modern handler (`streamable_http_manager.py`).
- baileyai-skills-service's tool-catalog runs stateless (`FASTMCP_STATELESS:
  'true'` in `.helm/common.values.yaml`, no per-environment override). In the
  legacy stateless path the server builds each request's `Connection` with no
  client info or capabilities, so a server tool or interceptor cannot learn
  what the client supports. The modern path carries capabilities in every
  request envelope. This came up while deciding how destructive-action
  confirmation should treat different callers (BAI-1114, closed without a
  change).

This ADR decides how to let the library use the modern protocol without a
flag-day cutover across repos.

## Decision Drivers

- No behavior change by default. The confirmation flow (SEP-2322
  `InputRequiredResult`, baileyai ADR 006) is a HIPAA-relevant gate and must
  not change implicitly.
- Rollout per environment, using what the repos already have: `.helm` common
  and per-environment values files.
- A server that does not speak the modern protocol must keep working, with no
  per-call failures.
- One place decides the handshake. Today baileyai's `tool_catalog_client.py`
  calls `session.initialize()` directly at three sites, bypassing the
  library's retry logic in `open_initialized_mcp_session`.
- Use public SDK API where possible.

## Considered Options

### Option A: Do nothing

Stay on the legacy handshake. No risk, but servers keep losing client
capability signals in stateless mode, and a later move has the same cost with
more callers.

### Option B: Env var selects a modern version to pin

The SDK supports `mode="2026-07-28"`, which adopts that version without a
probe. Rejected: a pinned version has no fallback. A server that does not
support it fails every call, so every server and every client would have to
move in lockstep.

### Option C: Env var selects `legacy` or `auto` (chosen)

`auto` runs the SDK's `negotiate_auto`: probe `server/discover`, adopt the
result if the server answers, otherwise fall back to the legacy `initialize`
handshake. A modern-only server that shares no version raises, and a legacy
server falls back. The default stays `legacy`, which is the current behavior.

## Decision Outcome

Chosen option: **C**.

### Design

1. **Setting.** Add a property to `LanguageModelCommonEnvironmentVariables`,
   for example `mcp_protocol_negotiation_mode`, reading
   `MCP_PROTOCOL_NEGOTIATION_MODE`. Accepted values: `legacy` (default) and
   `auto`. An unrecognized value logs a warning and uses `legacy`, matching
   how `mcp_tool_heartbeat_interval_seconds` treats an invalid value. Callers
   read it through the injected env-vars class, not `os.environ`.
2. **One handshake function.** Add `negotiate_session(session, *, mode)` in
   `session.py`. `legacy` calls `session.initialize()`. `auto` calls the
   SDK's probe-then-fallback. `open_initialized_mcp_session` calls it in place
   of `session.initialize()`, so the existing connect retry with backoff
   covers it and retries stay scoped to session establishment only.
3. **Callers.** Pass the mode into `open_initialized_mcp_session`, with
   `legacy` as the default argument so existing call sites are unchanged.
   `McpSessionPool` takes the mode in its constructor and uses it for every
   pooled session, so the `negotiation_mode` argument of `call_mcp_tool_raw`
   and `mcp_tool_to_langchain_tool` selects the handshake only for the one-shot
   path taken when no pool is given. Passing a mode that conflicts with the
   pool's raises `ValueError` instead of being silently ignored.
   Document and expose a helper so baileyai's three direct
   `session.initialize()` calls can move onto it in a follow-up change in that
   repo.
4. **Rollout.** Set `MCP_PROTOCOL_NEGOTIATION_MODE` in each repo's `.helm`
   values: servers first (verify they answer `server/discover` or reject it
   cleanly), then clients, dev before staging before prod. Reverting an
   environment is a one-value change.
5. **Not in scope.** Per-server override in the MCP server config, pinning a
   modern version, and any change to confirmation behavior.

### Risks and mitigations

- **Private SDK API.** `negotiate_auto` lives in `mcp.client._probe` and is
  only re-exported to `mcp.client.client.Client`. Importing it ties the
  library to a private module. Alternatives: build the session through the
  SDK's public `Client(mode="auto")` (a larger change, since this library
  owns the transport, pool and callbacks), or reimplement the probe here
  against public `ClientSession` methods (`send_discover`). Decided: import
  `negotiate_auto` privately, isolated inside `negotiate_session`, with a
  contract test (`test_sdk_probe_symbol_is_importable`) that fails if an `mcp`
  upgrade moves the symbol.
- **Extra round trip against legacy servers.** In `auto`, a legacy server
  costs one rejected `server/discover` before the handshake. Pooled sessions
  amortize this. Callers that open a fresh session per call (baileyai's
  `ToolCatalogMcpClient`) pay it each time. Caching the negotiated era per
  URL would need a shared store, not an in-process cache (multi-worker
  deployments), so it is not proposed here. Measure before enabling in prod.
- **Pool keys.** `McpSessionPool` keys on `(url, headers)`. With one mode per
  process this stays correct. If a per-server mode is added later, the mode
  must become part of the key so a legacy and a modern session to the same
  URL are never mixed.
- **Guard-tool flow on the modern path.** The `InputRequiredResult` resume
  path (`call_mcp_tool_raw`, `allow_input_required`) is built and tested
  against the legacy handshake. It must be tested against a modern server
  before any environment above dev uses `auto`.
- **Server-side errors.** A server that returns an unexpected error to
  `server/discover` instead of a clean rejection would fail the probe's
  fallback. Verify against baileyai-skills-service, mcp-fhir-agent and any
  third-party servers in the catalog.

### Consequences

- Good: no behavior change until an environment opts in, and a one-value
  revert after.
- Good: the handshake decision lives in one function, which also fixes the
  direct `initialize()` calls bypassing retry.
- Bad: new dependency on an SDK probe function that is not public API today.
- Bad: one extra round trip per fresh session against legacy servers when
  `auto` is on.
- Bad: the mode is per process, not per server, until a follow-up adds an
  override.

## Test Plan

- Unit: `negotiate_session` with `legacy` calls `initialize()` only; with
  `auto` adopts a discover result, falls back on a legacy rejection, and
  raises on a modern-only server with no shared version. Parametrized over the
  setting's accepted, empty and invalid values.
- Unit: `open_initialized_mcp_session` retries on a transient failure in both
  modes, and a cancellation propagates unretried.
- Integration (local): connect to baileyai-skills-service and mcp-fhir-agent
  in both modes; confirm a guard-tool pause and resume completes in `auto`.
- Contract check: the SDK symbol used for the probe is importable under the
  pinned `mcp` range.

## Open Questions

- ~~Public `Client(mode="auto")`, a thin reimplementation of the probe, or a
  private import?~~ Resolved in the implementation: private import, lazy and
  isolated in `negotiate_session` (see Risks).
- Is a per-server override needed in the first release, or can it wait for a
  server that cannot handle `auto`?
- Do any third-party MCP servers in the catalog mishandle `server/discover`?

## Related Work

- baileyai `adrs/006-mcp-guard-tool-elicitation-support.md` (guard-tool
  bridge, built on the legacy handshake).
- baileyai-skills-service `ConfirmationMcpCallInterceptor` and its
  stateless deployment (`.helm/common.values.yaml`).
- BAI-1114 (closed): destructive-hint handling discussion that surfaced the
  stateless capability gap.
