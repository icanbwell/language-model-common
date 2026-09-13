# Extend prompt caching to tool definitions and conversation history

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-706 (this design); follow-up to BAI-704 (Bedrock quota increase)
* Date: 2026-09-13

## Context and Problem Statement

On 2026-09-13, baileyai in `client-sandbox` surfaced a rate-limit fallback
message to a user mid-conversation. Root cause: AWS Bedrock returned a 429
(`Too many tokens, please wait before trying again`) for
`us.anthropic.claude-sonnet-4-20250514-v1:0` on the third of three
back-to-back model calls within a single tool-calling turn (initial call →
`search_clinical_notes` → `list_document_references` → `search_clinical_notes`
again). `language-model-common`'s converter does not retry once any part of
the response has streamed to the client — retrying would restart the whole
graph from the original messages (no checkpointer is configured; see
"Related work" below) and would duplicate output already shown to the user,
or worse, re-invoke non-idempotent tools already executed in that turn (e.g.
`save_fhir_resource`).

CloudWatch data on the account showed the incident's peak usage
(~42,600 tokens/min, 28 requests/min) was well under the account's
provisioned quota (200,000 TPM / 200 RPM) — this points at **Bedrock's
sub-minute burst smoothing**, not the full-minute quota, as the proximate
trigger. A quota increase (BAI-704, submitted, pending someone with
`servicequotas:RequestServiceQuotaIncrease` access) raises the ceiling but
doesn't change the burst pattern: 2-3 large calls firing within seconds of
each other because every call in the tool loop resends the full growing
conversation from scratch.

Reducing how many tokens each of those calls actually costs is a more direct
lever on the burst itself. This ADR evaluates extending prompt caching —
already used for the system prompt — to (1) tool definitions and (2) the
growing conversation/tool-call history within a multi-round tool-calling
turn.

### Current state (see `docs/prompt-caching.md`)

- System prompt blocks marked `"cache": true` in `PromptConfig` get
  `cache_control: {"type": "ephemeral"}` (`converters/langgraph_to_openai_converter.py:1234-1236`,
  mirrored in baileyai's `graph_builder.py:186-188`). Enabled today for two
  blocks in prod's `bailey.json`.
- Tool definitions (~44 registered tools for the FHIR agent plugin — the
  `ListingFilterMiddleware` log line in the incident lists all of them) get
  **no explicit `cache_control`**. `docs/prompt-caching.md` claims tools are
  "cached by provider alongside stable prefix," which would be true *if*
  Anthropic's caching semantics extend a breakpoint backward through
  everything preceding it in wire order (tools → system → messages) —
  worth confirming empirically before relying on it (see Open Question 1).
- Conversation/tool-call messages are explicitly **never** cached
  (`docs/prompt-caching.md:36`).
- `bailey_agent_services.py:_extract_token_usage` logs
  `"Prompt cache: read=%d tokens, write=%d tokens, uncached=%d tokens"`
  whenever `cache_read` or `cache_creation` is nonzero.

### Open Question 1 — is system-prompt caching actually working today?

A search of `baileyai-client-sandbox` logs for `"Prompt cache"` over an 8-hour
window (spanning the incident) returned **zero matches**, despite `cache:
true` being set in prod config. Either:

a. caching is not actually taking effect for the `ChatAnthropicBedrock` path
   in this environment (e.g. a version mismatch, a code path that drops
   `cache_control` before the request is sent), or
b. `usage_metadata.input_token_details` doesn't carry `cache_read`/
   `cache_creation` in the shape this code expects for this provider/model,
   and the log line is silently never firing even though caching works.

**This needs to be resolved before (or as the first phase of) doing any of
the work below.** Extending caching to more content is not useful if the
existing mechanism isn't verified to work, and if the failure is (b), we're
also flying blind on whether *any* of our caching is paying off today.

## Decision Drivers

- Reduce per-call token volume in multi-round tool-calling turns, directly
  addressing the burst pattern that triggered the incident.
- Reduce cost and latency (cached reads are billed and processed more
  cheaply/faster than fresh input tokens).
- No new dependency or vendor — this uses a capability already present in
  the pinned `langchain_aws` version and already partially wired up.
- Minimize risk to a shared library consumed by every service on
  language-model-common.

## Options Considered

### Option A — Do nothing beyond the quota increase (baseline)

Rely solely on BAI-704. Zero implementation cost. Leaves the burst pattern
itself unchanged — a busier client-sandbox session, or a tenant with a
longer tool-calling chain, can reproduce the same throttle at a higher
absolute token count. Doesn't reduce cost or latency either.

### Option B — Cache tool definitions

Tag the tool-definitions payload with a cache breakpoint so the ~44 tool
schemas (likely the single largest static chunk of every request, larger
than the system prompt) are read from cache on the 2nd/3rd/... call within a
turn instead of resent in full.

**Mechanism:** `create_react_agent` (LangGraph prebuilt, used by
`GraphBuilder.create_graph_for_llm_async`) calls `model.bind_tools(tools)`
internally with no cache hook. `ChatAnthropicBedrock.bind_tools()`
(`langchain_aws/chat_models/bedrock.py:1310`) converts tools via
`convert_to_anthropic_tool()` with no exposed `cache_control` kwarg for
tools specifically (message-level `cache_control` *is* exposed, via
`_apply_cache_control_to_messages`). To cache the tool list we would need to
pre-bind: convert tools ourselves, tag the last tool dict with
`cache_control`, and pass a pre-bound model into `create_react_agent`
(confirmed via `_should_bind_tools` in
`langgraph/prebuilt/chat_agent_executor.py:173` that a model whose bound
tools already match the `tools=` argument is not re-bound — so this
composes cleanly with the existing prebuilt helper).

**Trade-off:** duplicates a slice of `convert_to_anthropic_tool`'s logic
outside the library boundary — a drift risk if `langchain_aws` changes that
conversion. Contained to `graph_builder.py`; no new dependency.

### Option C — Cache tool definitions + conversation/tool-call history

In addition to Option B, place a cache breakpoint at the end of the message
list before each subsequent model call within a multi-round tool-calling
turn (and/or at the end of each completed turn, for cross-turn reuse) — the
pattern used by agentic coding tools such as Claude Code: cache
"everything so far," pay full price only for the newest delta.

This is the mechanism that most directly targets the incident: within one
turn, calls 2 and 3 would read most of their input from cache instead of
resending the entire growing history, cutting the actual token volume (and
therefore TPM pressure) of the exact burst that tripped the 429.

## Pros and Cons of Caching Non-System (User/Assistant/Tool) Messages

This is the crux of Option C and deserves its own analysis, since it's a
qualitatively different decision from caching the system prompt (stable,
provider-controlled, no PHI by construction) or tools (stable, no PHI).

**Pros**

- **Directly targets the burst.** Every call after the first in a
  tool-calling turn currently resends the *entire* prior history at full
  price. Caching it converts that into a cheap read for everything except
  the newest tool result — the single biggest lever available for reducing
  the token volume of exactly the pattern that caused this incident.
- **Compounding cost/latency win**, on top of the TPM benefit: cached reads
  are billed at a fraction of fresh input tokens and process faster
  (lower time-to-first-token), which matters for a chat product.
- **Proven pattern.** This is exactly how Claude Code and similar agentic
  tools handle multi-step tool loops — cache the conversation-so-far after
  each turn, pay full price only for the increment. Not a novel technique
  for this org to be pioneering; well understood.
- **Composes with Options A and B** rather than replacing them.

**Cons / risks**

- **Cache invalidates on any exact-prefix change.** `SmartHistoryManager` /
  `ConversationHistoryManager` trims older messages once a conversation
  exceeds `max_messages`/`max_tokens`. Trimming rewrites the prefix and
  busts the cache at exactly the point (long conversations) where caching
  matters most. The breakpoint needs to be re-anchored immediately after
  each trim, not placed at a fixed offset from the start.
- **Anthropic's 4-breakpoint-per-request limit.** System prompt already
  uses 2 (per current `bailey.json`); adding tools (Option B) and history
  (Option C) needs to fit in the remaining budget — likely requires
  collapsing the system prompt to a single breakpoint to leave headroom for
  tools + a rolling history breakpoint.
- **TTL sensitivity.** Standard cache TTL is short (~5 minutes, extendable
  to 1 hour at a higher write cost). Within one agentic tool-calling turn
  (seconds apart) hit rate should be very high — the exact case this
  incident represents. Across separate user turns (a human takes minutes to
  type a reply), a cache miss is plausible and reduces the benefit for
  slow-paced conversation, though it never makes things *worse* than today
  (a miss just falls back to a normal write).
- **Write premium.** The first request against a new/moved breakpoint pays
  a write premium (~25% more for the 5-minute TTL) for that portion. Net
  benefit depends on reads outweighing writes — true for multi-round tool
  loops (2+ reads against 1 write per turn), less clearly true for
  single-shot conversations that never call a tool.
- **PHI exposure surface is larger.** Conversation messages (patient
  questions, clinical note contents, tool results containing FHIR resources)
  are far more likely to contain PHI than the system prompt or tool
  schemas. Before caching this content, we should get explicit confirmation
  that AWS Bedrock's prompt-cache storage falls within the existing HIPAA
  BAA / data handling commitments already covering Bedrock inference calls
  — this is very likely already true (it's the same backend, same
  encryption-at-rest posture, transient TTL-bound storage), but it should be
  a stated, verified assumption rather than an implicit one, given this
  org's PHI stance. This is a call for security/EA sign-off, not a reason to
  block the design.
- **Multi-tenant isolation.** Cache keys are the literal token prefix, so
  isolation is a structural property (a different patient's conversation is
  a different prefix, full stop) rather than something we configure — but
  it's worth a test asserting two concurrent sessions never share a cache
  hit, rather than relying on that being true by construction.
- **Added complexity.** Cache-lifecycle logic (where to place/move
  breakpoints, how it interacts with trimming/compaction) adds real
  surface area to history management code that today is comparatively
  simple.

## Recommendation

Sequence the work rather than doing it all at once:

1. **Phase 0 — verify the existing mechanism.** Resolve Open Question 1
   before extending scope. If system-prompt caching isn't actually taking
   effect today, fix that first; if it's a telemetry bug, fix the metric
   extraction so we have ground truth for phases 1-2.
2. **Phase 1 — Option B (tool caching).** Contained, no new dependency,
   likely the single largest remaining win (tool schemas probably outweigh
   the system prompt in token count). Lower risk than Option C.
3. **Phase 2 — Option C (history caching), scoped to within-turn only
   first.** Anchor a breakpoint at the end of the message list before each
   subsequent model call in the *same* tool-calling turn — this is the
   part that directly addresses burst throttling and has the least TTL
   risk (calls are seconds apart). Cross-turn (user-to-user) history
   caching, and its interaction with `SmartHistoryManager` trimming, is a
   larger follow-up worth its own validation once Phase 2 lands.

Each phase should ship with a test asserting cache hits actually occur
(via `cache_read`/`cache_creation` on a real or recorded response), not just
that `cache_control` is present in the outgoing request — Phase 0 exists
precisely because the latter can be true while the former silently isn't.

## Consequences

- Positive: reduced per-call token volume for tool-heavy turns, reduced
  cost and latency, reduced (but not eliminated) recurrence risk of the
  TPM-burst 429 pattern independent of the quota increase.
- Negative: added complexity in history management; a breakpoint-budget
  constraint to manage as more content wants caching; a PHI-handling
  assumption to get explicit sign-off on for Phase 2.
- Neutral: does not require Tech Design Review under the org's "is it new
  technology" test — this extends an already-adopted capability
  (`cache_control` on `ChatAnthropicBedrock`) rather than introducing a new
  vendor, datastore, or pattern.

## Open Questions

1. Is system-prompt caching actually taking effect in client-sandbox today?
   (See "Open Question 1" above — blocks everything else.)
2. Does marking only the system-prompt block with `cache_control` already
   implicitly cache the preceding tools block per Anthropic's prefix-caching
   semantics, as `docs/prompt-caching.md` currently claims? If so, Phase 1
   may already be partially realized and just needs verification, not new
   code.
3. What's the actual breakpoint budget once Phase 1 and 2 are both in play,
   and does it require collapsing the two current system-prompt breakpoints
   into one?
4. Confirm with security/EA that Bedrock prompt-cache storage is covered by
   existing PHI/BAA commitments before Phase 2 ships.

## Related Work

- BAI-704 — Bedrock Sonnet 4 V1 TPM/RPM quota increase for the same
  incident (submitted; blocked on `servicequotas:RequestServiceQuotaIncrease`
  access in the `cloud-lead-prod` account).
- A related but separate lever discussed and **not** part of this ADR: a
  durable, shared LangGraph checkpointer (Postgres/Redis-backed) would make
  mid-stream retry safe by allowing resumption from the last completed node
  instead of restarting the graph. baileyai's compiled graph currently
  passes `checkpointer=None`
  (`baileyai/services/bailey_agent_services.py:266-273`). That's a larger
  infra change (new persistence dependency, likely its own Tech Design
  Review) and is out of scope here; caching reduces the *frequency* of the
  problem, a checkpointer would change what happens *when* it still occurs.
