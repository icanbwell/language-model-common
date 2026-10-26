# Persist learned MCP tool titles in a shared store

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1306; follow-up to BAI-1301 (PR #149)
* Date: 2026-10-26

## Context and Problem Statement

A tool reached through the `call_tool` discovery meta-tool is never bound as a
LangChain tool, so its MCP title is only learned from a `search_tools` result
(`title`) or a `call_tool` result (`tool_title`). PR #149 (BAI-1301) added a
bounded, process-wide `LearnedTitleCache` so a later request on the same pod
renders the start progress line with the title an earlier request already saw.

That cache is per pod. The service runs as multiple pods that restart and
scale, so a new pod starts empty and the first `call_tool` start line for a
tool renders the humanized fallback (for example "Get Vitals") until that pod
itself sees a result containing the title. The end event is already correct;
the effect is cosmetic.

`McpToolListStore` does not cover this. It persists each server's `list_tools`
output, which for a discovery server is only `search_tools` and `call_tool`.
Target tools are never listed, so their titles are not in that store.

## Decision Drivers

* Titles for `call_tool` targets must survive pod restarts and be visible to
  every pod.
* `call_tool` start events are on the streaming hot path; a persistent read
  must not add per-event latency or let a store outage break a stream.
* Titles are catalog metadata (tool name to title template). No PHI, no tenant
  or user data may be stored.
* No new technology: reuse the `py-key-value-aio` Mongo store already used by
  `McpToolListStore` and the prompt store.
* Titles can change on the server, so stored entries must expire.

## Considered Options

### Option A: Keep the in-process cache only (PR #149)

No new storage. A cold pod renders the humanized fallback for a tool until it
sees a title. Rejected as the end state, acceptable as the baseline.

### Option B: Seed the in-process cache from `McpToolListStore`

Rejected. The store holds `list_tools` output, which does not contain
`call_tool` targets (see Context).

### Option C: Persist learned titles in a dedicated store, read-through into the in-process cache (chosen)

A `LearnedTitleStore` on the existing `py-key-value-aio` Mongo store, with its
own collection and TTL. The in-process `LearnedTitleCache` stays as the
synchronous first level the mapper reads.

### Option D: Read the persistent store on every `call_tool` start event

Rejected. Adds a Mongo read to every start event, and still needs a local
cache in front of it to be acceptable.

## Decision Outcome

Shipped in PR #149 together with the in-process cache, so the PR delivers the
complete fix.

Option C.

### Design

* `LearnedTitleStoreProtocol` (`put_title(*, tool_name, title)`,
  `get_title(*, tool_name)`) and a `LearnedTitleStore` implementation backed by
  a `py-key-value-aio` store. Consumers depend on the protocol.
* Entries are keyed by tool name, store only the title string, and carry a TTL
  (default 24h, configurable). The same length caps as `LearnedTitleCache`
  apply before anything is written.
* **Write-through:** when `ToolEventHandler` learns a title from a
  `search_tools` or `call_tool` result, it also writes it to the store. A write
  failure is logged and swallowed.
* **Read-through:** `ToolEventHandler.handle_tool_start` is async. For a
  `call_tool` start whose target is not in the request mapper or the
  in-process cache, it awaits one `get_title` under an explicit timeout and, on
  a hit, fills the in-process cache before the line renders. A miss, timeout,
  or store error falls back to the humanized name.
* A name with no stored title is remembered as a miss for 60 seconds, so an
  untitled tool does not cost a store read on every start event.
* Result: at most one persistent read per tool name per pod after a cold
  start; steady state is served from memory.
* Registered as a singleton in `container_factory.py`. New settings are
  properties on `LanguageModelCommonEnvironmentVariables` (database,
  collection, TTL, read timeout), with defaults that match the existing stores.
* Precedence is unchanged: static config, then live tool titles, then learned
  titles (memory, then store), then the humanized name.

### Risks and mitigations

* **Store outage or slowness:** bounded by the read timeout; failures degrade
  to the humanized name and never block or fail the stream.
* **Stale titles:** the TTL bounds staleness; a fresh `search_tools` or
  `call_tool` result overwrites the stored value.
* **Cross-catalog name collision:** unchanged from PR #149 — learned titles
  apply only to the `call_tool` target, never to directly bound tools.
* **Tenant isolation:** only tool-name to title-template pairs are stored; no
  request arguments, results, or identifiers. Placeholders are substituted per
  call from that call's own arguments.

### Consequences

* Good: cold pods render the right title on the first `call_tool` start line
  once any pod has seen it.
* Good: no new datastore technology.
* Bad: one more Mongo collection and a small read on first use per tool name
  per pod.

## Test Plan

* Round trip across two handler instances sharing one `MemoryStore`.
* TTL expiry returns a miss and falls back to the humanized name.
* Store read error and read timeout fall back without raising.
* Store write error is swallowed.
* Length caps are enforced before writing.
* Precedence: static config and live tool titles win over stored titles.
* All new public methods are called with keyword arguments.

## Related Work

* BAI-903: `call_tool` discovery meta-tool and per-request title learning.
* BAI-1301 / PR #149: in-process `LearnedTitleCache`.
* `McpToolListStore`: existing `py-key-value-aio` Mongo usage this reuses.
