# Keep tool display titles in Mongo only and pre-warm them from the tool catalog

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers, CTO
* Ticket: BAI-1732; supersedes the in-process cache parts of ADR 0007 (BAI-1306, BAI-1301)
* Date: 2026-11-11

## Context and Problem Statement

baileyai runs 10 to 100 pods. The CTO decision is: no in-process or in-memory
caching; everything cached lives in Mongo. The tool display-title path broke
that rule in two places: the process-wide `LearnedTitleCache` LRU used by
`ToolDisplayNameMapper`, and the `_misses` and `_recent_writes` dicts in
`LearnedTitleStore`.

Separately, titles were only written to the store from `search_tools` listings.
A tool the model calls directly through `call_tool` (`load_skill`,
`search_skills`) never appears in a listing, never gets a stored title, and its
start line shows the humanized fallback ("Load Skill").

## Decision Drivers

* No process-local data caches (CTO decision); all pods must agree.
* The `call_tool` start event is on the streaming hot path: a store problem must
  never break or stall a stream.
* Titles are catalog metadata (tool name to title template): no PHI, no tenant
  or user data.
* No new technology: reuse the existing `py-key-value-aio` Mongo store.

## Considered Options

* A: Keep the process-wide LRU and only add the warm. Rejected: violates the
  decision.
* B: Remove the LRU; read Mongo on a `call_tool` start with no title; hold the
  result only in the request's own mapper; pre-warm Mongo from the catalog.
  Chosen.
* C: Read Mongo on every `call_tool` start with no request-level memory.
  Rejected: an untitled tool would cost one read per call.

## Decision Outcome

Option B.

### No in-process cache

* `LearnedTitleCache` is deleted. `is_storable_title(*, tool_name, title)` (the
  shared length caps: 128-char name, 256-char title) replaces its static
  method.
* The per-request mapper from `ToolDisplayNameMapper.with_tools()` keeps its
  titles in its own plain dict, alive for that request only. No state is shared
  between mapper instances, and the process-wide singleton is never mutated: if
  a start event arrives with only the singleton, the handler renders with a
  throwaway scoped copy.
* `LearnedTitleStore._misses` and `_recent_writes` are deleted. A request
  remembers a store miss in its mapper (`note_store_lookup_miss`) so an untitled
  tool costs at most one read per request.
* On a `call_tool` start with no title in the request mapper, the handler calls
  `get_title` (bounded by `LEARNED_TITLE_STORE_TIMEOUT_SECONDS`, default 0.5s)
  and registers a hit in the request mapper. Precedence is unchanged: static
  config, live tool titles, request-learned or stored titles, humanized name.
* Cost: `search_tools` results rewrite their titles every time (no write
  de-duplication), and each new request reads Mongo once per untitled `call_tool`
  target. Both are small keyed operations.

### Circuit breaker kept

The 30-second failure cool-down stays as a circuit breaker: after a store error
or timeout the read and learn paths are skipped, so a degraded Mongo costs one
timeout per pod per 30 seconds instead of one per start event. It is a single
timestamp flag, holds no title data, and is per process because it describes
the health of a dependency, not shared data. Without it a Mongo outage would add
the read timeout to every tool start line on every pod. The bulk operations
(`replace_titles`, `clear`) neither check nor trip it.

### Replace and clear on reload

`LearnedTitleBulkStoreProtocol` adds `replace_titles(*, titles, prune=True)` and
`clear()`, each returning a `TitleReplaceResult(ok, written, deleted)` and never
raising (bounded by `LEARNED_TITLE_STORE_BULK_TIMEOUT_SECONDS`, default 10).

The py-key-value Mongo store cannot enumerate keys and its `destroy_collection`
leaves the store's setup state stale (the next write fails), so deletion is
driven by a manifest entry (`__warmed_titles_manifest__`) listing the names the
last warm wrote. `replace_titles` reads the manifest, writes the new titles,
writes the new manifest, then deletes names that were in the old manifest and
not in the new set. Write-then-delete means a concurrent reader never sees an
empty window. `clear()` deletes the manifest names and the manifest.
Search-learned entries are not in the manifest; they carry a TTL and age out.
Warmed entries carry `warmed: true`; `put_titles` reads just the entries it is about
to write (`get_many`, never the manifest) and skips warmed ones, so a `search_tools`
write cannot replace a warmed entry with an expiring one.

`prune=False` upserts and keeps every previously warmed name. The warmer uses it
when the catalog response is partial (see below).

### TTL decision

Warmed entries are written without a TTL. The only thing that repopulates them is
a reload, so any finite TTL would eventually blank titles if reloads stop (a
failed reload, a deploy gap). A removed or renamed tool is deleted by the next
successful reload via the manifest. Entries learned from `search_tools` keep the
existing TTL (`LEARNED_TITLE_STORE_TTL_SECONDS`, default 24h).

### Warmer

`ToolTitleWarmerProtocol.warm(*, config: MCPConnectionConfig, server_name: str)
-> ToolTitleWarmResult` and the `ToolTitleWarmer` implementation (constructor
takes the bulk store, timeout, max titles, max response bytes). It calls the
catalog's `list_tool_titles` through `call_mcp_tool_raw` (the same machinery as
the discovery meta-tools) with the caller's connection config, so the caller's
auth headers are forwarded; nothing is logged but counts and a reason code. It
never raises; the result has `ok`, `reason`, counts and `partial`.

Validation: the response must be a dict whose `titles` is a dict; entries with
non-string names or titles, blank titles or entries over the length caps are
dropped; unknown fields are ignored; a response over the size cap or with more
than the max titles is rejected whole (never truncated, because a truncated set
followed by a prune would delete real titles); an empty valid set is rejected
(no wipe). If `unresolved_servers` is non-empty, or any entry was dropped, the
warm upserts without pruning, so a server that is down keeps its titles.

### Catalog contract this depends on

Built in baileyai-skills-service (BAI-1732): the tool-catalog MCP server has a
read-only tool `list_tool_titles` with no required arguments returning structured
content `{"titles": {"<name passed to call_tool>": "<title template>"},
"unresolved_servers": ["<server>", ...]}`. Keys are exactly the `name` the model
passes to `call_tool`. Unknown fields are ignored; adding fields is
backward compatible, changing the two above is not.

### Consequences

* Good: all pods see the same titles, no per-pod drift, no process-local data.
* Good: `load_skill` and `search_skills` get real start-line titles after the
  first reload.
* Bad: a title changed in the catalog shows only after the next reload.
* Bad: one Mongo read per request for each untitled `call_tool` target.
* Public API removed: `LearnedTitleCache`, the mapper's `learned_titles`
  constructor argument and `remember_shared_title`. No consumer in the org uses
  them.
* Reload wiring (calling `warm` after a catalog reload) and the version pin are
  done in baileyai, not here.

## Related Work

* ADR 0007 (persisted learned titles); BAI-903, BAI-1301, BAI-1306.
