# Cache per-client config overrides in the shared model-config store

* Status: proposed (a human accepts it; the cache ships behind a flag, default off)
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1380 (this design); source analysis is Finding 1 of baileyai PR #603 (`docs/latency-analysis.md`)
* Date: 2026-10-30
* Implementation plan: `docs/bai-1380-override-cache-plan.md`

## Context and Problem Statement

`ConfigReader.read_model_configs_async(client_id=...)` is called by baileyai on every chat request
(`baileyai/routers/api_router.py:708`). Base model configs are served from the shared Mongo
model-config cache (`ConfigReader._read_base_models_async`, `config_reader.py:105`). The per-client
override is not cached: `_read_override_models_async` (`config_reader.py:308`) calls
`read_models_from_path_async` (`:340`), which calls `GitHubDirectoryHelper.resolve_github_path` and
`GithubDirectoryDownloader.download` (`github_directory_downloader.py:49`). `download` has no
skip-if-present check; whenever the advisory lock is acquired (or no store is configured) it calls
`_do_download` and re-fetches the directory from GitHub.

Two consequences, both measured in prod by the PR #603 analysis (not re-measured here):

1. About 1.7 s of serial GitHub API calls (8 calls, 0.15-0.23 s each) before the first LLM call, on
   12 of 22 sampled chat requests.
2. `_do_download` is synchronous (fsspec HTTP, `shutil`, and `time.sleep` in `_download_with_retry`,
   `github_directory_downloader.py:207`) and runs on the event loop thread from an `async def`. Every
   other request on that worker stalls while it runs, and the `AdvisoryLock` heartbeat task
   (`oidcauthlib`) is starved for the same period.

The org rules for this work: no in-process cache (multiple workers per pod, several pods; use a
shared store), and a new cache needs an ADR covering strategy, size limit, TTL, eviction and tenant
isolation. `client_id` is untrusted caller input (an HTTP header) used to build a GitHub path, so it
is also a security boundary (`config_reader.py:517-576`, `docs/config-reader-security.md`).

### Facts that constrain the design (verified in code)

* The model-config cache is keyed `v{SCHEMA_VERSION}:{ref_hash}:{model_name}` (`:154`), read by
  prefix scan `v1:{ref_hash}:` (`:172`), and cleared by prefix `v1:` (`:741`). A naive override key
  such as `v1:{ref_hash}:clients/<id>` would be read back by the base reader as a base model, which
  is a cross-tenant leak. The override namespace must not share the base prefix.
* `_merge_model_configs` (`:585`) merges overrides with `model_dump(exclude_none=True,
  exclude_unset=True)` (`:614`). Pydantic tracks which fields were set. A cache round trip through
  `model_dump()` then `model_validate()` marks all 20 `ChatModelConfig` fields as set (checked
  against the current schema), so every default would overwrite the base value on merge. The cache
  must store `model_dump(exclude_unset=True)` so the set-fields are preserved.
* `_read_override_models_async` collapses four different outcomes into `[]`: no override directory
  (`FileNotFoundError`), download failure (`ValueError`/other exception), a malformed override, and
  "download lock held by another worker" (`resolved is None`, `:367-373`; and `download` also returns a
  pod-local, possibly stale on-disk copy when the lock is held, `:104-113`). Only a confirmed missing
  directory is safe to cache as "no override".
* fsspec raises a bare `FileNotFoundError` for any non-2xx GitHub reply (bad token, rate limit, private
  repo), so the exception alone does not prove the override is absent. Absence is confirmed with a
  status-checked listing of the parent directory (`GitHubDirectoryHelper.confirm_directory_absent`)
  before `ABSENT` is written; an unconfirmed 404 is `FAILED` and never replaces a `FOUND` row.
* `/reload` in baileyai calls `ConfigReader.clear_cache` (`:739`), which deletes every key with
  prefix `v1:`.

## Decision Drivers

* Remove GitHub from the steady-state request path (target: 0 GitHub GET spans per chat request).
* Overrides stay reasonably fresh, and `/reload` still forces a refresh.
* GitHub outage must not break chat for clients with overrides (serve stale).
* No in-process cache; multi-worker, multi-pod safe; tenant isolation is a correctness constraint.
* Minimal diff, reuse existing infrastructure (Mongo cache store, `AdvisoryLock`, TTL, `/reload`).
* No new datastore, broker or cache technology (avoids a registry proposal).

## Considered Options

1. **Keep as is** (live GitHub read per request).
2. **Per-client rows in the existing shared model-config cache (Mongo), soft TTL plus hard TTL,
   stale-while-revalidate, negative caching of "no override".**
3. **Conditional revalidation (ETag / commit SHA)**: per request, one cheap conditional call to
   GitHub (`If-None-Match`), re-download only on change.
4. **Background poller**: a scheduled task refreshes all known clients' overrides; requests only read
   the cache.
5. **Process-local cache (dict / `lru_cache`)** per worker.
6. **Move overrides out of GitHub** (read from Mongo or `baileyai-configuration` at deploy time).

## Decision Outcome

Chosen: **Option 2**, with the blocking-download fix (`asyncio.to_thread`) shipped first and
independently, and Option 3 kept as a later optimisation of the refresh step only.

Rationale: Option 2 reuses the store, TTL, lock and `/reload` mechanisms the base config cache
already uses, so it adds no technology. Option 1 is the problem. Option 3 still puts one network
call on every request (about 0.15-0.2 s, plus GitHub rate-limit exposure) and fsspec's
`GithubFileSystem` does not expose ETags, so it needs a new bespoke client; it is still useful
inside the background refresh to turn a refresh into a 304. Option 4 needs a client registry the
library does not have (the set of valid client ids is open) and a scheduler. Option 5 violates the
org no-in-process-cache rule and is wrong across workers. Option 6 is an architectural change to
where config lives, out of scope for a latency fix.

### Cache design (the required ADR fields)

**Strategy.** Read-through, stale-while-revalidate, single-flight refresh.

* Hit within soft TTL: return cached override, no GitHub call.
* Hit past soft TTL (but entry still in store): return the cached override immediately, and schedule
  one background refresh guarded by `AdvisoryLock`. A failed refresh keeps the stale entry.
* Miss: refresh inline, guarded by `AdvisoryLock`. If another worker holds the lock, poll the cache
  (up to 3 s) and return as soon as its row appears; on timeout return no override for this request,
  uncached (there is no "fetch directly" bypass: the downloader has its own lock). Every lock-held
  return, including the on-disk fallback, is reported as lock-held (`raise_on_lock_held`) and is never
  written as "no override" or as fresh data.
* Failure (`ValueError`, timeout, malformed override, unconfirmed 404): never cached as "absent". Serve
  stale if present (and defer the next attempt by 15 s), else behave as today (return no override and
  log a warning).

**Key and value.** Same collection as the base cache (`model_config_cache_collection_name`). One row
per (config ref, client): key `v{SCHEMA_VERSION}:ovr:{ref_hash}:{client_id}`. `ovr` is not a 12-hex
hash, so the base prefix `v1:{ref_hash}:` can never match an override row, and the existing
`clear_cache` prefix `v1:` does match it, so `/reload` clears overrides with no change to the reload
path. Value:

```json
{"client_id": "<id>", "status": "found" | "absent", "fetched_at": 0,
 "models": [ChatModelConfig.model_dump(exclude_unset=True)]}
```

**Tenant isolation.**

* The key is built only after `_is_valid_client_id` (allowlist `^[a-zA-Z0-9_-]+$`, unchanged) passes,
  and only for ids that also pass `_is_cacheable_client_id` (strict full match, at most 64 chars). An
  invalid id never reaches the cache or GitHub, same as today; a valid but over-long id is read live and
  never cached, so flag-off and flag-on behaviour match for it.
* The key embeds `ref_hash` (hash of the config path including `?ref=`), so a pod pointed at a
  different ref never reads another ref's rows (same rule as BAI-720 for base rows).
* Reads are point `get` by exact key (no regex), and the stored `client_id` is compared with the
  requested one before use; a mismatch is treated as a miss and logged.
* Overrides are configuration (model definitions), not patient data. No PHI is stored or logged.
  Logs carry the client id and outcome only, never the override contents.

**Size limit.** Positive rows: one per client that has an override directory (small; tens). Negative
rows: bounded by TTL times request rate, because any well-formed id creates a row. Controls: negative
soft TTL of 300 s and hard TTL equal to it, a client id length cap (64), and a metric on negative
writes. The id is attacker-controlled input today (each unknown id already costs a GitHub call per
request, so this is strictly better), but if abuse shows up the follow-up is a cached `clients/`
directory listing (one call per ref) used to answer "no override" without writing a row per id.

**TTL.** Three env-configurable values (new properties on
`LanguageModelCommonEnvironmentVariables`): positive soft TTL 900 s, negative soft TTL 300 s, hard TTL
(store TTL, upper bound for serving stale) 86400 s. Defaults were chosen by the implementer on Imran's instruction to use the proposed values; a
reviewer can change any of them (environment variables only, no code change).

**Eviction.** Store-level TTL (`put(..., ttl=hard_ttl)`), the same mechanism as the base cache; no
LRU. `/reload` deletes all `v1:` rows including overrides. `SCHEMA_VERSION` bump invalidates all.

**Stampede protection.** `AdvisoryLock` (already used by the downloader, key
`github_download:<hash>`) with a new lock key per client override refresh; best-effort (the lock is
get-then-put, not atomic, which is acceptable because a duplicate refresh is idempotent).

**Feature flag and rollback.** `MODEL_CONFIG_OVERRIDE_CACHE_ENABLED` (default off in the library
release). Off means today's behaviour exactly. Rollback is the flag; rows are inert and expire or are
cleared by `/reload`.

### Consequences

* Good: steady state removes about 1.7 s and 8 GitHub calls from affected requests; GitHub outage no
  longer breaks clients with overrides; the event loop is no longer blocked by downloads (with the
  `to_thread` step).
* Good: no new technology; `/reload` and release-gate clears work unchanged.
* Bad: an override edit can take up to the soft TTL (900 s) to appear without `/reload`. Acceptable
  because the base config already has a longer TTL.
* Bad: the unresolved-MCP edge case (override has `mcp_server` refs the plugin server could not
  resolve) must be handled explicitly so it neither caches broken config for long nor re-hits
  GitHub per request; the plan specifies a short soft TTL for that case.
* Risk: moving the download to a thread removes the accidental serialisation the blocked loop gave;
  same-process concurrent downloads of one target would share `staging.{pid}`. The plan makes the
  staging directory unique per call.
* Neutral: `clear_cache` relies on `_get_cache_keys_by_prefix`, which only works for the Mongo store
  (`:192-213`, returns `[]` otherwise). Unchanged and out of scope; noted so reviewers do not assume
  other backends clear.

## Defaults chosen (change in review if needed)

| Setting | Default |
|---|---|
| Feature flag `MODEL_CONFIG_OVERRIDE_CACHE_ENABLED` | off |
| Positive soft TTL `..._SOFT_TTL_SECONDS` | 900 s |
| Negative TTL `..._NEGATIVE_TTL_SECONDS` | 300 s |
| Hard TTL `..._HARD_TTL_SECONDS` | 86400 s |
| Unresolved-MCP soft TTL / failed-refresh back-off / peer wait | 60 s / 15 s / up to 3 s |
| Cache-key client id cap | 64 chars, cache only |
| Cache store error | log and fall back to the live read (the override cache is an optimisation layer; the base cache still fails fast) |
| Per-client invalidation | `ConfigReader.clear_client_override`; no baileyai `/reload <client>` target yet |

## Open Questions

1. Prod `MODEL_CONFIG_CACHE_TTL_SECONDS`: 86400 s in the analysis versus code default 3600 s
   (`language_model_common_environment_variables.py`). Does not affect this design.
2. Does baileyai want a per-client `/reload <client>` target?

## Related Work

* BAI-1380 plan: `docs/bai-1380-override-cache-plan.md`
* baileyai PR #603 latency analysis (Finding 1, Finding 5)
* ADR 0002 (prompt-library lock and prompt-store TTL versioning), ADR 0003 (MCP negotiation mode)
* BAI-720 (ref-scoped cache keys), BAI-933/BAI-941 (download lock, diagnostics)
* Out of scope siblings: BAI-1379 (metrics), BAI-1381, BAI-1382, BAI-1383
