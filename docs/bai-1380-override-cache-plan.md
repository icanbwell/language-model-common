# BAI-1380 implementation plan: cache per-client config overrides, move the blocking GitHub download off the event loop

Status: plan only, no code changes. Decision record: `adrs/0004-cache-per-client-config-overrides.md`.
Source: Finding 1 (and Finding 5) of baileyai PR #603, `docs/latency-analysis.md`.
Line references are to `origin/main` of this repo at 41ed81f, verified by reading the files.

## Constraints that apply

* Tenancy: `client_id` is untrusted input and becomes a cache key. Isolation rules in step 2.
* PHI: none stored (model configuration only). Logs carry client id and outcome, never override contents.
* Public contracts: `ConfigReader.read_model_configs_async(client_id=...)` signature and result are
  unchanged. New public surface is additive (env properties, optional `clear_client_override`).
* Event schemas: none. New dependencies: none (Mongo store, `AdvisoryLock`, `asyncio` already used).
* No in-process cache. Only the shared store.
* All new Python uses keyword-only arguments, empty `__init__.py`, env vars through
  `LanguageModelCommonEnvironmentVariables`, services registered in `container_factory.py`.

## Verification of the analysis claims

| Claim | Result |
|---|---|
| Override is read live from GitHub each request; `download` has no skip-if-present check | Confirmed. `download` (`github_directory_downloader.py:49-131`) always calls `_do_download` when the lock is acquired or no store is set. The on-disk directory is only used as a fallback when the lock is held (`:110-113`). |
| `_fetch_to_directory` is synchronous and blocks the loop; `time.sleep` in retries | Confirmed. `_do_download` is called un-awaited at `:115` and `:123`; `time.sleep(delay)` at `:207` (up to 2 s then 4 s). Also blocking on the loop: `FileConfigReader.read_model_configs` (`config_reader.py:375`). Extra effect the analysis did not mention: the `AdvisoryLock` heartbeat task is starved while the loop is blocked. |
| Base configs come from Mongo cache with TTL 86400 s | Partly unverified. The code default for `MODEL_CONFIG_CACHE_TTL_SECONDS` is 3600 (`language_model_common_environment_variables.py:169-175`). 86400 must be a prod env value; confirm in helm/`baileyai-configuration`. |
| 8 serial GitHub calls, about 1.7 s, 12 of 22 requests | Not re-measured (trace data was not re-queried). Consistent with the code: a present override directory with two files costs recursive tree listings plus file reads through fsspec; a missing directory costs a failing call and, with `log_diagnostics_on_not_found=False` (`:318-321`), no extra probe. Which of the 8 calls (if any) are the prompts probe is not established, see step 0. |
| `_retry_mcp_resolution` is a per-request cost | Partly wrong. It runs only when `not self._mcp_resolved_once` (once per `ConfigReader` instance, flag set at `:87`) or `_has_unresolved_mcp_servers` (`:82`). In steady state it is not per request unless some `mcp_server` ref stays unresolved. The MCP traffic seen before the first LLM call has other candidate sources, see step 4. |

## Step 0: confirm what the 8 spans and the MCP spans actually are (half a day, no code)

Before cutting code, pull one prod trace each for a request with and without the GitHub spans and
read span URLs/attributes:

* Which GitHub paths are fetched: `clients/<id>` only, or also the prompts folder. If the prompts
  folder is probed every request, `_resolve_prompt_library_async` (`:685`) is a second uncached GitHub
  path: `resolved_path` is only truthy when a prompt base path is configured or a previous discovery
  succeeded (`prompt_library_manager.py:51-53`, `config_reader.py:689-692`), so a repo without a
  `prompts/` folder re-probes GitHub on every request. If confirmed, add it to step 2 (cache the
  discovery result, including "none").
* Which MCP spans precede the first LLM call and who issued them (this library's `McpJsonFetcher`
  versus baileyai's tool catalog fetch).

## Step 1 (separable, ship first): move the blocking download off the event loop

Goal: stop stalling other requests. Independent of the cache; releasable alone.

Files:

* `languagemodelcommon/configs/config_reader/github_directory_downloader.py`
  * `download` (`:49`): run the sync work with `await asyncio.to_thread(self._do_download, ...)` at
    `:115` (inside the `AdvisoryLock`) and `:123`. Keyword arguments only (wrap in a small
    `functools.partial` or lambda, since `to_thread` forwards kwargs).
  * `_download_with_retry` (`:164`): make it `async`. Run each `_fetch_to_directory` attempt via
    `asyncio.to_thread`, and replace `time.sleep(delay)` (`:207`) with `await asyncio.sleep(delay)`,
    so a retry backoff does not occupy a thread-pool worker for up to 6 s. `_do_download` (`:133`)
    becomes `async` or is inlined.
  * `_fetch_to_directory` (`:300`): make `staging_dir` and `old_dir` unique per call
    (`{target}.staging.{pid}.{uuid8}`), not per pid (`:311-312`). Today concurrent same-target downloads
    in one process are impossible only because the loop is blocked. After `to_thread` two requests for
    the same client can run it concurrently, race on `rmtree(staging_dir)` and on the swap
    (`:367-374`). Add an in-process `asyncio.Lock` per target key in `download` as well, so concurrent
    same-process callers collapse (this lock guards a download, it is not a cache).
  * `_log_fetch_failure_diagnostics` (`:222`) mutates `self._last_probed_at` (`:260`) from worker
    threads. Plain dict set/get is safe under the GIL; note it as a diagnostic throttle, not a data
    cache, in a comment. It also uses a blocking `httpx.Client` with a 5 s timeout: fine in a thread
    once step 1 lands, and the `_PROBE_TIMEOUT_SECONDS` comment (`:39-44`) should be updated.
* `languagemodelcommon/configs/config_reader/config_reader.py`: wrap the sync disk read at `:375` and
  `:385` (`self._file_config_reader.read_model_configs`) in `asyncio.to_thread`.
* `github_directory_helper.py:191-198`: no change needed (it already awaits `download`).

Tests (`tests/configs/test_github_directory_downloader_integration.py`, `tests/configs/test_config_reader.py`):

* Event loop stays responsive: run `download` against a fake slow `_fetch_to_directory` (sleep) and
  assert a concurrent `asyncio.sleep(0)` ticker keeps ticking (loop lag below a threshold).
* Retry path: parameterized (transient failure then success; exhausted retries raises `ValueError`;
  `FileNotFoundError` is not retried) and asserts backoff uses `asyncio.sleep`, not `time.sleep`.
* Two concurrent downloads of the same target in one process produce a valid directory and no
  `FileNotFoundError` from the swap.
* Existing tests keep passing; fix any that patch `time.sleep`.

Effort: 1 to 1.5 days including review. Risk: low; behaviour change is only scheduling.

## Step 2: cache the per-client override (the main change)

### 2.1 Result type so outcomes stop collapsing

`_read_override_models_async` (`config_reader.py:308-338`) returns `[]` for four different outcomes.
Add `languagemodelcommon/configs/config_reader/override_read_result.py`: a frozen dataclass
`OverrideReadResult(outcome: OverrideOutcome, models: tuple[ChatModelConfig, ...])` with
`OverrideOutcome` = `FOUND | ABSENT | LOCK_HELD | FAILED`. Mapping from the current branches:

* `FileNotFoundError` (and the `"FileNotFoundError" in str(e)` string check at `:324`) -> `ABSENT`
* `read_models_from_path_async` returns `[]` because `resolved is None` (`:367-373`) -> `LOCK_HELD`.
  This needs `read_models_from_path_async` to signal it (raise a typed `DownloadLockHeld` or return
  a sentinel); today it is indistinguishable from "directory with zero models".
* `ValueError` (download failed, malformed) and any other `Exception` -> `FAILED`
* success -> `FOUND` (an empty directory is `FOUND` with zero models)

Only `FOUND` and `ABSENT` are ever written to the cache. `LOCK_HELD` and `FAILED` never are.

### 2.2 Cache component

Add `languagemodelcommon/configs/config_reader/override_config_cache.py`: class
`ClientOverrideConfigCache` (single responsibility: key, serialisation, TTL, tenant check). Constructor
takes `store: BaseStore | None`, `collection: str`, `schema_version: int`, and a TTL settings value
object. Registered in `container_factory.py`; `ConfigReader` receives it by constructor injection
(optional, default `None` keeps current behaviour, like `model_config_cache_store`).

* Key: `v{SCHEMA_VERSION}:ovr:{ref_hash}:{client_id}`; `ref_hash = compute_short_ref_hash(config_path)`
  (`utilities/ref_hash.py`, same as `_model_config_cache_key` at `:154`). Built only after
  `_resolve_override_config_path` (`:518`) has applied `_is_valid_client_id` (`:566`) and, for local
  paths, the `resolve()/relative_to()` containment check (`:553-560`). Both checks stay (see the
  SECURITY docstring at `:523-532`). Add a length cap (64) inside `_is_valid_client_id`; an over-long id
  is treated as invalid exactly like today's invalid id (returns no override, logs a warning).
  The cache never accepts a raw `client_id`; it takes the validated one and the already-resolved
  override path so that key and path are derived from the same validated value.
* Why `ovr`: the base reader scans prefix `v1:{ref_hash}:` (`:172-173`); a 12-hex hash can never equal
  `ovr`, so override rows cannot be read as base models. A unit test asserts this explicitly.
* Value: `{"client_id", "status", "fetched_at", "models"}`. Serialise models with
  `model_dump(exclude_unset=True)` and read with `model_validate`, so `_merge_model_configs`
  (`:585-620`, uses `exclude_unset=True` at `:614`) behaves identically on cached and fresh data.
  Verified: a plain `model_dump()` round trip marks all 20 fields as set and would let defaults
  overwrite base values. Test: merge result from cached override equals merge result from fresh one.
* Read: exact-key `get` (no regex), reject and log if stored `client_id` differs from the requested
  one, return a typed `CachedOverride(result, age_seconds, is_fresh)`.
* Write: `put(..., ttl=hard_ttl)` where negative rows use the negative TTL as hard TTL.
* Delete: `delete(client_id)` for the optional per-client invalidation.
* Env properties on `LanguageModelCommonEnvironmentVariables` (next to
  `model_config_cache_ttl_seconds`, `:169`): `model_config_override_cache_enabled` (default false),
  `model_config_override_cache_soft_ttl_seconds` (900),
  `model_config_override_cache_negative_ttl_seconds` (300),
  `model_config_override_cache_hard_ttl_seconds` (86400). Env names use the `MODEL_CONFIG_OVERRIDE_CACHE_`
  prefix.

### 2.3 Read flow in `ConfigReader`

Modify `_read_override_models_async` (`:308`) to delegate to a new `_read_override_with_cache`
(feature flag off or no cache store -> current behaviour, byte for byte):

1. Local filesystem config paths (neither GitHub nor S3) skip the cache (cheap, and
   `_resolve_override_config_path` already does `exists()`).
2. `get` row. Fresh (age < soft TTL): return `FOUND` models, or `[]` for `ABSENT`. No network.
3. Stale (age >= soft TTL, row still present): return its content immediately and schedule one
   background refresh (`asyncio.create_task`, task reference kept on the reader and awaited/cancelled on
   shutdown; the task itself is not a cache). The refresh runs under `AdvisoryLock` (new lock key
   `override_refresh:<hash of cache key>`, `ttl_seconds` 60). Lock not acquired means someone else is
   refreshing: do nothing.
4. Miss: refresh inline under the lock. If the lock is held by another worker, poll the cache
   (50 ms steps, max 3 s) for the winner's row; on timeout fetch directly (idempotent, bounded by
   step 1's timeouts). Never treat `LOCK_HELD` as `ABSENT`.
5. Refresh = call the existing read path (`read_models_from_path_async(config_path=override_path,
   log_diagnostics_on_not_found=False)`), classify into `OverrideReadResult`, write `FOUND` or `ABSENT`
   only. On `FAILED` keep the existing row untouched (serve stale up to hard TTL) and write nothing; if no
   row exists, return `[]` and log a warning as today (not cached, so the next request retries; add a
   short failure back-off marker, 15 s, to avoid hammering GitHub during an outage with many requests).
6. Unresolved MCP refs: `read_models_from_path_async` ends with `_resolve_mcp_servers_async` (`:396`),
   so override models are cached after MCP resolution, like base rows. If the fresh override still has
   unresolved `mcp_server` refs (`_has_unresolved_mcp_servers`, `:464`), write it with a 60 s soft TTL,
   so it re-resolves soon without re-hitting GitHub per request (resolution only needs the plugin
   server; a follow-up can re-resolve MCP from the cached models without a GitHub call).

Errors from the cache store (get/put) follow the base cache's fail-fast convention (`:166-168`,
`:220-222`) when the flag is on, so a misconfigured store surfaces immediately. Decide with reviewers
whether a store outage should instead degrade to the live read (recommended: log and fall back to live
read for the override only, since this is an optimisation layer).

### 2.4 Invalidation

* `/reload` (`baileyai/services/message_preprocessing/reload_service.py:68-90`) calls
  `ConfigReader.clear_cache` (`config_reader.py:739-751`), which deletes all keys with prefix `v1:`
  via `_get_cache_keys_by_prefix` (`:192-213`). Override rows (`v1:ovr:...`) match, so full `/reload`
  needs no baileyai change. The release gate
  (`baileyai/services/release_cache_gate_service.py`) is documented as performing the same clear as
  `ReloadService`, so a new release also clears overrides (confirm in the implementation PR, the
  file was only skimmed).
* Caveat (existing, unchanged): `_get_cache_keys_by_prefix` returns `[]` for any non-Mongo store, so
  `clear_cache` is a no-op there. Add a test pinning the Mongo behaviour with overrides; flag the
  limitation in the PR description.
* `/reload` warm (`reload_service.py:92-136`) only warms base configs, not overrides; overrides refill
  on first request after reload. Acceptable (single inline fetch per client); document it.
* Optional, additive: `ConfigReader.clear_client_override(*, client_id)` plus a baileyai
  `/reload <client>` target. Only if Imran wants per-client reload (ADR open question 2); not in the
  first release.

### 2.5 Observability

The repo has no metrics today (OpenTelemetry appears only in `mcp/interceptors/tracing.py`). BAI-1379
owns metrics and is out of scope here, so this step ships structured logs with stable field names that
BAI-1379 can turn into metrics, and wraps the refresh in a tracing span if the span helper from that
ticket is available:

* One log line per override read: `override_cache outcome=hit|stale|miss|absent_hit|bypass`,
  `age_s`, `client_id` (placeholder-safe, id only), `duration_ms`.
* Refresh: `override_refresh result=found|absent|lock_held|failed`, `download_ms`,
  `served_stale=true|false`, `github_calls` if cheaply known.
* Warning on `FAILED` with exception type only. No override contents, no tokens.

### 2.6 Tests (`tests/configs/test_config_reader.py`, new `tests/configs/test_client_override_cache.py`)

All new tests are `pytest.mark.parametrize` where there are more than two variations, use keyword
arguments when calling production code, and use placeholder client ids (`client-a`, `client-b`).

* Outcome matrix (parameterized): fresh hit, stale hit schedules refresh, miss fetches, absent hit,
  flag off (no store calls, live read), store `None`.
* Negative caching: `FileNotFoundError` writes `ABSENT` with the negative TTL; second call makes zero
  downloader calls. `FAILED` and `LOCK_HELD` write nothing.
* Serve stale when GitHub fails: stale row present, refresh raises `ValueError`; request returns the
  stale override and the row is unchanged.
* Cross-client isolation: client-a override cached, request for client-b returns client-b's
  (or none), never client-a's; corrupted row whose stored `client_id` differs is ignored.
* Key namespace: an override row is never returned by `_read_from_model_config_cache`; a base-model
  named like a client id is never returned as an override.
* Untrusted input: `../x`, `a/b`, empty, over-length, unicode ids never create a row or a download.
* Merge fidelity: `_merge_model_configs` output identical for fresh versus cached override
  (guards the `exclude_unset` bug above).
* `clear_cache` removes override rows (Mongo-shaped fake store with `_collections_by_name`).
* Stampede: N concurrent misses for one client produce one downloader call (lock + in-process
  collapse); lock-held path polls then fetches.
* Contract test: `read_model_configs_async` public result unchanged for existing scenarios.

### 2.7 Rollout and rollback

1. Release library with `MODEL_CONFIG_OVERRIDE_CACHE_ENABLED` default false (no behaviour change).
   Version comes from the release tag (`python-publish.yml` writes it into `VERSION` on release;
   `VERSION` in the repo is the placeholder 0.0.1). Cut a release; do not edit `VERSION` by hand.
2. baileyai: bump `language-model-common` in `pyproject.toml` (currently `>=4.1.0`, `:67`), run
   `uv lock` (the repo's `exclude-newer` for this package is `0 days`, `:219`, so the new release is
   picked up immediately). Set the flag in dev helm values, verify with GitHub spans gone, then
   staging, then prod.
3. Rollback: set the flag to false (instant, no deploy of code); rows are inert and expire or are
   cleared by `/reload`.

### 2.8 Success measure

* GitHub GET spans per chat request (prod trace sample, same query as the analysis): from 8 on 12/22
  requests to 0 in steady state (plus at most one refresh per client per soft TTL, off the request path
  for stale hits).
* Pre-LLM time (request start to first Bedrock call) p50/p95: expected about 1.7 s lower on affected
  requests. Baseline per analysis: 2.45 s in the sampled request. Measure with the `pre_pipeline` group
  and TTFT metric from the baileyai ticket (analysis recommendation 1), otherwise from traces.
* Event loop lag on a worker during a download (should no longer show multi-second stalls).
* No cross-client config leakage (isolation tests, plus a canary check that two clients with different
  overrides resolve different model lists).

## Step 3 (second step, only if step 0 confirms): reduce MCP resolution cost in the config path

Finding: the premise that `_retry_mcp_resolution` is a per-request path is not accurate (see the table
above). The MCP calls seen before the first LLM call come from some combination of:

* `_retry_mcp_resolution` (`config_reader.py:486`) running every request because some `mcp_server`
  ref never resolves. Cause candidates: `McpJsonFetcher.fetch_plugin_async` returns `(None, None)` when
  a plugin has no servers (`mcp_json_fetcher.py:99-101`), so the ref stays unresolved forever and the
  retry (plus the full base-cache rewrite at `:498`, one `put` per model) repeats every request.
* Override reads calling `_resolve_mcp_servers_async` (`:396`) on every request. Step 2 removes this
  (cached after resolution).
* baileyai's own tool-catalog fetch (not this repo; analysis Finding 5).

Cost shape inside this repo: `fetch_plugins_async` (`mcp_json_fetcher.py:177-194`) is serial per
plugin and `fetch_plugin_async` opens a fresh session per plugin (`:50-56`). With
`MCP_PROTOCOL_NEGOTIATION_MODE=auto` (ADR 0003) every fresh session also does a `server/discover`
probe before `initialize`, which matches the `server/discover` spans in the trace.

Changes if step 0 shows this repo is the source:

* `fetch_plugins_async`: one session for all plugins, `call_tool` per plugin (or the existing
  `fetch_all_async` single call, `:111`), and `asyncio.gather` over plugins if one session is not
  possible. Preserves the `(configs, errors)` return contract.
* Remember "plugin has no servers" so it stops counting as unresolved: either treat `(None, None)` as
  resolved-empty in `_has_unresolved_mcp_servers`, or throttle `_retry_mcp_resolution` with a back-off
  marker (like the 15 s failure marker in 2.3).
* `_retry_mcp_resolution`: write back to the cache only if resolution changed something.
* Tests: parallel fetch preserves ordering and error collection; no-server plugin does not trigger a
  retry on the next request; unresolved-forever case is throttled.

Effort: 1 day, after step 0 evidence. If the source is baileyai's tool catalog, close this step and
leave it to the baileyai ticket.

## Effort summary

| Item | Effort |
|---|---|
| Step 0 trace confirmation | 0.5 day |
| Step 1 off-loop download (separate PR) | 1 to 1.5 days |
| Step 2 override cache, tests, ADR to accepted | 3 to 4 days |
| Step 3 MCP resolution (conditional) | 1 day |
| baileyai version bump and flag rollout across dev, staging, prod | 0.5 day plus soak |

Total about 6 to 7 engineering days, shippable in three independent PRs (step 1, step 2, step 3).

## Governing artifacts and review

* ADR `adrs/0004-cache-per-client-config-overrides.md` (this PR, status proposed). New cache pattern, so
  EA is informed through the PR architecture review workflow; no new technology (existing Mongo store),
  so no approved-tech registry proposal, no FDR (no FHIR), no public API change.
* Update `docs/config-reader-security.md` in the step 2 PR with the new key rules and length cap.

## Open questions

1. TTL defaults (900 s positive, 300 s negative, 86400 s hard): acceptable?
2. Per-client `/reload <client>` needed?
3. Confirm prod `MODEL_CONFIG_CACHE_TTL_SECONDS` (86400 per the analysis versus code default 3600).
4. Store failure on the override cache: fail fast like the base cache, or fall back to the live read?
5. Add the `client_id` length cap (64)? It is a small additive tightening of validation.
