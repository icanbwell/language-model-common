---
title: Lock PromptLibraryManager's first-time GitHub resolution and add TTL/ref-hash versioning to PromptStore
status: accepted
date: 2026-09-23
decision-makers: Imran Qureshi
consulted: (pending)
informed: (pending)
---

# Lock PromptLibraryManager's first-time GitHub resolution and add TTL/ref-hash versioning to PromptStore

## Context and Problem Statement

While fixing BAI-932 (mcp-fhir-agent's `ClientScopedConfigReader` racing its own advisory GitHub
download lock for every new/unseen client_id), an audit for the same failure shape and for missing
cache-version-gating elsewhere in this codebase found two related issues in this repo:

1. `PromptLibraryManager._ensure_local_path()` (`configs/prompt_library/prompt_library_manager.py`)
   resolves a `github://` prompt-library path with no lock around its check-then-fetch
   (`if not self._github_resolved: ...`). Concurrent first-time calls (e.g. several prompt lookups
   fired at once during startup, before any resolution has happened) each independently see
   `_github_resolved is False` and call `GitHubDirectoryHelper.resolve_github_path()`, racing the
   same advisory download lock `GithubDirectoryDownloader` takes per `github://` path. A loser gets
   `None` back and the method raises `FileNotFoundError` for that one call -- the same race shape as
   BAI-932, just narrower in practice since it only recurs on the first-ever resolution per process,
   not on every new client_id the way BAI-932 did.
2. `PromptStore` (`configs/prompt_library/prompt_store.py`) caches prompt content keyed by the bare
   prompt name only -- no ref/source-path scoping and no TTL. Once seeded, an entry is cached
   forever; a new prompt version pushed upstream is never picked up without a manual `clear()`. This
   is the same class of gap `ConfigReader`'s model-config cache already avoids via its
   `SCHEMA_VERSION` + `_config_ref_hash(config_path)` + TTL scheme (BAI-720).

## Decision Drivers

* Must not require every existing caller of `PromptStore` to change simultaneously -- this is a
  shared library consumed by multiple services; the fix must be additive.
* Should reuse the versioning shape `ConfigReader` already established in this same codebase
  (BAI-720 precedent) rather than invent a second convention.
* Must not touch `PromptLibraryEnvironmentVariables` (a narrow `@runtime_checkable` `Protocol`) with
  a new required member -- `PromptLibraryManager.__init__` does an `isinstance()` check against it,
  so any new required property would break every existing concrete environment-variables class
  across every consuming repo that doesn't yet implement it.

## Considered Options

1. **Add the new TTL as a required property on `PromptLibraryEnvironmentVariables`.** Rejected: this
   protocol is `@runtime_checkable` and structurally enforced via `isinstance()` in
   `PromptLibraryManager.__init__`; adding a new required member is a breaking change for every
   existing implementer, in every consuming repo, not just this one.
2. **(Chosen) Add `source_ref`/`ttl_seconds` as optional constructor parameters on `PromptStore`
   itself, both defaulting to values that fix the problem (`ttl_seconds` defaults to 3600s; `source_ref`
   defaults to `None`, which preserves the exact bare-name key shape).** Purely additive to the
   constructor signature. `LanguageModelCommonEnvironmentVariables` (a concrete class, not the
   Protocol) gains a new `prompt_store_ttl_seconds` property (env var `PROMPT_STORE_TTL_SECONDS`,
   default 3600) so DI wiring can pass a configurable TTL; `container_factory.py`'s
   `_create_prompt_store` passes `source_ref=env.prompt_library_path` and this new TTL.
3. **Add locking to `PromptLibraryManager` via a module-level lock.** Rejected: a per-instance
   `asyncio.Lock` (mirroring `ConfigReader._lock`'s existing pattern) is sufficient and avoids a
   global mutable singleton (`No Hidden Global State`).

## Decision

* `PromptLibraryManager._ensure_local_path()` now uses double-checked locking around its
  check-then-fetch, via a new per-instance `self._resolve_lock: asyncio.Lock`, mirroring
  `ConfigReader._read_base_models_async`'s own pattern: the cheap `_github_resolved` check stays
  outside the lock (the common already-resolved path never contends on it); the lock only
  serializes the rare first-time-resolution path.
* `PromptStore.__init__` gains two optional parameters: `source_ref: str | None = None` (hashed into
  a `v{SCHEMA_VERSION}:{ref_hash}:` key prefix, same scheme as `ConfigReader._config_ref_hash`) and
  `ttl_seconds: int | None = 3600` (passed through to the underlying store's `put(..., ttl=...)`).
  Both are backward compatible: omitting `source_ref` preserves the old bare-name key exactly;
  `ttl_seconds` defaults to a real value rather than `None`, since the entire point of this change is
  to stop permanent caching -- callers that genuinely want the old infinite-cache behavior can pass
  `ttl_seconds=None` explicitly.
* `container_factory.py`'s `_create_prompt_store` is updated to pass both new parameters using a new
  `prompt_store_ttl_seconds` property on `LanguageModelCommonEnvironmentVariables` (env var
  `PROMPT_STORE_TTL_SECONDS`, default 3600) and the existing `prompt_library_path`.

## Consequences

* Good: closes the BAI-932-shaped race in `PromptLibraryManager`, and the missing-version-gating gap
  in `PromptStore`, using the exact precedent (`ConfigReader`'s locking and versioning patterns)
  already established in this codebase.
* Good: fully additive -- no existing call site of `PromptStore(...)` or any implementer of
  `PromptLibraryEnvironmentVariables` needs to change to keep working exactly as before.
* Neutral: existing deployments that don't update their DI wiring keep the old bare-name key shape
  (no `source_ref` passed) but DO get the new default TTL (3600s) automatically, since `ttl_seconds`
  defaults to a real value rather than `None`. This is a deliberate behavior change (entries now
  expire after an hour instead of never) for anyone who doesn't explicitly opt out.
* Neutral: the in-process `asyncio.Lock` only serializes concurrent calls within one process/pod;
  a cross-process race (multiple pods resolving the same never-cached `github://` path
  simultaneously) still depends on `GithubDirectoryDownloader`'s own advisory lock and its
  lock-held-returns-`None`-then-raises behavior, unchanged by this ADR.

## References

* BAI-932 -- the originating fix in `mcp-fhir-agent`'s `ClientScopedConfigReader`, whose shape this
  ADR's Option 2 mirrors.
* BAI-933 -- this ADR's tracking ticket.
* BAI-720 -- `ConfigReader`'s `SCHEMA_VERSION` + `_config_ref_hash` scheme, reused here.
* `languagemodelcommon/configs/config_reader/config_reader.py` -- `ConfigReader._read_base_models_async`
  and `_config_ref_hash`, the patterns this decision mirrors.
* `languagemodelcommon/configs/prompt_library/prompt_library_manager.py`,
  `languagemodelcommon/configs/prompt_library/prompt_store.py` -- this decision's implementation.
