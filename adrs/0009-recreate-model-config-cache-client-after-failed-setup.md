# Recreate the model config cache Mongo client after a failed setup

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1374; follow-up to BAI-1372 (ADR 0008, Option C)
* Date: 2026-10-30

## Context and Problem Statement

ADR 0008 retries the open-time MongoDB ping inside
`ValidatingMongoDBStore._setup()` so a brief outage does not reach the base
store's close-on-failure path. The retry budget is bounded (about 43 s). If
MongoDB is still unreachable after it, `_setup()` raises, `py-key-value-aio`
closes the store and the `AsyncMongoClient` it owns, and every later `setup()`
re-runs `_setup()` on that closed client and fails with
`Cannot use AsyncMongoClient after close`. The process stays without model
configs until the pod restarts, even after MongoDB is back.

This affects all four stores built through `create_cache_store` (model config
cache, prompt store, MCP tool list store, learned title store).

## Decision Drivers

* A process should recover once MongoDB is reachable again, without a restart.
* While MongoDB is down, callers should not each wait out the full retry budget.
* No new dependency, and no change for callers of `create_cache_store`.
* Touch as little `py-key-value-aio` private state as possible.

## Considered Options

### Option A: Do nothing; rely on a pod restart

What ADR 0008 left in place. A pod that missed MongoDB for longer than the
budget stays broken until liveness or an operator restarts it.

### Option B: Rebuild the client at the start of the next setup, after a cooldown (chosen)

When the previous setup failed, `_setup()` first replaces the closed client
and database handle with new ones, but not sooner than a cooldown after the
failure. Within the cooldown it raises immediately without pinging.

### Option C: Rebuild immediately on every setup after a failure

Simplest, but while MongoDB is down every caller would wait the whole retry
budget in turn (setup runs under the store's lock).

### Option D: Subclass or wrap the base `setup()` to skip its close-on-failure

Avoids the closed client altogether, but depends on more of the base class's
internals, and a later close would still be needed for the client to be released.

## Decision Outcome

Option B. `PingRetryPolicy` gains `recovery_cooldown_seconds` (default 30 s).
`ValidatingMongoDBStore` records when a setup failed. On the next setup:

* within the cooldown it raises `ConnectionError` without touching the network;
* after the cooldown it builds a fresh `AsyncMongoClient` from the store's URL
  (injectable `client_factory`), re-points its database handle, and runs a
  single probe ping with `recovery_ping_timeout_seconds` (default 5 s), not
  the full retry budget. The store's setup lock is held while it runs and every
  caller queues behind it, so the probe must be short.

A store constructed with its own `client`, or without a `url`, is never rebuilt:
the former belongs to the caller, the latter has nothing to rebuild from.
`create_cache_store` always passes a URL.

State touched on the base store: `_client` and `_db`. A rebuild only happens
while the store's setup has never completed, and every operation goes through
`setup()` first, so nothing from a successful setup can still hold the old
client, and no collection handles or setup flags exist yet to go stale.

The cooldown must be greater than 0. The base class closes a failed store
after releasing its setup lock, so a zero cooldown would let a queued caller
rebuild the client while that close is still running and then have the new
client closed under it.

Any exception raised while opening the store, including from the client's own
connect step (for example an SRV DNS failure before the ping), is recorded as a
failed setup, so every failed setup leads to a rebuild.

### Risks and mitigations

* **Callers stall during each recovery probe** while MongoDB is still down:
  with the defaults, up to 5 s in every 35 s (30 s cooldown plus the probe).
  Callers queued on the store's setup lock wait for the probe, then see its
  failure and fail fast. Using the full retry budget for each recovery attempt
  was rejected because it would stall callers for about 43 s of every 73 s.
* **Reliance on private base-class attributes:** the same ones the existing
  `ValidatingMongoDBStore` and `McpToolListStore` already use; covered by tests
  that run the real `setup()` against a fake client that is unusable once closed.

## Test Plan

* After a failed setup and the cooldown, a fresh client is built from the
  store's URL and setup succeeds; the old client stays closed.
* Within the cooldown, setup raises without building a client or pinging.
* Repeated failures: each rebuild waits out its own cooldown.
* A recovery attempt is one ping, even when the policy allows several.
* A failure before the ping (the client's own connect step) still leads to a
  rebuild.
* A healthy store is never rebuilt.
* A caller-owned client is never rebuilt.
* Cooldown boundary: just under the cooldown fails fast; at and after it
  rebuilds.
* Two callers set up at once share one attempt and one failure.
* A zero, negative or NaN cooldown or recovery ping timeout is rejected.
* Checked once, outside the test suite, against `pymongo 4.19.0` and
  `py-key-value-aio 0.4.6` (the versions baileyai ships; this repo's own lock
  has `pymongo 4.16.0` and `py-key-value-aio 0.4.5`, whose `setup()` code is the
  same) with a real unreachable client: after a failed setup the next setups
  fail with a new connection error, not `Cannot use AsyncMongoClient after close`.
  The automated tests use a fake client that becomes unusable once closed.
