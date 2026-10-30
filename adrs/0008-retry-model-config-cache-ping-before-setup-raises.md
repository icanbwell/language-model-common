# Retry the model config cache ping before store setup raises

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1372
* Date: 2026-10-30

## Context and Problem Statement

`ValidatingMongoDBStore._setup()` pings MongoDB when the store is opened so a
misconfigured model config cache fails fast. `py-key-value-aio`'s
`BaseContextManagerStore.setup()` closes the store, and the `AsyncMongoClient`
it owns, whenever `_setup()` raises, and leaves setup incomplete so the next
call runs `_setup()` again on the closed client.

So one transient ping failure (for example `AutoReconnect` while a pod is
starting) leaves the store permanently unusable for the life of the process:
every later `setup()` fails with `Cannot use AsyncMongoClient after close`.
The model config reader then reads 0 configs, requests fail with
`Chat model config not found`, and the baileyai post-deploy security
regression fails and rolls the environment back (staging, 8.17.0, 2026-10-30).

`ConfigReader._read_configs_with_retry` does not help. It retries the source
read, but every attempt goes through the same closed client, so all attempts
fail identically.

## Decision Drivers

* A brief Mongo blip at pod start must not poison the process.
* Keep the fail-fast behaviour for a real misconfiguration (bad URL,
  credentials) and bound the time before giving up.
* Every external call has an explicit timeout; retries use exponential backoff
  with jitter.
* No new dependency and no behaviour change for callers of `create_cache_store`.

## Considered Options

### Option A: Retry the ping inside `ValidatingMongoDBStore._setup()` (chosen)

Retry connection failures and timeouts a bounded number of times before
`_setup()` raises, so the base class never sees the transient failure.

### Option B: Retry in `ConfigReader`

Already present, and ineffective: the store it retries against is closed after
the first failure.

### Option C: Recreate the Mongo client when the store finds it closed

Recovers even after an outage longer than the retry budget. Needs to touch
private state of `MongoDBStore` and coordinate with the base class's setup
state, which is a larger and more fragile change.

### Option D: Let the pod fail liveness and restart

Works but converts a blip into pod churn and a failed deploy.

## Decision Outcome

Option A. `PingRetryPolicy` (frozen dataclass) bounds the retry: 4 attempts,
3 s per ping, exponential backoff from 0.5 s capped at 4 s, full jitter.
Worst case before giving up is about 15 s. Only `pymongo.errors.ConnectionFailure`
(which includes `AutoReconnect` and `ServerSelectionTimeoutError`) and
timeouts are retried; authentication and other operation errors still fail on
the first attempt.

### Risks and mitigations

* **Outage longer than the budget:** the store is still closed after the final
  failure, as before this change, so that process stays without model configs.
  Option C is the follow-up if that proves common.
* **Slower startup when Mongo is down:** bounded by the policy above; logged at
  WARNING per retry with the exception type only.

## Test Plan

* Transient failures then success (0, 1, 3 failures): setup succeeds, store
  stays open, one sleep per failure.
* Failures on every attempt: raises after `max_attempts` with the existing
  `Model config cache failed to connect` message.
* Non-transient error (`OperationFailure`): not retried.
* Hung ping: times out and is retried.
* Backoff delay stays within the exponential ceiling and is jittered; policy
  rejects zero attempts.
