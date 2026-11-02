# Always emit llm_call timing and a prepare item on the Responses API stream

* Status: proposed (a human accepts it; client-team confirmation is still needed, see Consequences)
* Deciders: language-model-common maintainers, baileyai maintainers, baileyai-chat-ui maintainers
* Ticket: BAI-1440
* Date: 2026-11-01
* Related: baileyai PR #627, `adrs/019-llm-call-timing-and-prepare-stream-items.md` (baileyai repo)

## Context and Problem Statement

The chat UI's "Used N agents" summary shows how long each tool took but not how long each model
call took, and not how long the request spent before the first model call. The Responses API wrapper
already has `llm_call` start/done items (BAI-882), but they are emitted only when
`enable_debug_logging` is on, because they carry prompt and response text.

How do we give every client per-call timing without sending prompt or response content to clients
that did not opt into debug output?

## Decision Drivers

* No prompt or response content (PHI) in the always-on payload.
* Additive wire contract: existing consumers keep working (event schemas change additively only).
* Overlapping model invocations (parallel subagent tools) must time independently.
* A model call that raises or is retried must not leave a client item stuck at `in_progress`.
* The wire contract is shared with baileyai and baileyai-chat-ui, which release separately.

## Considered Options

1. Keep emitting `llm_call` items only in debug mode and time them client-side.
2. Always emit `llm_call` start/done items, with content (`request`, `output`) still debug-only, and
   add a content-free `prepare` item for the pre-model-call phase.
3. Add a separate timing-only event type and leave `llm_call` debug-only.

## Decision Outcome

Chosen option: **2**.

* `llm_call` start/done items are always emitted. `runtime_seconds` is on the done item. With debug
  off the start item has no `request` key and the done item has no `output` key.
* Start and done items pair by the LangChain run id (`invocation_id`, item id
  `llm_<request_id>_<run_id>`). Durations use a monotonic clock, clamped at 0, and are omitted (not
  `null`) when no start was seen.
* An attempt that raises or is retried closes its open items with `status: "failed"` and the runtime
  so far.
* One `prepare` item per turn (`response.output_item.done` only): the time from request receipt to
  the first model call. It is emitted only when the caller sets `RequestInformation.request_started_at`.
* Chat completions is unaffected; the base wrapper methods are no-ops.

### Consequences

* Good: every Responses API client can show per-call and preparation timing; no content leaves the
  server unless debug is on.
* Good: all changes are additive (new item types, new `runtime_seconds` field).
* Bad: clients now receive `llm_call` items they did not before. Consumers must treat `request` and
  `output` as optional, and a start item with no done item means the call did not complete.
* Bad: with debug on and no streamed text, `output` is now `""`; with debug off it is omitted.
* Open: approval of baileyai ADR 019 and confirmation from the baileyai-chat-ui team that
  always-on items are accepted. This ADR stays `proposed` until a human records it.

## Pros and Cons of the Options

### Option 1: debug-only plus client-side timing

* Good: no wire change.
* Bad: the client sees only when events arrive, not when the model call started or ended
  server-side, and cannot measure the preparation phase at all.

### Option 2: always emit, content stays debug-only (chosen)

* Good: reuses the existing item shape and pairing; content gating is a single flag.
* Bad: widens what non-debug clients receive (mitigated by being additive and content-free).

### Option 3: separate timing-only event type

* Good: leaves `llm_call` semantics untouched.
* Bad: duplicates start/done pairing and run-id handling in a second item type, and the client would
  have to join two item types for one invocation.
