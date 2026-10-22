# Open Responses streams with `response.created`, and deprecate `thread_id` / `user_id`

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1256 (follow-up to BAI-1244 / ADR 0004; plan ticket BAI-1237)
* Date: 2026-10-22
* Governing documents: ADR 0004; plan `docs/superpowers/plans/2026-10-21-responses-conversation-field.md` (PR #131, Open questions 1, 4 and 8). Builds on PR #132 (`conversation_id` on the Responses wrapper). Adopting the thread id as the conversation id was split out of this change and is decided separately.

## Context and Problem Statement

ADR 0004 added `ResponsesApiRequestWrapper.conversation_id` and deferred two items:

1. The live LangGraph stream never emits `response.created` (only `write_response` streams do), so the conversation id reaches a client only on `response.completed`, at the end of the stream.
2. `thread_id` and `user_id` on `ResponsesRequest` are accepted and never read.

## Decision Drivers

* The client should get the conversation id at the start of a stream.
* Existing callers must keep working (baileyai tests send `thread_id`; no abstract signature changes).

## Considered Options

### Item 1, event
* **A. Converter asks the wrapper for a stream-start event (chosen).** A no-op default `create_stream_start_sse_event` on `ChatRequestWrapper` (the pattern of `create_task_progress_sse_event`); the Responses wrapper returns `response.created`, chat completions returns `None`. No `isinstance` checks, no abstract signature change.
* **B. `isinstance` check in the converter.** Couples the converter to one wrapper. Not chosen.

### Item 2, deprecation
* **A. Deprecate, keep accepting (chosen).** `Field(deprecated=True)` (JSON schema `deprecated: true`), a `DeprecationWarning` attributed to the caller and a log line (field name only) when a non-null value is supplied.
* **B. Reject the fields.** Returns a 400 and breaks baileyai tests. Not chosen.

## Decision Outcome

* `ChatRequestWrapper.create_stream_start_sse_event(*, request_id, source)` (default `None`). `ResponsesApiRequestWrapper` returns its `response.created` event, with `conversation: { id }` when `conversation_id` is set and `conversation: null` otherwise. `_stream_resp_async_generator` yields it first, inside the stream's `try`, so a failure goes through the stream's error handling.
* `thread_id` and `user_id` stay accepted and ignored. A non-null value logs a warning and raises a `DeprecationWarning`; an explicit `null` or a `model_dump` round-trip does not.

### Consequences

* **New first chunk on live Responses streams.** `response.created` is now the first chunk of every live Responses stream (it was absent), with `conversation: null` unless the caller set `conversation_id`. Consumers that assume the first chunk is content, or that count events, must tolerate it. It is a single-line `data:` event, so baileyai's newline-splitting `message_id` rewrite is unaffected.
* `write_response` and `stream_response` already emit their own `response.created`; they are alternative paths, so there is no duplicate.
* Python hides `DeprecationWarning` by default outside `__main__` and shows it once per location, so the log line is what operators see. Test suites that run with warnings as errors will fail until they stop sending `thread_id` or `user_id`.
* Test doubles that duck-type `ChatRequestWrapper` without subclassing it must add `create_stream_start_sse_event`; subclasses inherit the no-op default.
* Rollback: stop relying on the event; a published version cannot be unpublished.

## Addendum (BAI-1263): history outcome in `conversation`

Additive, no change to the decisions above. `PipelineContext` gains `history_status` (`loaded` | `empty` | `failed`, default `None`) and `history_loaded` (prior messages loaded, default `0`). `ResponsesApiRequestWrapper` gains matching `history_status` / `history_loaded` properties that the caller copies from the context. When `history_status` is set and `conversation_id` is set, the `conversation` object on `response.created`, `response.completed` and the non-streaming body becomes `{ id, history_status, history_loaded }`; otherwise it is unchanged (`{ id }` or `null`), so existing callers get byte-identical payloads. A status without a conversation id is not emitted. The producer rejects unknown status values; the upstream `Conversation` model allows extra keys, so consumers reading a value added later do not fail. The package does not log or carry message content for this signal. Tests: `TestHistoryOutcomeEcho` in `tests/structures/openai/request/test_responses_api_request_wrapper.py`, `tests/pipeline/test_context_history_fields.py`.

## Test Plan

`tests/structures/openai/request/test_stream_start_sse_event.py` (event on both wrappers), `tests/converters/test_converter_stream_start_event.py` (stream opens with `response.created`; no extra chunk when the wrapper has none; a failing start event and a failing stream both go through error handling), `tests/schema/openai/test_responses_request_deprecated_fields.py` (warns per supplied field, silent otherwise, caller attribution, log without the value, schema marks `deprecated`).
