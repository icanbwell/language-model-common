# Converter adopts the thread id as the conversation id, opens Responses streams with `response.created`, and `thread_id`/`user_id` are deprecated

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1256 (follow-up to BAI-1244 / ADR 0004; plan ticket BAI-1237)
* Date: 2026-10-22
* Governing documents: ADR 0004; plan `docs/superpowers/plans/2026-10-21-responses-conversation-field.md` (PR #131, Open questions 1, 3, 4 and 8). Stacked on PR #132 and gated on the same Tech Design Review and EA decisions; do not merge or release before they clear.

## Context and Problem Statement

ADR 0004 added `ResponsesApiRequestWrapper.conversation_id` (caller-set) and deferred three items:

1. The live LangGraph stream never emits `response.created` (only `write_response` streams do), so the conversation id reaches a client only on `response.completed`, at the end of the stream.
2. `thread_id` and `user_id` on `ResponsesRequest` are accepted and never read.
3. ADR 0004 Option C (the converter copies `RequestInformation.conversation_thread_id` onto the wrapper) was rejected.

## Decision Drivers

* The client should get the conversation id at the start of a stream.
* Existing callers must keep working (baileyai tests send `thread_id`; no abstract signature changes).
* Privacy: the thread id can embed the subject (baileyai ids are `<sub>:chat-<hex>`).

## Considered Options

### Item 3, adoption
* **A. Converter adopts the thread id, always on (chosen, by explicit decision).** Every Responses consumer sees `conversation.id` equal to the thread id unless the caller already set one.
* **B. Opt-in through an environment variable.** Existing consumers see no change. Not chosen.
* **C. Keep ADR 0004's decision (no adoption).** Not chosen.

### Item 1, event
* **A. Converter asks the wrapper for a stream-start event (chosen).** A no-op default `create_stream_start_sse_event` on `ChatRequestWrapper` (the pattern of `create_task_progress_sse_event`); the Responses wrapper returns `response.created`, chat completions returns `None`. No `isinstance` checks, no abstract signature change.
* **B. `isinstance` check in the converter.** Couples the converter to one wrapper. Not chosen.

### Item 2, deprecation
* **A. Deprecate, keep accepting (chosen).** `Field(deprecated=True)` (JSON schema `deprecated: true`) and a `DeprecationWarning` when the field is supplied.
* **B. Reject the fields.** Returns a 400 and breaks baileyai tests. Not chosen.

## Decision Outcome

* `ChatRequestWrapper.adopt_conversation_id(*, conversation_id)` (no-op default) and `create_stream_start_sse_event(*, request_id, source)` (default `None`).
* `ResponsesApiRequestWrapper.adopt_conversation_id` sets the id only when none is set; a caller-set id wins. A blank or over-long id (the ADR 0004 validator) is skipped with a warning that does not log the value.
* `LangGraphToOpenAIConverter.call_agent_with_input` calls `adopt_conversation_id(conversation_id=request_information.conversation_thread_id)` before the streaming/non-streaming branch. `_stream_resp_async_generator` yields the stream-start event first.
* `thread_id` and `user_id` stay accepted and ignored.

### Consequences

* **Behavior change for all Responses consumers:** payloads carry `conversation: { id: <thread id> }` where they carried `null`. For baileyai the id embeds the subject; it is echoed only to the caller that sent the request. Consumers that do not want it must set `conversation_id` themselves or stop using the thread id.
* **New event on existing streams:** `response.created` is now the first chunk of every live Responses stream (it was absent). Consumers that assume the first chunk is content, or that count events, must tolerate it. The chunk is a single-line `data:` event, so baileyai's newline-splitting `message_id` rewrite is unaffected.
* Callers that already set `conversation_id` (baileyai Task 11) keep their value.
* **Limits of adoption:** it happens in `call_agent_with_input`, the converter's entry point. `write_response` and `stream_response` take only a `request_id` and are called by baileyai for its fallbacks (format_error, slash-command short-circuit, validation errors); those turns keep `conversation: null` unless baileyai calls `adopt_conversation_id` (or sets `conversation_id`) first. Direct callers of `get_streaming_response_async` also skip it.
* A thread id longer than 256 characters or blank is skipped (warning, value not logged) and `conversation` stays `null` for that turn; baileyai only requires `x-chat-id` to start with `<sub>:`, so a long client-supplied id can hit this.
* The thread id (which embeds the subject in baileyai) now also appears in the opt-in `log_input_and_output` log line and in any client or proxy log that captures event bodies.
* Test doubles that duck-type `ChatRequestWrapper` without subclassing it must add `adopt_conversation_id` and `create_stream_start_sse_event`; subclasses inherit the no-op defaults.
* Supplying a non-null `thread_id` or `user_id` logs a warning (field name only) on every request and raises a `DeprecationWarning` attributed to the caller. Python hides `DeprecationWarning` by default outside `__main__` and shows it once per location, so the log line is what operators see; test suites that run with warnings as errors will fail until those fields are dropped. An explicit `null` or a `model_dump` round-trip does not warn.
* Rollback: stop relying on the adoption (set `conversation_id` explicitly); a published version cannot be unpublished.

## Test Plan

`tests/structures/openai/request/test_conversation_id_adoption.py` (adopt, keep caller-set, skip invalid, no value in logs, chat completions no-op, stream-start event on both wrappers), `tests/converters/test_converter_conversation_id.py` (stream opens with `response.created`; no extra chunk when the wrapper has none; non-streaming adoption including caller-set, missing and blank thread ids), `tests/schema/openai/test_responses_request_deprecated_fields.py` (warns per supplied field, silent otherwise, schema marks `deprecated`).
