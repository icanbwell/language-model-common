# Opt-in adoption of the thread id as the Responses conversation id

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1264 (split out of BAI-1256; builds on BAI-1244 / ADR 0004)
* Date: 2026-10-23
* Governing documents: ADR 0004 (Option C, rejected there), plan `docs/superpowers/plans/2026-10-21-responses-conversation-field.md` (Open question 3), EA review of PR #135 (rules 04, 06, 08, 12).

## Context and Problem Statement

ADR 0004 added a caller-set `ResponsesApiRequestWrapper.conversation_id`, echoed as `conversation: { id }`. Its Option C, having the converter copy `RequestInformation.conversation_thread_id` onto the wrapper, was rejected: every consumer would start echoing a thread id it never asked to expose, and baileyai's thread id (`<sub>:chat-<hex>`) embeds the user subject. The EA review of the first adoption attempt (always on) raised the same exposure and asked for a rollout lever.

## Decision Drivers

* A client should be able to read the conversation id from the response body.
* Existing consumers must see no change unless they ask for it.
* The thread id can embed the user subject, so exposure must be a deliberate choice.
* Callers such as baileyai use `get_streaming_response_async` directly, not `call_agent_with_input`.

## Considered Options

* **A. Always on.** Every consumer echoes the thread id. Rejected: exposure without consent, no rollout lever.
* **B. Opt-in environment flag (chosen).** `ADOPT_THREAD_ID_AS_CONVERSATION_ID`, off by default; also the rollout and rollback lever.
* **C. Opaque id at the source.** baileyai sets `conversation_id` itself to an id without the subject. Compatible with B and not blocked by it; baileyai's decision.
* **D. Keep ADR 0004's decision (no adoption).** Leaves every caller to set the id itself.

## Decision Outcome

Option B.

* `LanguageModelCommonEnvironmentVariables.adopt_thread_id_as_conversation_id` (default `false`).
* `ChatRequestWrapper.adopt_conversation_id(*, conversation_id)`: no-op default. `ResponsesApiRequestWrapper` sets the id only when none is set (a caller-set id wins) and skips a thread id that fails `validate_conversation_id` (blank, padded, over-long, or containing control characters) with a warning that does not log the value.
* `LangGraphToOpenAIConverter` adopts `conversation_thread_id` when the flag is on, at both entry points that build a response: `get_streaming_response_async` and `call_agent_with_input`. The step is idempotent, so the streaming path through `call_agent_with_input` adopts twice harmlessly.

### Consequences

* With the flag off (the default) nothing changes.
* With it on, `conversation: { id: <thread id> }` appears on `response.completed` and the non-streaming body (and on `response.created` where emitted). In baileyai that id embeds the subject and is echoed only to the requesting caller.
* Not covered: `write_response` and `stream_response` take only a `request_id` and are used by callers for fallbacks (format_error, slash-command short-circuit, validation errors); those responses keep `conversation: null` unless the caller sets `conversation_id` or calls `adopt_conversation_id` first.
* A thread id that fails validation is skipped, so `conversation` stays `null` for that turn; baileyai only requires `x-chat-id` to start with `<sub>:`, so a long client-supplied id can hit this.
* The thread id also appears in the opt-in `log_input_and_output` log line and in any client or proxy log that captures event bodies.
* Test doubles that duck-type `ChatRequestWrapper` without subclassing must add `adopt_conversation_id`; subclasses inherit the no-op.
* Rollback: unset the flag.

## Test Plan

`tests/utilities/test_adopt_thread_id_environment_variable.py` (flag default and values), `tests/structures/openai/request/test_conversation_id_adoption.py` (adopt, caller-set wins, invalid ids including padded and control-character, value-free "failed validation" warning, chat completions no-op), `tests/converters/test_converter_conversation_id_adoption.py` (flag on and off; streaming entry point called directly, as baileyai does; `call_agent_with_input` streaming and non-streaming; unusable thread ids).
