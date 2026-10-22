# `conversation` field on the Responses API models

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers
* Ticket: BAI-1244 (sub-task of BAI-1231; plan ticket BAI-1237)
* Date: 2026-10-21
* Governing documents: baileyai `docs/tech-designs/conversation-api-for-chat-ui-client.md`, item 7 (baileyai PR #543); plan `docs/superpowers/plans/2026-10-21-responses-conversation-field.md` (PR #131). The Tech Design Review ticket does not exist yet; the EA decisions are pending. This change ships as a draft PR until both are done.

## Context and Problem Statement

Callers of baileyai pass a conversation id in the `x-chat-id` header and read it back from `X-Conversation-Id`. Browsers cannot read that header without CORS `expose_headers`, and on streaming responses it is only returned when the client sent one. OpenAI's Responses API takes a `conversation` request parameter (an id string or `{ id }`) that cannot be combined with `previous_response_id`. Open Responses (2026-04-24) defines no conversation field and permits implementation-specific fields.

`ResponsesRequest` has `extra="forbid"`, so a `conversation` field is rejected today. `ResponsesApiRequestWrapper` builds the `Response` object for `response.created`, `response.completed` and the non-streaming body, and the openai `Response` model already has an optional `conversation: { id }` field that serializes as `null`.

Not confirmed: the format of OpenAI conversation ids, and whether OpenAI's Response object returns `conversation`.

## Decision Drivers

* Additive and backward compatible; no abstract `ChatRequestWrapper` signature changes (six converter call sites, the Chat Completions wrapper and baileyai implement or call them).
* The echoed id must be the id the server used; the backend (baileyai derives `<sub>:chat-<hex>`) decides it and may ignore the client's value.
* Match the OpenAI request parameter where possible.

## Considered Options

### Option A: New keyword on `create_first_sse_message`, `create_final_sse_message`, `create_non_streaming_response`
Explicit, but changes three abstract signatures, the Chat Completions implementation (which has no use for it) and every call site.

### Option B: Caller-set `conversation_id` property on `ResponsesApiRequestWrapper` (chosen)
Default `None`; the three `Response` builds read it. No signature changes; existing callers are unaffected.

### Option C: The converter copies `RequestInformation.conversation_thread_id` onto the wrapper
No work for callers, but every consumer would start exposing LangGraph thread ids it never asked to expose, and baileyai's id embeds the subject.

## Decision Outcome

Option B.

* `ResponsesRequest.conversation: str | ConversationParam | None`. `ConversationParam` is `{ id }` (`extra="forbid"`, frozen). The id is a non-blank string of at most `MAX_CONVERSATION_ID_LENGTH = 256` characters. Wrong types are rejected, not coerced.
* A validator rejects `conversation` together with `previous_response_id` when `previous_response_id is not None` (so an empty-string `previous_response_id` also conflicts).
* `ResponsesRequest.requested_conversation_id` returns the id in either form.
* `ResponsesApiRequestWrapper.conversation_id` (property and setter; the setter rejects blank or over-long values, `None` clears it). The package never copies the request's `conversation` onto it.
* When unset, payloads keep `conversation: null`, exactly as today (verified byte-identical on `main` for the three payloads).
* `extra="forbid"` stays; `conversation` is the only new accepted key. `thread_id` and `user_id` stay and are not aliased: nothing in this package or in baileyai code reads them, but baileyai tests send `thread_id`.
* Event chunk shape is unchanged (`output_text.done`, `response.completed`, `[DONE]`, single-line `data:` events), so baileyai's top-level `message_id` rewrite still works. `conversation` sits inside `response`, not at the event top level, and is not held until persistence.

### Consequences

* On a normal live agent turn only `response.completed` is emitted by this package; `response.created` comes from `write_response` streams (canned and error paths). Clients get the id at the end of the stream, or from the header earlier.
* baileyai must set `conversation_id` on the wrapper and add `conversation` to the events it builds by hand (`baileyai/services/pipeline/output/responses_api_output.py`).
* An old backend returns an error for a body `conversation` (baileyai maps the pydantic error to HTTP 400), so clients stay on the header until the new package is deployed.
* Rollback: stop setting `conversation_id` and ignore the field; a published version cannot be unpublished.

## Test Plan

`tests/schema/openai/test_responses_request_conversation.py` (accept string and `{id}`, reject both forms with `previous_response_id`, blank, over-long, extra keys, wrong types, unknown top-level field) and `TestConversationEcho` in `tests/structures/openai/request/test_responses_api_request_wrapper.py` (echo on all three builds and the streamed events, parity, unset stays null, request value not echoed, setter validation, terminal-chunk rewrite compatibility).

## Open Questions

* Tech Design Review ticket and EA decisions (unknown or other-user `conversation` id behavior, `thread_id` deprecation, the 256 bound).
* Should the converter emit an initial `response.created` on live streams (new event; not done here)?
