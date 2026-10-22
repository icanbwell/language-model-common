# Responses API `conversation` field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept an OpenAI-style `conversation` request field on `ResponsesRequest` and echo the conversation id the server used on the `response.created` / `response.completed` response objects and in the non-streaming body, so callers can pass and read a conversation id in the body instead of only the `x-chat-id` / `X-Conversation-Id` headers.

**Architecture:** `ResponsesRequest` gains `conversation: str | ConversationParam | None` plus a validator that rejects it together with `previous_response_id`. `ResponsesApiRequestWrapper` gains a caller-set `conversation_id` property (default `None`); the three places that build an OpenAI `Response` object put `conversation={"id": ...}` on it when it is set. The package never decides the id: the backend (baileyai derives `<sub>:chat-<hex>`) sets it on the wrapper after it has resolved the id, so no abstract `ChatRequestWrapper` signature changes.

**Tech Stack:** Python 3.12+, pydantic v2, `openai` SDK types (`Response`, `Conversation`), pytest. Package `languagemodelcommon`, released to PyPI by GitHub Release.

**Spec:**
- `docs/tech-designs/conversation-api-for-chat-ui-client.md`, item 7 and "Required change in `language-model-common`" (baileyai repo, branch `origin/IQ-BAI-1237`, PR #543).
- `docs/superpowers/plans/2026-10-21-conversation-api-and-test-ui.md`, Task 10 (this plan expands it) and Task 11 (baileyai side) (baileyai repo, same branch).
- `docs/superpowers/specs/2026-10-21-chat-ui-bailey-client-design.md` section 6.7 and `docs/superpowers/plans/2026-10-21-chat-ui-bailey-client.md` Tasks 13-14 (baileyai-chat-ui repo, branch `origin/IQ-BAI-1237`).

Repo rules followed: branch `IQ-BAI-1237`; commit messages start `BAI-1237 `, no AI-attribution trailer, no `--no-verify` (`CLAUDE.md` Commit Messages, Code Ownership); keyword-only arguments, empty `__init__.py`, no in-process caches; no version bump or changelog edit in the PR, the version comes from the release tag (`CONTRIBUTING.md`, `playbooks/MERGING.md:16`); plans live in `docs/superpowers/plans/`, ADRs in `adrs/`.

## Global Constraints

- Additive only. No field, parameter, method or signature is removed or renamed. `ChatRequestWrapper` abstract method signatures do not change.
- Unknown fields stay forbidden except `conversation`: `ResponsesRequest` keeps `ConfigDict(extra="forbid")`; `conversation` is the only new accepted key.
- Backward compatible for existing callers: a caller that never sets `conversation` on the request and never sets `conversation_id` on the wrapper gets the same behavior as today (the response object's `conversation` stays `null`, as it serializes today).
- No PHI: the conversation id is an opaque identifier; it is never logged by this package with message text, and no test fixture contains real user data.
- Keyword-only arguments per the repo's rules: every new public function, method and constructor uses the `*` separator and every call site and test passes keywords (property setters are the framework-required exception).

## Verified current state (2026-10-21, `origin/main` 0654b17)

- `languagemodelcommon/schema/openai/responses.py`: `ResponsesRequest` has `model_config = ConfigDict(extra="forbid")`; fields include `previous_response_id: Optional[str]`, `store`, and, under "LangGraph context", `thread_id`, `user_id`, `metadata` (lines 45-50). No validators today.
- `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py` (875 lines) builds an openai `Response` in three places: `create_first_sse_message` (`Response(` at line 224, `type="response.created"` at 237), `create_final_sse_message` (`Response(` at 539, `type="response.completed"` at 553), `create_non_streaming_response` (`Response(` at 639). Each uses `model_dump_json()` / `model_dump(mode="json")` with default options.
- The installed `openai` `Response` model has `extra="allow"` and its own optional `conversation: Conversation | None` field (`Conversation` has only `id: str`), so today every response object already serializes `"conversation": null`. "Omit when unset" is therefore not today's behavior; "stays null" is.
- `create_first_sse_message` is called only from `ResponsesApiRequestWrapper.stream_response` (line 738), which is used by `ChatRequestWrapper.write_response`. The LangGraph streaming path (`languagemodelcommon/converters/langgraph_to_openai_converter.py`, `streaming_manager.py`) calls `create_final_sse_message` and `create_sse_message` only (six call sites) and never `create_first_sse_message`. baileyai's pipeline streams through the converter, so on the live agent stream `response.created` is not emitted by this package; `response.completed` is. `response.created` is emitted only for `write_response` streams (for example baileyai's slash-command short-circuit and error paths).
- `create_non_streaming_response` is called by `ChatRequestWrapper.write_response` / `write_non_streaming_response` and by the converter (`langgraph_to_openai_converter.py:518`).
- `RequestInformation.conversation_thread_id` (`languagemodelcommon/utilities/request_information.py:44`) is the LangGraph `thread_id` carrier; the converter reads it (`langgraph_to_openai_converter.py:1073,1088,1336`). It never reaches the wrapper.
- Tests: `tests/structures/openai/request/test_responses_api_request_wrapper.py` (helper `_make_wrapper`, `_make_env`), `test_tool_*`, `test_image_output_sse_event.py`, `test_mcp_app_sse_event.py` construct `ResponsesApiRequestWrapper`/`ResponsesRequest`. No test for `ResponsesRequest` alone exists; there is no `tests/schema/` directory.

## Findings the plan relies on

**`thread_id` on `ResponsesRequest`:** accepted and never read. In this package `grep` finds `request.thread_id` nowhere; every other `thread_id` hit is the LangGraph config key or `RequestInformation.conversation_thread_id`. In baileyai (`origin/main`) no code reads the body's `thread_id`; only tests send it (`tests/bailey_endpoints/test_bailey_router.py:783`, `tests_api_responses.py:175,194`) and they pass today, so removing the field would break them with a 400. `user_id` is likewise unread. Decision: keep both untouched, do not alias `conversation` onto `thread_id` (different semantics: `thread_id` is client-supplied and unused, `conversation` will be validated by the backend), add a description on `thread_id` stating it is accepted for compatibility and not used. Deprecation of `thread_id` is a separate decision (Open question 4).

**What depends on `extra="forbid"`:** baileyai's router validates with `ResponsesRequest.model_validate(chat_request)` (`baileyai/routers/api_router.py:434`, comment says validators must run); pydantic's `ValidationError` is a `ValueError`, and the router's `except ValueError` returns HTTP 400 with `{"error": {"type": "invalid_request_error"}}` (`api_router.py:525-540`), not 422. So today an unknown field, and tomorrow `conversation` + `previous_response_id`, is a 400 from baileyai, not the 422 the design doc states. Tests in baileyai that use `ResponsesRequest.model_construct(...)` (`tests/routers/test_api_router_extended.py:1089`, `tests/services/message_preprocessing/test_slash_command_hidden_metadata.py:52`, `tests/services/pipeline/test_assemble_responses_messages_step.py:35`) bypass validators, so the new validator is not exercised there.

**Consumers and compatibility surface:** baileyai (`language-model-common>=3.0.37`, locks 3.0.37; uses `ResponsesRequest`, `ResponsesApiRequestWrapper`, `write_response`, `create_final_sse_message`); language-model-gateway (archived 2026-10-16; its `chat_completion_router.py` imports both classes at its pinned version, unaffected); baileyai-skills-service and mcp-fhir-agent (no `ResponsesRequest` / wrapper import found on `origin/main`). Other unchanged surface: new optional field, new optional property, same abstract signatures, same event types. Existing payloads keep `"conversation": null` until a caller sets the id.

**Coexistence with baileyai's `message_id` rewrite (PR #541, head 7614df7):** `baileyai/services/pipeline/output/responses_api_output.py` holds the chunk returned by `create_final_sse_message` (`hold_terminal_chunk`), and `baileyai/services/pipeline/committed_message.py::attach_message_id_to_completed_event` later `json.loads` each `data:` line whose `type == "response.completed"`, sets top-level `event["message_id"]`, and re-dumps; `attach_message_id_to_body` does the same for the non-streaming JSON body. Both are round-trip rewrites that preserve every other key, so `response.conversation.id` written by this package survives unchanged and `message_id` stays a sibling of `response` (not inside it). The `conversation` field is NOT held: the id is resolved in `get_request_information` before the graph runs, so it is already on the wrapper when the terminal chunk is built, and it must also appear in `response.completed` if persistence fails (the rewrite then adds no `message_id`, `conversation` is still present). Constraint for this plan: the chunk shape stays `data: <single-line JSON>\n\n` ×3 (`output_text.done`, `response.completed`, `[DONE]`), because the rewrite splits on `\n` and matches `"response.completed" in chunk`.

## Gate: Tech Design Review and EA decisions (nothing merges or publishes before this)

This plan changes a public request/response contract consumed by baileyai and baileyai-chat-ui. `CLAUDE.md` ("Process References") requires a Tech Design Review for public API and cross-team changes, and requires the governing document to be referenced in the PR and commit message. The baileyai plan lists the review ticket and any EA exception as an unresolved blocker (its Open question 4, "Blocks implementation").

- [ ] **Gate 0 (before Task 1):** a Tech Design Review ticket (type "Tech Design Review", EA project) exists for the baileyai design `docs/tech-designs/conversation-api-for-chat-ui-client.md` (item 7), and its status is approved. Record the ticket key in the ADR (Task 3), in the PR description, and in each commit body.
- [ ] **Gate 0b:** the EA decisions that shape the contract are answered by the owners: unknown or other-user `conversation` id behavior (baileyai design question 6), `thread_id` deprecation (Open question 4 below), and the id length bound (Open question 2).
- An executor that cannot confirm Gate 0 stops. Implementing Tasks 1-3 on a branch is allowed only as a draft PR; do not mark it ready, merge it, or run Task 4 until the gate is confirmed by the user. Who may create a GitHub Release in this repo is `unverified` (`playbooks/DEPLOYMENT.md`, item 6); Task 4 Step 1 is a user action.

## Review Focus

Failure modes the spec implies but no happy-path test would exercise, most likely first. Each has a named test in the owning task.

1. `conversation` and `previous_response_id` both set (either form of `conversation`): must be rejected with a `ValidationError`, not silently prefer one. Task 1 `test_conversation_with_previous_response_id_is_rejected`.
2. `conversation: ""`, `{ "id": "" }`, whitespace-only, over-length, or `{ "id": "x", "extra": 1 }`: rejected, never reaches the backend as an empty or unbounded id. Task 1 `test_invalid_conversation_values_are_rejected`.
3. `conversation` of the wrong type (`123`, `["a"]`, `{ "id": 123 }`, `true`): rejected, not coerced to a string. Task 1 `test_wrong_typed_conversation_is_rejected`.
4. Streaming vs non-streaming parity: the same id appears on `response.created`, `response.completed`, and the non-streaming body, and in the stream-built and `write_response`-built payloads alike. Task 2 `test_conversation_id_is_identical_across_all_three_payloads`.
5. An old caller that never sets the new property (or never sets `conversation` on the request): `conversation` stays `null` in all three payloads, the client-supplied `conversation` is never echoed by the package on its own (the server may have ignored it), and the existing 73 wrapper tests pass unchanged. Task 2 `test_unset_conversation_id_leaves_payload_conversation_null` and `test_request_conversation_is_not_echoed_unless_the_caller_sets_conversation_id`.

Also covered: a blank id set on the wrapper is rejected (Task 2 `test_setting_blank_conversation_id_raises`), and the `message_id`-style rewrite compatibility (Task 2 `test_completed_chunk_survives_a_top_level_key_rewrite`).

## File structure

- Modify `languagemodelcommon/schema/openai/responses.py`: `ConversationParam`, `MAX_CONVERSATION_ID_LENGTH`, the `conversation` field, `requested_conversation_id`, the exclusivity validator, the `thread_id` description.
- Modify `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py`: `conversation_id` property and setter, `_conversation_payload`, the three `Response(...)` constructions.
- Create `tests/schema/__init__.py`, `tests/schema/openai/__init__.py` (both empty), `tests/schema/openai/test_responses_request_conversation.py`.
- Modify `tests/structures/openai/request/test_responses_api_request_wrapper.py`: new test classes.
- Create `adrs/0004-responses-conversation-field.md` (MADR, Task 3).

---

### Task 1: `conversation` on `ResponsesRequest`

**Files:**
- Modify: `languagemodelcommon/schema/openai/responses.py`
- Test: `tests/schema/openai/test_responses_request_conversation.py` (plus the two empty `__init__.py` files)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MAX_CONVERSATION_ID_LENGTH: int = 256` (module constant; Open question 2)
  - `class ConversationParam(BaseModel)`: `model_config = ConfigDict(extra="forbid", frozen=True)`; `id: str` with `min_length=1`, `max_length=MAX_CONVERSATION_ID_LENGTH`, non-blank.
  - `ResponsesRequest.conversation: Optional[Union[str, ConversationParam]] = None` (same length and non-blank constraints on the string form).
  - `ResponsesRequest.requested_conversation_id -> Optional[str]` (property: the string form or `conversation.id`, else `None`).
  - Validator `_reject_conversation_with_previous_response_id(self) -> "ResponsesRequest"` (`@model_validator(mode="after")`): raises `ValueError` when `conversation is not None and previous_response_id is not None`.

- [ ] **Step 1: Write the failing tests** in `tests/schema/openai/test_responses_request_conversation.py` (parametrize per repo rules; build requests with `ResponsesRequest.model_validate({...})`):
  - `test_conversation_string_is_accepted`: `{"model": "m", "input": "hi", "conversation": "u1:chat-ab12"}` validates; `requested_conversation_id == "u1:chat-ab12"`.
  - `test_conversation_object_is_accepted`: `"conversation": {"id": "u1:chat-ab12"}` validates; `requested_conversation_id == "u1:chat-ab12"`.
  - `test_request_without_conversation_is_unchanged`: `conversation is None`, `requested_conversation_id is None`, `model_dump(mode="json")["conversation"] is None`.
  - `test_conversation_with_previous_response_id_is_rejected` (parametrized over string and `{id}` forms): `pytest.raises(ValidationError)`, message contains `previous_response_id`.
  - `test_invalid_conversation_values_are_rejected` (parametrize): `""`, `"   "`, `{"id": ""}`, `{"id": "  "}`, `"x" * 257`, `{"id": "x" * 257}`, `{"id": "a", "extra": 1}`, `{}`; each raises `ValidationError`. A 256-character id is accepted.
  - `test_wrong_typed_conversation_is_rejected` (parametrize): `123`, `True`, `["a"]`, `{"id": 123}`, `{"id": None}`; each raises `ValidationError`.
  - `test_unknown_top_level_field_is_still_rejected`: `{"model": "m", "input": "hi", "conversations": "x"}` raises `ValidationError` (`extra="forbid"` intact).
  - `test_thread_id_and_user_id_are_still_accepted`: request with `thread_id="t"`, `user_id="u"` validates and `requested_conversation_id is None` (thread_id is not aliased).
- [ ] **Step 2: Run to verify failure**
  Run: `~/git/language-model-common/.venv/bin/python -m pytest tests/schema/openai/test_responses_request_conversation.py -q -p no:cacheprovider` from the worktree root.
  Expected: FAIL (the field is an unknown extra today).
- [ ] **Step 3: Implement** the interface above in `responses.py`. Use `Annotated[str, StringConstraints(min_length=1, max_length=MAX_CONVERSATION_ID_LENGTH)]` plus a `field_validator` that rejects whitespace-only values for both forms; keep the new field next to `previous_response_id`; add `Field(description=...)` on `thread_id` saying it is accepted for compatibility and not used by this package. No coercion: do not enable `strict=False` overrides.
- [ ] **Step 4: Run to verify pass**, same command. Expected: PASS. Then `~/git/language-model-common/.venv/bin/python -m pytest tests/structures/openai/request -q -p no:cacheprovider`. Expected: 73 passed (unchanged).
- [ ] **Step 5: Commit**
  ```bash
  git add languagemodelcommon/schema/openai/responses.py tests/schema
  git commit -m "BAI-1237 Accept a conversation field on ResponsesRequest"
  ```

---

### Task 2: echo the conversation id on the response objects

**Files:**
- Modify: `languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py` (`__init__` near line 85; `create_first_sse_message` line 220; `create_final_sse_message` line 523; `create_non_streaming_response` line 606)
- Test: `tests/structures/openai/request/test_responses_api_request_wrapper.py`

**Interfaces:**
- Consumes: Task 1's `ResponsesRequest.conversation` (only for the test that proves it is not echoed). The `openai.types.responses.response.Conversation` model.
- Produces (baileyai Task 11 and this repo's tests use these):
  - `ResponsesApiRequestWrapper.conversation_id` property `-> Optional[str]`, default `None`; setter `conversation_id.setter(value: Optional[str]) -> None` raising `ValueError` for an empty or whitespace-only string and for a value longer than `MAX_CONVERSATION_ID_LENGTH`; `None` clears it.
  - `ResponsesApiRequestWrapper._conversation_payload(self) -> Optional[Conversation]` (private): `Conversation(id=self._conversation_id)` or `None`.
  - Unchanged: every `create_*` signature. The three `Response(...)` constructions pass `conversation=self._conversation_payload()`.
  - Serialized shape: `"conversation": {"id": "<id>"}` on the response object when set, `"conversation": null` when not. Nothing is added at the event top level, so baileyai's `message_id` stays a top-level sibling.

Why a wrapper property, not a keyword on the methods: the three methods are abstract on `ChatRequestWrapper` and implemented by `ChatCompletionApiRequestWrapper` too, and are called from six converter call sites and from baileyai; a keyword would touch all of them and force a Chat Completions implementation that has no use for it. The converter is deliberately not changed to set the property from `request_information.conversation_thread_id` (Open question 3).

- [ ] **Step 1: Write the failing tests** (extend `_make_wrapper` with `conversation: str | dict[str, Any] | None = None`; keep its other behavior):
  - `TestConversationEcho`:
    - `test_created_event_carries_conversation_id`: set `wrapper.conversation_id = "u1:chat-ab12"`; parse `create_first_sse_message(request_id="r", source="t")` (strip `data: `); assert `event["type"] == "response.created"` and `event["response"]["conversation"] == {"id": "u1:chat-ab12"}`.
    - `test_completed_event_carries_conversation_id`: same for `create_final_sse_message(request_id="r", usage_metadata=None, source="t")`, picking the `response.completed` line; also assert the chunk still has exactly three `data:` lines ending in `data: [DONE]`.
    - `test_non_streaming_body_carries_conversation_id`: `create_non_streaming_response(request_id="r", json_output_requested=False, responses=[AIMessage(content="ok")])["conversation"] == {"id": "u1:chat-ab12"}`.
    - `test_conversation_id_is_identical_across_all_three_payloads` (Review Focus 4): the three ids are equal, and a `write_response(request_id="r", response_messages=[AIMessage(content="ok")])` streamed through `stream_response` carries the same id on both its `response.created` and `response.completed`.
    - `test_unset_conversation_id_leaves_payload_conversation_null` (Review Focus 5): without setting the property, all three payloads have `conversation is None` and the existing keys (`id`, `model`, `status`, `usage`) are unchanged.
    - `test_request_conversation_is_not_echoed_unless_the_caller_sets_conversation_id` (Review Focus 5): wrapper built from a request with `conversation="u1:chat-zz"` (no `previous_response_id`) and no `conversation_id` set: all three payloads have `conversation is None`.
    - `test_setting_blank_conversation_id_raises` (parametrize `""`, `"  "`, over-length): `pytest.raises(ValueError)`; `None` after a valid set clears it.
    - `test_completed_chunk_survives_a_top_level_key_rewrite`: take the `response.completed` line, `json.loads`, add `event["message_id"] = "m1"`, `json.dumps`, parse again: `response.conversation.id` unchanged and `message_id` is top-level (mirrors baileyai `attach_message_id_to_completed_event`).
- [ ] **Step 2: Run to verify failure**
  Run: `~/git/language-model-common/.venv/bin/python -m pytest tests/structures/openai/request/test_responses_api_request_wrapper.py -q -p no:cacheprovider -k "ConversationEcho"`
  Expected: FAIL (`conversation_id` does not exist).
- [ ] **Step 3: Implement** the property/setter and `_conversation_payload` on `ResponsesApiRequestWrapper`, initialize `self._conversation_id: Optional[str] = None` in `__init__`, and pass `conversation=self._conversation_payload()` in the three `Response(...)` calls. Validation reuses `MAX_CONVERSATION_ID_LENGTH` from `languagemodelcommon.schema.openai.responses`. Import `Conversation` from `openai.types.responses.response`.
- [ ] **Step 4: Run to verify pass**
  Run: `~/git/language-model-common/.venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/mcp -x` then the mypy and ruff checks below.
  Expected: PASS (full suite apart from tests that need external services; if any fails on `main` too, record it).
- [ ] **Step 5: Commit**
  ```bash
  git add languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py tests/structures/openai/request/test_responses_api_request_wrapper.py
  git commit -m "BAI-1237 Echo the conversation id on Responses API response objects"
  ```

---

### Task 3: ADR and pre-commit gate

**Files:**
- Create: `adrs/0004-responses-conversation-field.md`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: the recorded decisions that Open questions 1-4 resolve.

- [ ] **Step 1: Write the ADR** in MADR format (copy the section layout of `adrs/0003-configurable-mcp-protocol-negotiation-mode.md`), Ticket BAI-1237. Options considered for how the server's id reaches the payloads: (a) new keyword on the three abstract `create_*` methods, (b) caller-set property on the Responses wrapper (chosen), (c) converter copies `request_information.conversation_thread_id` automatically. Also record: echo only the id the server used, `conversation` and `previous_response_id` exclusive, `thread_id` kept and not aliased, `Response.conversation` was already `null` in payloads, standards position (Open Responses 2026-04-24 defines no conversation field and permits implementation-specific fields; OpenAI's `conversation` request parameter is mirrored; whether OpenAI's Response object returns `conversation` and the ID format are not confirmed). Cite the approved Tech Design Review ticket (Gate 0) and the baileyai tech design (PR #543) as the governing documents.
- [ ] **Step 2: Run the repo gates** from the worktree root: `~/git/language-model-common/.venv/bin/python -m ruff check languagemodelcommon tests`, `~/git/language-model-common/.venv/bin/python -m ruff format --check languagemodelcommon tests`, `~/git/language-model-common/.venv/bin/python -m mypy --strict --python-version=3.12 languagemodelcommon/schema/openai/responses.py languagemodelcommon/structures/openai/request/responses_api_request_wrapper.py`. Expected: no findings. The canonical full run is `make run-pre-commit` and `make tests` (Docker); CI runs both (`.github/workflows/build_and_test.yml`).
- [ ] **Step 3: Commit**
  ```bash
  git add adrs/0004-responses-conversation-field.md
  git commit -m "BAI-1237 Add ADR for the Responses conversation field"
  ```
  The commit hook (`pre-commit-hook`, Docker) must pass; do not use `--no-verify`.

---

### Task 4: release and consumer bump (no code in this repo)

**Interfaces:**
- Consumes: Gate 0 and 0b confirmed by the user, and Tasks 1-3 merged to `main` through a PR (branch protection; `CLAUDE.md` Branch Protection). The PR needs the org risk/type/semver labels (`pr-label-classifier` if not scaffolded; semver is minor: new optional field and property).
- Produces: a published `language-model-common` version containing the change (next after the latest tag `3.0.38`, so `3.0.39` unless another release lands first; the tag must be valid PEP 440).

- [ ] **Step 1: Release (precondition: Gate 0 confirmed; a published version cannot be unpublished).** A person creates a GitHub Release with tag `3.0.39` in `icanbwell/language-model-common`; `.github/workflows/python-publish.yml` (`on: release: created`) writes the tag into `VERSION`, runs `uv build`, publishes to PyPI via trusted publishing. `VERSION` in the repo (`0.0.1`) is a placeholder and is not edited. Release notes go on the Release page (no `CHANGELOG.md`). Watch `Upload Python Package` to green; merging does not publish (`playbooks/MERGING.md:33`).
- [ ] **Step 2: Verify publication.** Confirm the version on PyPI (`pip index versions language-model-common` or the PyPI project page) before telling baileyai.
- [ ] **Step 3: baileyai bump (baileyai Task 11 or a bump PR before it).** In `baileyai/pyproject.toml` raise the floor `"language-model-common>=3.0.37"` to `">=3.0.39"` (line 57) and re-lock `uv.lock` (currently `language-model-common` 3.0.37). `[tool.uv.exclude-newer-package] language-model-common = "0 days"` already lets a fresh release resolve at once. Re-lock command: the baileyai `Makefile` target `update-fast` runs `uv lock --upgrade` in a container and needs `JFROG_READ_TOKEN` (prior bumps: `BAI-1130 bump language-model-common to 3.0.37 (#509)`); a targeted `uv lock --upgrade-package language-model-common` is the minimal-diff form but its offline/JFrog behavior is unverified. Commit style from history: `BAI-<ticket> bump language-model-common to 3.0.39`.
- [ ] **Step 4: Verify in baileyai** with its own suite against the new version, then its normal deploy and the post-deploy `Security Regression` check.

## Compatibility matrix

| | Old baileyai (floor 3.0.37, never sets `conversation_id`) | New baileyai (Task 11, sets `conversation_id`, reads `conversation`) |
|---|---|---|
| **Old package (<= 3.0.38)** | Today's behavior. A body `conversation` is a 400 (pydantic `ValidationError` is a `ValueError` in the router). Payloads carry `conversation: null`. | Not possible to ship: baileyai Task 11 requires the new package (floor bump). A client that sends `conversation` to a backend still on the old package gets 400, so the chat-ui stays on `conversationVia: 'header'`. |
| **New package (>= 3.0.39)** | Accepts `conversation` in the request model, but baileyai ignores the value (it reads only `x-chat-id`) and never sets `conversation_id`, so payloads still carry `conversation: null`. No behavior change. | Full feature: body id accepted and validated by baileyai, the used id echoed on `response.completed` (and `response.created` where emitted) and in the body, `X-Conversation-Id` still set. |

Chat-ui clients: an old client keeps working on the header against any backend; a new client with `conversationVia: 'header'` (the default) keeps working against any backend; `'body'` requires the new package and new baileyai deployed everywhere the host runs.

## Rollback

All changes are additive. If a consumer regresses after the bump, per `playbooks/DEPLOYMENT.md`: stop that consumer's bump, pin it back to the previous version in its `pyproject.toml` and relock, or fix forward with a new release. A published PyPI version cannot be replaced or re-uploaded, so there is no unpublish. Inside baileyai, rollback of the feature alone is not setting `conversation_id` and ignoring the body field; the package change stays inert. The header path is untouched throughout.

## Cross-repo dependencies

| Item | This repo (Tasks 1-4) | baileyai (Task 11 of its plan; PR #543 design) | baileyai-chat-ui (plan `2026-10-21-chat-ui-bailey-client.md`) |
|---|---|---|---|
| Request field accepted | Task 1 `ResponsesRequest.conversation` | reads `requested_conversation_id`, validates `<sub>:` ownership, rejects header/body mismatch, maps model errors to the client error contract | Task 13 sends `conversation: '<id>'` (string form) in the body when `conversationVia === 'body'` |
| Exclusive with `previous_response_id` | Task 1 validator | router returns 400 today for a `ValueError` (design says 422; decide) | none |
| Id echoed on events and body | Task 2 `conversation_id` on the wrapper | sets `wrapper.conversation_id` after resolving the id (for example in `AuthenticateStep` next to `context.conversation_id = ri.conversation_thread_id`); also adds `conversation` to its hand-built events (`_error_sse_events`, `_security_block_sse` in `responses_api_output.py`) and to `format_error` paths that use `write_response` | Task 14 reads `conversation.id` from `response.created` and `response.completed`, falls back to `X-Conversation-Id` |
| `message_id` rewrite coexistence | tests prove the chunk survives a top-level key rewrite | `attach_message_id_to_*` unchanged; add a test that `response.conversation.id` and `message_id` are both present after the rewrite | Task 14 must not assume `conversation` is top-level |
| Release | Task 4 | floor bump and relock before Task 11 | pre-release after backend is on dev |

Order: this repo merged and released, then baileyai floor bump and Task 11 to dev, then chat-ui Tasks 13-14, then `conversationVia: 'body'` in dev.

## Open questions

1. `response.created` is not emitted by the live LangGraph stream (only by `write_response` streams), so chat-ui Task 14's "set from `response.created`" never fires on a normal turn; the id arrives on `response.completed`, at the end of the stream. Is that acceptable for a new conversation (the header covers it earlier if baileyai item 1 ships), or should the converter emit an initial `response.created`? That is a new event on existing streams; not in this plan. Owners: baileyai, chat-ui.
2. Maximum id length: 256 is proposed (baileyai ids are `<sub>:chat-<8 hex>`, and `sub` length is not bounded here). Confirm against the longest real `sub`.
3. Should the converter set `conversation_id` automatically from `request_information.conversation_thread_id`? Not done: it would start exposing thread ids to every consumer that never asked, and baileyai's `<sub>:chat-...` shape embeds the subject. Revisit only if a second consumer wants it.
4. Is `thread_id` (and `user_id`) on `ResponsesRequest` to be deprecated? Unused in this package and in baileyai code, but baileyai tests send `thread_id`. Not changed here.
5. Exclusivity uses `previous_response_id is not None`; an empty-string `previous_response_id` therefore also conflicts. Confirm that is intended.
6. Does the `openai>=2.5.0` floor in `pyproject.toml` include `Response.conversation` / `Conversation`? Verified present in 2.41.0 (this repo's lock) and 2.54.0 (local venv), not checked for 2.5.0. If absent, raise the floor in Task 2.
7. Not confirmed anywhere: the format of OpenAI conversation ids and whether OpenAI's Response object returns `conversation`; the field name and `{ id }` shape here follow the request parameter and the SDK's `Response.conversation` model.

## Self-review

- Spec coverage: request field + validator (Task 1); echo on `response.created`, `response.completed`, non-streaming (Task 2); mechanism for receiving the server's id (Task 2, ADR options in Task 3); `message_id` coexistence (Findings, Task 2 test, dependency table); `thread_id`, `extra="forbid"` dependents, consumers, release and bump (Findings, Task 4); compatibility matrix, rollback, open questions, cross-repo table present.
- Step scan: every code step names the file, signature, and constraint values; tests list names and assertions; bodies are left to the implementer.
- Type consistency: `conversation_id`, `_conversation_payload`, `requested_conversation_id`, `MAX_CONVERSATION_ID_LENGTH`, `ConversationParam` are used with the same names in Tasks 1-3, the findings, and the dependency table.
- Review Focus: five items, each with a named test (1-3 in Task 1, 4-5 in Task 2).
- Proportion: roughly spec length plus verified findings; no function bodies.
- Corrections to the upstream design and plan found while verifying: payloads already contain `"conversation": null` (Task 10's "omit it otherwise" should read "stay null"); baileyai returns 400, not 422, for model validation errors; `response.created` is not on the live stream.
- Not verified: Docker `make tests` and the Docker pre-commit hook were not run for this plan; the test commands above use the main checkout's venv from the worktree root (cwd precedes site-packages, confirmed to import the worktree's `languagemodelcommon`); `uv run` was not tried (fails offline per notes); the `uv lock` form for baileyai.
