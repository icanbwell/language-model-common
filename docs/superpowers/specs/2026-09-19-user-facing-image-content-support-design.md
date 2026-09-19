# User-Facing Image Content Support Design

Extend `language-model-common`'s response layer (SSE and non-streaming) so an image
content block produced by a tool call, and already correctly converted onto the
LangChain message per BAI-678, actually reaches the OpenAI-compatible response
instead of being silently dropped or stringified.

**Status:** Proposed (Draft — not yet reviewed or accepted)
**Ticket:** BAI-806, Phase 3 of the cross-repo Tech Design Review EA-2651
(`mcp-fhir-agent` PR #736, `docs/design/tool-image-content-support.md`).
**Repos affected:** `language-model-common` (this implementation); `baileyai`
(dependency bump + two smaller call-site fixes, tracked separately as BAI-804).

## Problem

BAI-678 (shipped, `language-model-common` PR #83) fixed the *discovery-path*
tool-calling flow (`CallToolTool._arun`) so an MCP `ImageContent`/
`EmbeddedResource` reaches the LangChain `ToolMessage.content` as a real
`ImageContentBlock` (via `convert_call_tool_result`,
`languagemodelcommon/mcp/mcp_client/content_conversion.py`) instead of a text
placeholder. The direct adapter path (`mcp_tool_to_langchain_tool`,
`languagemodelcommon/mcp/mcp_client/langchain_adapter.py:94-104`) has done the
same since 2026-05-20. Either way, once the LLM turn completes, the resulting
`AIMessage`/`ToolMessage` content can legitimately contain an `ImageContentBlock`
(`type: "image"`, from `langchain_core.messages.content.create_image_block` —
keys `url`/`base64`/`mime_type`/`file_id`).

Nothing downstream of the model turn preserves that block for the *user-facing*
response:

1. **Streaming (`stream=true`).** `LangGraphStreamingManager._handle_on_chat_model_stream`
   (`languagemodelcommon/converters/streaming_manager.py:216-259`) calls
   `iter_message_content_text_chunks(content=chunk.content,
   include_non_text_placeholders=False)`. Non-text blocks land in
   `ContentChunks.non_text_blocks` but, with placeholders off, produce **zero**
   text chunks — nothing is ever yielded for them outside debug mode
   (`_handle_non_text_content_debug` only handles `reasoning`/`reasoning_content`,
   and only when `chat_request_wrapper.enable_debug_logging`). Even if it did
   yield something, the SSE frame it would ride on
   (`ChatCompletionApiRequestWrapper.create_sse_message`,
   `chat_completion_api_request_wrapper.py:208-247`) sets
   `ChoiceDelta.content` — typed `Optional[str]` by `openai`'s own
   `ChatCompletionChunk` schema — so an image cannot be forced through that
   field as a token-delta append even if we tried; the wire format itself
   forbids it, matching the design doc's framing that "images cannot be
   token-streamed."
2. **Non-streaming (`stream=false` on `/chat/completions` and `/responses`).**
   `utilities/chat_message_helpers.py`'s `langchain_to_chat_message`
   (lines 168-213): the `AIMessage` case calls
   `convert_message_content_to_string(message.content)`, which only extracts
   blocks with `type == "text"` — an image block is silently skipped (not
   even a placeholder). The `ToolMessage` case only inspects `message.artifact`
   (a legacy plain-string convention some non-MCP tools use) and ignores
   `message.content` entirely — for the MCP tool-calling paths in this repo,
   `response_format="content_and_artifact"` returns `(content_blocks, None)`
   (`langchain_adapter.py:94-105`), so `artifact` is always `None` and the
   entire content list — including any image block — is dropped, the
   `ToolMessage` never becomes a `ChatCompletionMessage` at all.

`openai`'s own `ChatCompletionMessage.content` field is also typed strictly
`Optional[str]` (verified against the installed `openai` package: `content:
Optional[str] = None`, no union with a content-part list, unlike the *request*-side
`ChatCompletionUserMessageParam.content` which does allow parts). Any additive
image-carrying content therefore has to go around, not through, plain
Pydantic field validation on that model — see Decision 3 below.

## Goals

- A `ToolMessageContentBlock`/`ImageContentBlock` (`type: "image"`, or the
  OpenAI/Anthropic-style `image_url`/`input_image`/`output_image` dict shapes
  `iter_message_content_text_chunks` already partially recognizes) present on
  an `AIMessage` or `ToolMessage` reaches the client in both the streaming and
  non-streaming response paths.
- Additive only. No existing field is removed, renamed, or retyped.
  `ChoiceDelta.content` stays `str | None`; `ChatCompletionMessage.content`'s
  declared type is untouched (see Decision 3 for how we still carry a list
  through it in practice); every current text-only test keeps passing
  unchanged.
- Reuse the existing content-block vocabulary
  (`languagemodelcommon/utilities/chat_message_helpers.py`'s
  `iter_message_content_text_chunks`, and the `ToolMessageContentBlock` union
  from `content_conversion.py`) rather than inventing a second, parallel
  image representation.
- Match the existing atomic-item precedent for things that can't be
  incrementally streamed: tool-call start/end already ride
  `create_tool_start_sse_event`/`create_tool_end_sse_event` as whole-payload
  `response.output_item.added`/`done` events on the Responses API wrapper,
  and are a no-op on the Chat Completions wrapper (which has no atomic-item
  transport at all — confirmed by reading both
  `chat_completion_api_request_wrapper.py` and
  `responses_api_request_wrapper.py`: only the latter overrides these hooks).
  Image output follows the same shape and the same asymmetry.

## Non-Goals

- **`baileyai` call-site fixes** (`tool_catalog_client.py`,
  `mcp_demographics_client.py` never importing `ImageContent`) — tracked as
  BAI-804, a separate, smaller change in a different repo.
- **`baileyai-skills-service`'s `/tool-catalog` proxy** collapsing content to
  placeholder text — a different repo/phase of EA-2651, not touched here.
- **`baileyai-chat-ui` rendering** — `useBaileyChat.ts`'s `applyEvent` only
  recognizes `item.type` of `function_call`/`mcp_call` on
  `response.output_item.added`/`done` today; an unrecognized `output_image`
  item type falls through to `return false` (a silent no-op, not a crash —
  verified by reading `src/logic/useBaileyChat.ts:227-249`). This is
  intentionally forward-compatible: this repo can ship the emission side now,
  and chat-ui's rendering side can land independently later, per this org's
  "additive, unknown-type-safe" schema evolution rule. Not part of BAI-806.
- **A payload-size ceiling for image content in the response layer.** ADR-0015
  (`mcp-fhir-agent`) already caps size at the MCP tool boundary; this repo's
  response layer does not add a second cap. Flagged as an open question below.
- **Chat Completions API atomic image events.** Per the existing
  tool-start/tool-end precedent, `ChatCompletionApiRequestWrapper` does not
  support atomic item events at all (`create_tool_start_sse_event`/
  `create_tool_end_sse_event` are unoverridden no-ops there); the new
  `create_image_output_sse_event` hook follows the same asymmetry rather than
  inventing atomic-item support for a transport that doesn't have it. A
  `stream=true` request against `/chat/completions` (not `/responses`) that
  triggers an image tool call still won't surface the image in that mode —
  only the non-streaming `/chat/completions` path (Decision 3) and both modes
  of `/responses` do.

## Alternatives Considered

### Option A: Force the image through `ChoiceDelta.content` as a data-URI-in-text chunk ❌ Rejected

Base64-encode the image into the same string path used for text deltas (one
big "delta" containing a markdown-image-like data URI). Rejected for the same
reasons the parent EA-2651 design doc rejected the equivalent at the
chat-ui layer (Option B there): it reintroduces multi-hundred-KB payloads into
a field every log/metric/middleware in this stack treats as bounded, human
text, and removes the `type: "image"` marker other code already filters on
(e.g. `_summarize_content_block` in `streaming_formatters.py`, which exists
specifically to avoid stringifying image blocks into trace/debug output).

### Option B: New atomic SSE event + additive non-streaming content-part list (chosen)

Emit a whole-payload SSE item event for images (mirroring the existing
tool-call item-event pattern) instead of forcing them through the token-delta
path, and extend the non-streaming message's `content` to a content-part list
only when an image is actually present. Consistent with how this repo already
treats "things that can't be usefully chunked" (tool call args/output).

### Option C: Only fix the non-streaming path; defer streaming ⚠️ Considered, rejected as incomplete

Streaming is the default and more heavily used path in production
(`stream=true` is baileyai's default per `docs/prompt-caching.md` and this
repo's own SSE-focused test suite). Shipping only the non-streaming half would
leave the more common path silently still dropping images — not an acceptable
partial fix for a ticket whose whole point is closing that gap.

## Decisions

### Decision 1 — `iter_message_content_text_chunks` gets an explicit `"image"` branch

Today, LangChain's own `ImageContentBlock` (`type: "image"`) falls into the
function's generic `else` branch (same bucket as any unrecognized type) —
functionally it does land in `ContentChunks.non_text_blocks` already, but only
by accident of the catch-all, with a generic `[image]`-shaped debug
placeholder computed from `content_item_type` rather than an intentional,
named branch. Add an explicit `elif content_item_type == "image":` case,
matching the style of the existing `image_url`/`input_image`/`output_image`
branches, so the block is recognized on purpose and the optional debug
placeholder is meaningful.

### Decision 2 — one shared image-part extractor: `extract_image_output_parts`

New function in `chat_message_helpers.py`:

```python
def extract_image_output_parts(
    non_text_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ...
```

Takes the `non_text_blocks` list `iter_message_content_text_chunks` already
produces and returns a list of normalized parts:

```python
{"type": "output_image", "image_url": "<url-or-data-uri>", "mime_type": "<mime or None>"}
```

Recognizes both shapes already in play in this stack:

- LangChain's `ImageContentBlock` (`type: "image"`): resolves `image_url` from
  `url` if present, else builds a `data:{mime_type};base64,{base64}` URI from
  `base64`+`mime_type` (the same shape `create_image_block` produces via
  `convert_call_tool_result`). A block with neither `url` nor `base64` (e.g. a
  bare `file_id` reference) is skipped rather than emitting a part with
  nothing to render.
- The OpenAI/Anthropic-style `image_url`/`input_image`/`output_image` dict
  shapes `iter_message_content_text_chunks` already partially recognized
  (`image_url` key either a dict with `url` or a bare string).

One function, reused by both the streaming and non-streaming code paths below
— no second, drifting implementation of "how do we get a URL out of an image
block."

### Decision 3 — non-streaming: additive content-part list via `model_construct`

New function `build_openai_message_content`:

```python
def build_openai_message_content(
    content: str | list[str | Dict[str, Any]],
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    chunks = iter_message_content_text_chunks(
        content=content, include_non_text_placeholders=False
    )
    text = "".join(chunks.text_chunks)
    image_parts = extract_image_output_parts(chunks.non_text_blocks)
    if not image_parts:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.extend(image_parts)
    return parts
```

Text-only content (the overwhelming majority of messages today) returns a
plain `str`, byte-for-byte the same value `convert_message_content_to_string`
would have produced — this is the non-breaking guarantee for every existing
consumer. Only when an image part is actually present does it return a list.

`langchain_to_chat_message` uses this for the `AIMessage` case (replacing
`convert_message_content_to_string`), and, for the `ToolMessage` case, uses it
against `message.content` (not just `message.artifact`) to recover image
parts the current `artifact`-only check drops for the MCP tool-calling paths
in this repo (where `artifact` is always `None`, per the Problem section).

**The `openai`-SDK type wrinkle.** `ChatCompletionMessage.content` is declared
`Optional[str]` in the installed `openai` package — confirmed via
`ChatCompletionMessage.model_fields["content"]`. Normal Pydantic construction
(`ChatCompletionMessage(role="assistant", content=[...])`) would raise a
validation error for a list. Since this is an *additive extension* of the
OpenAI-compatible contract, not a real OpenAI API, we accept passing outside
that declared type: when (and only when) `build_openai_message_content`
returns a list, construct via `ChatCompletionMessage.model_construct(role=
"assistant", content=parts)` — Pydantic's documented escape hatch that skips
field validation while still setting field defaults and serializing
(`model_dump()`/`model_dump_json()`) exactly what was assigned. Verified
directly against the installed `pydantic`/`openai` versions: `model_dump()`
on a `model_construct()`-built message with a list `content` serializes the
list correctly end to end through the outer `ChatCompletion.model_dump()`
call in `create_non_streaming_response`. The only observable side effect is a
`PydanticSerializationUnexpectedValue` `UserWarning` at serialization time
(the type checker/schema still nominally expects `str`) — noisy but harmless,
and not silenced here since suppressing it would need to reach into
`openai`'s own model, out of scope for a minimal, additive fix. Text-only
messages are entirely unaffected — they still go through the normal validated
`ChatCompletionMessage(...)` constructor with `content: str`, so this
wrinkle is scoped exclusively to the new, image-carrying branch.

### Decision 4 — streaming: new atomic `output_image` item event

`ChatRequestWrapper` (base class) gets a new hook, following the exact
pattern of `create_tool_start_sse_event`/`create_tool_end_sse_event`:

```python
def create_image_output_sse_event(
    self,
    *,
    request_id: str,
    image_part: dict[str, Any],
) -> str | None:
    """Emit an SSE event carrying an image content part atomically (not as a
    token delta -- images can't be incrementally streamed the way text can).
    Default no-op; see create_tool_end_sse_event for the same
    override-only-where-supported pattern."""
    return None
```

`ResponsesApiRequestWrapper` overrides it to emit a `response.output_item.done`
event with a new, custom item type `output_image` (not one of `openai`'s
built-in `Response` item types — this is this repo's own additive extension,
same as `function_call`'s use in `create_tool_start_sse_event` is already a
raw dict rather than a validated `openai.types.responses` model):

```python
{
    "type": "response.output_item.done",
    "output_index": 0,
    "sequence_number": <n>,
    "item": {
        "type": "output_image",
        "id": "img_<request_id>_<n>",
        "status": "completed",
        "image_url": "<url-or-data-uri>",
        "mime_type": "<mime or None>",
    },
}
```

`ChatCompletionApiRequestWrapper` does not override this hook (stays the
inherited no-op), matching the existing asymmetry for tool-call events (see
Non-Goals).

`LangGraphStreamingManager._handle_on_chat_model_stream` calls
`extract_image_output_parts(content_chunks.non_text_blocks)` right after
computing `content_chunks` (unconditionally — this is real response content,
not debug-only output, unlike `_handle_non_text_content_debug`, which stays
gated on `enable_debug_logging` for `reasoning`/`reasoning_content` blocks),
and for each part calls `chat_request_wrapper.create_image_output_sse_event(...)`,
yielding the result when non-`None`.

**Item type naming — `output_image`.** Chosen to match the parent EA-2651
design doc's own proposed naming ("a new `output_image` (or similarly named)
item," `docs/design/tool-image-content-support.md`'s Implementation Details
table) and the non-streaming content-part `"type": "output_image"` from
Decision 2/3, so the same string identifies "this is an image" whether it
arrives via SSE item or non-streaming content part.

## Rollout

1. Ship this change in `language-model-common`, released as usual via a
   GitHub Release (this repo's `python-publish.yml` sets `VERSION` from the
   release tag on publish — no manual `VERSION` file edit in this PR; the
   next tag after this merges, e.g. `3.0.7`, is the version to pin downstream).
2. `baileyai` (BAI-804, separate ticket/PR, not part of this change) bumps its
   `language-model-common` pin in `pyproject.toml`/`uv.lock` to that release
   and ships its two `ImageContent`-import call-site fixes
   (`tool_catalog_client.py`, `mcp_demographics_client.py`).
3. No `baileyai-chat-ui` change required for this phase to be safe to ship —
   unrecognized `output_image` items are ignored, not mis-rendered (Non-Goals).
4. No FDR/schema change — this is an OpenAI-compatible response contract
   extension, not FHIR data modeling.
5. No new dependency or vendor — reuses `langchain_core.messages.content`
   (already a runtime dependency, per BAI-678) and the already-installed
   `openai`/`pydantic` packages' documented `model_construct` API.

## Testing

- `iter_message_content_text_chunks`: parametrized case for `type: "image"`
  landing in `non_text_blocks` with no text chunk when placeholders are off,
  and a `[image]` placeholder when they're on (Decision 1).
- `extract_image_output_parts`: LangChain `image` block with `url`; with
  `base64`+`mime_type` (data-URI construction); with neither (skipped);
  OpenAI-style `image_url` dict; `input_image`/`output_image` with a bare
  string `image_url`; mixed list with a text block interspersed (ignored).
- `build_openai_message_content`: text-only content returns a `str` unchanged
  (regression); content with an image returns a list with a `text` part (when
  text is non-empty) followed by `output_image` part(s); image-only content
  (no text) returns a list with no `text` part.
- `langchain_to_chat_message`:
  - `AIMessage` with only text content — unchanged `str` result (regression
    against the existing test `test_ai_message_converts_to_chat_completion_message`).
  - `AIMessage` with a list content containing a text block and an image
    block — result is a `ChatCompletionMessage` whose `content` is a list
    containing both parts.
  - `ToolMessage` with `artifact` set and no image content — unchanged
    `f"\n[{artifact}]\n"` string result (regression against
    `test_tool_message_with_artifact_wraps_output`).
  - `ToolMessage` with `artifact` unset/empty and no image content — still
    returns `None` (regression against
    `test_tool_message_without_artifact_returns_none`).
  - `ToolMessage` with `content` containing an image block (the MCP
    `response_format="content_and_artifact"` shape, `artifact=None`) — now
    returns a `ChatCompletionMessage` carrying the image part instead of
    `None`.
- `ChatRequestWrapper.create_image_output_sse_event`: default (unoverridden)
  returns `None`.
- `ResponsesApiRequestWrapper.create_image_output_sse_event`: emits a
  `response.output_item.done` event with `item.type == "output_image"` and the
  expected `image_url`/`mime_type`.
- `ChatCompletionApiRequestWrapper`: no override exists, so no new test is
  needed beyond confirming (via the base-class test above) that it inherits
  the no-op.
- `LangGraphStreamingManager._handle_on_chat_model_stream`: an `AIMessageChunk`
  whose `content` is `[{"type": "image", ...}]` (no text) produces one call to
  `create_image_output_sse_event` and no text SSE chunk; a chunk with mixed
  text + image content produces both a text SSE chunk and an image SSE event;
  a text-only chunk produces no image event (regression against the existing
  `test_resuming_after_tool_call_inserts_missing_separator`-style behavior).

## Open Questions

| # | Question | Needed From | Impact |
|---|----------|-------------|--------|
| 1 | Should the response layer enforce its own image-size ceiling, or rely entirely on the MCP-boundary cap (ADR-0015 in `mcp-fhir-agent`)? | Imran / EA | Not blocking this change; flagged in the parent EA-2651 doc's own Open Question 2 for the analogous `baileyai-skills-service` truncation interceptor. |
| 2 | Does `baileyai-chat-ui` want `output_image` items surfaced in the trace panel (like `task.progress`) as an interim step before full inline rendering ships? | baileyai-chat-ui owners | Purely additive UI decision; doesn't change this repo's contract either way. |

## Related Work

- `mcp-fhir-agent` PR #736 / `docs/design/tool-image-content-support.md` (EA-2651) —
  the parent cross-repo Tech Design Review this ticket implements Phase 3 of.
- BAI-678 (`language-model-common` PR #83) — fixed the discovery-path
  tool-calling conversion this ticket builds on; establishes
  `convert_call_tool_result`/`ImageContentBlock` as the shared conversion
  pattern reused here.
- BAI-804 (`baileyai`, separate ticket) — the two downstream call-site fixes
  and the `language-model-common` version bump that consumes this change.
- `languagemodelcommon/mcp/mcp_client/content_conversion.py`,
  `languagemodelcommon/mcp/mcp_client/langchain_adapter.py` — the existing,
  correct MCP→LangChain conversion this ticket's response layer sits
  downstream of.

## Governance

Repo-local implementation decision extending an existing, already-adopted
content-block vocabulary (`langchain_core.messages.content`, `openai`'s own
SDK types used via its documented `model_construct` escape hatch) — no new
vendor/technology, so no Tech Design Review is triggered under the "is it
NEW?" test on its own; the cross-repo coordination itself is already covered
by EA-2651 at the parent design-doc level. This document stands in place of a
narrower ADR because the decision spans multiple call sites (streaming +
non-streaming, two message types) and benefits from the fuller
problem/alternatives structure; no separate ADR is planned unless review
surfaces a narrower, ADR-shaped sub-decision.
