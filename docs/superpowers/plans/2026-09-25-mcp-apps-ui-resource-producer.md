# MCP Apps (SEP-1865) — `language-model-common` SSE Type Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `type` discriminator to the `event: mcp_app` SSE payload so
`baileyai-chat-ui`'s parser (which dispatches on the JSON payload's own
`type` field, not the SSE `event:` line) can recognize it.

**Ticket:** BAI-960

## Correction — this plan originally proposed building a UI-resource
producer from scratch in this repo. That producer already exists — it was
built here in April–May 2026, then deliberately moved to
`baileyai-skills-service` under BAI-730 (commit `b9b5042`) and is live on
that repo's `main` today. See `baileyai`'s `adrs/010-mcp-apps-support.md`
for the full correction. This repo's *only* real remaining gap is the
missing `type` field below — everything else in the original version of
this plan (a new `ui_resource.py`, threading a fetch through
`mcp_tool_to_langchain_tool`, retiring `MCP_APP_PROXY_BASE_URL`) is
unnecessary. `MCP_APP_PROXY_BASE_URL` is **not dead** — it's the intended
wiring point for `baileyai-skills-service`'s proxy route (BAI-961), which
just hasn't been connected yet in that repo, not this one.

**Tech Stack:** Python 3.13, `pytest`.

## Global Constraints

- Keyword-only public functions; keyword-call-sites.
- Additive only — this must not change the payload shape for any field that
  already exists, only add new ones, so a client on an older version of
  `baileyai-chat-ui` (or any other consumer) is unaffected.

---

### Task 1: Add `type`/`protocolVersion` to `create_mcp_app_sse_event`

**Files:**
- Modify: `structures/openai/request/chat_request_wrapper.py` (abstract
  base + its default no-op — this one doesn't build a real payload, but its
  docstring/signature should still mention the new params for consistency)
- Modify: `structures/openai/request/chat_completion_api_request_wrapper.py`
- Modify: `structures/openai/request/responses_api_request_wrapper.py`
- Modify: `tests/converters/test_tool_event_handlers.py`'s stub override
  (line ~70) to accept (and ignore, or assert on) the new kwargs, matching
  its existing `**kwargs: Any` signature — likely needs no change at all
  since it already accepts arbitrary kwargs; verify at implementation time.
- Test: extend the existing SSE-event tests for both concrete wrapper
  subclasses.

**Design:**

```python
MCP_APPS_PROTOCOL_VERSION = "2026-01-26"  # module-level constant, matches SEP-1865's version string

def create_mcp_app_sse_event(
    self,
    *,
    html: str,
    title: str | None = None,
    csp: dict[str, Any] | None = None,
    permissions: dict[str, Any] | None = None,
    prefers_border: bool | None = None,
    display_mode: str | None = None,
    resource_uri: str | None = None,  # new
) -> str | None:
    payload: Dict[str, Any] = {
        "type": "mcp_app",  # new — the actual fix
        "protocolVersion": MCP_APPS_PROTOCOL_VERSION,  # new
        "html": html,
    }
    if resource_uri:
        payload["resourceUri"] = resource_uri
    # ...existing title/csp/permissions/prefersBorder/displayMode fields unchanged
    return f"event: mcp_app\ndata: {json.dumps(payload)}\n\n"
```

- [x] **Step 1:** Write failing tests: assert the emitted payload for both
      concrete wrappers now includes `"type": "mcp_app"` and
      `"protocolVersion": "2026-01-26"`; assert `resource_uri=None` (default)
      omits the `resourceUri` key, matching the existing `if csp:`/
      `if permissions:` minimal-payload pattern; assert every pre-existing
      field (`html`, `title`, `csp`, `permissions`, `prefersBorder`,
      `displayMode`) is unchanged in both shape and behavior.
- [x] **Step 2:** Implement in both concrete wrappers (and update the
      abstract base's signature/docstring for consistency, even though its
      body stays a no-op).
- [x] **Step 3:** Run the full test suite for this module
      (`pytest tests/structures/openai/request/ tests/converters/`) to catch
      any test relying on the old minimal payload shape.
- [x] **Step 4: Commit** `BAI-960 add type/protocolVersion discriminator to mcp_app SSE event`

---

### Task 2: Fix the stale migration doc — DONE

**Files:**
- Modified: `docs/migration-model-factory-and-reasoning.md` (section 6)

`docs/migration-model-factory-and-reasoning.md`'s MCP Apps section described
`mcp_client/ui_resource.py`/`MCPToolProvider.fetch_mcp_app_embed()` as living
in this repo (they were removed under BAI-730) and linked a `docs/mcp-apps.md`
that was never created. Corrected in place with a note pointing to
`baileyai`'s `adrs/010-mcp-apps-support.md`.

(Note: the two `baileyai`-repo spec docs this task originally named —
`docs/superpowers/specs/2026-09-10-mcp-image-embedded-resource-content-design.md`
and `2026-09-19-mcp-secondary-client-image-content-design.md` — live in
`baileyai`, not this repo; fixing those, if still needed, is out of scope
for this repo's PR.)

- [x] **Step 1:** Correct section 6 in place.
- [ ] **Step 2: Commit** (part of the main commit below)

---

## Non-Goals

- Everything in the original version of this plan (producer, `fetch_ui_resources`
  kwarg, retiring the env var) — superseded, see Correction above.

## Sequencing

No upstream dependency. Independent of BAI-961 (`baileyai-skills-service`)
and the `baileyai-chat-ui` renderer work — all three can proceed in
parallel; only the frontend's *testing* benefits from this shipping first
(it can otherwise test against a hand-written fixture payload).
