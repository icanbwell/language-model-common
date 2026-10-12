# Logging

Last verified: 2026-10-12 against 8bc128d by Claude

How this repo logs, and what a log line must carry so it can be counted, filtered and explained.
Org-wide rules (never log PHI or credentials; single-line JSON for Groundcover; trace propagation)
live in the wiki's `standards/observability/` and the org logging standard; link them, do not
restate them. This repo is a library, so it logs through whatever handlers the host service
configures. Where to find the lines is in [MONITORING.md](MONITORING.md); what to do with a signal is
in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Logger and format

Standard library `logging`, one module-level `logger = logging.getLogger(__name__)` per module (for example `languagemodelcommon/mcp/mcp_client/session.py`). The library does not set a JSON formatter. `languagemodelcommon/utilities/logger/log_levels.py` installs a plain-text root handler to stdout only when the host has not configured any (format in [MONITORING.md](MONITORING.md)). Single-line JSON output is therefore the host's responsibility (`unverified` per consumer). Multi-line exception tracebacks are produced by `logger.exception` and `ExceptionLogger` (`languagemodelcommon/utilities/logger/exception_logger.py`).

## Levels

Per-source levels are set by `LOG_LEVEL` and `<SOURCE>_LOG_LEVEL` env vars; sources are listed in `log_levels.py` and a module picks its source with `logger.setLevel(SRC_LOG_LEVELS.<SOURCE>)` (example: `HTTP` in `languagemodelcommon/utilities/logger/logging_transport.py`). What each level means here: WARNING is used for handled retries and fallbacks (MCP session retry, task-protocol fallback, cache trouble); INFO for milestones; DEBUG for per-item detail such as HTTP request and response summaries. Whether ERROR is reserved for failed operations: `unverified` (no written rule in the repo). The one-ERROR-per-failure rule from the template: `unverified`; since this is a library, the host is expected to log the failed request once, and library layers re-raise rather than also logging (as `LoggingTransport` does with `logger.exception` before `raise`, which can produce a second line for one failure).

## Required fields

None defined. The library emits free-text messages with no required structured fields (`unverified` that no field is injected by a filter; none found in `languagemodelcommon/utilities/logger/`). Request ids, session ids and trace ids come from the host. Trace context is propagated to MCP servers by the tracing interceptor (`languagemodelcommon/mcp/interceptors/tracing.py`). Because no schema exists, no dashboard or monitor in this repo depends on a field name; a consumer's monitor may depend on message text, so renaming a message that [MONITORING.md](MONITORING.md) lists is a breaking change for consumers (`unverified` which consumers match on text).

## Per-request and per-failure lines

The library emits no per-request line. A single failure can write several lines: the MCP session retry writes one WARNING per failed attempt, then raises; `LoggingTransport` logs `logger.exception` for an HTTP error and re-raises, so the consumer may log it again. Count requests from the consumer's own per-request line, not from lines here.

## Latency and timing

No timing field is logged by the library. Timeouts and intervals that bound timing are env-driven: `TOOL_CALL_TIMEOUT_SECONDS`, `MCP_TOOL_HEARTBEAT_INTERVAL_SECONDS`, `STREAMING_BUFFER_FLUSH_INTERVAL_SECONDS` (`language_model_common_environment_variables.py`). MCP tool calls are timed by the OpenTelemetry span `mcp.tool.<name>` (`tracing.py`), not by a log line.

## Error classification

No bounded error-class set. Library exceptions are in `languagemodelcommon/exceptions/` (`bailey_exception.py`, `rate_limit_exception.py`) and `languagemodelcommon/mcp/exceptions/`; they are raised to the host, which classifies them. Whether a class name is logged as a field: `unverified`.

## Never log

Org rule: no PHI or credentials in logs (see `AGENTS.md`, Security). Repo specifics:

- Use `summarize_for_logging(value=...)` from `languagemodelcommon/utilities/logger/summarization.py`: it returns type, length and identifier-shaped key names, never values. Enforced by tests in `tests/utilities/test_summarize_for_logging.py` for the summarizer itself; its use at call sites is reviewer-checked.
- `LoggingTransport` logs header names only, and presence and length for `Authorization`, and a summarized body (`languagemodelcommon/utilities/logger/logging_transport.py`; test `tests/utilities/test_logging_transport.py`).
- Message and response bodies are logged only when `LOG_INPUT_AND_OUTPUT=1` (default off) or when debug output is enabled for a request through the `Debug-Mode` header or `CLIENT_IDS_FOR_DEBUG_OUTPUT` (`languagemodelcommon/utilities/logger/debug_configuration.py`, `language_model_common_environment_variables.py`). Do not enable these against real patient conversations without PHI handling in place.
- MCP tool-call arguments are set as span attribute `mcp.arguments`, and `tracing.py` logs `request.headers` as-is at DEBUG (no redaction in that line); treat both as sensitive and keep DEBUG for the `MCP` and `TOOLS` sources off outside local work.
- `none; reviewer only` for the call-site use of any of these.

Identifiers (person ids, patient ids, GUIDs) may appear in Groundcover logs as named fields; never write real values into any file.

## When you change logging

- Level fits the meaning above; a failure is logged at one layer.
- No PHI, tokens or full bodies; use `summarize_for_logging`.
- A message text that [MONITORING.md](MONITORING.md) lists is renamed only with that file updated in the same PR, and the consumers that match on it checked.
- A new signal responders should look for goes in [TROUBLESHOOTING.md](TROUBLESHOOTING.md); a new format, noise source or dependency goes in [MONITORING.md](MONITORING.md).
- Review tooling: no repo config or workflow invokes a logging reviewer or log auditor (grep of `origin/main` for `logging-reviewer` and `log-audit` finds nothing). Whether the `logging-reviewer` and `log-audit` skills are available to authors is `unverified`; do not treat them as a gate. Call-site log review is manual (see [REVIEW.md](REVIEW.md)).
