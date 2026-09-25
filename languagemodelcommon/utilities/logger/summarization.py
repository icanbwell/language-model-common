"""Redaction-safe summarization for values that would otherwise be logged raw.

BAI-955: this is the shared primitive `mcp-fhir-agent`, `baileyai`, and
`baileyai-skills-service` each take a contract dependency on, replacing their
own local (or absent) copies of this pattern. Modeled on
`baileyai-skills-service`'s `_safe_summarize_args()`
(`mcp_servers/middleware/audit_middleware.py`), generalized to recurse into
nested dicts so a JSON request/response body's shape stays inspectable
without ever surfacing a raw value.

NOTE on call signature: this repo's own style guide (AGENTS.md) mandates
keyword-only arguments for public functions. This function is a deliberate,
narrow exception - its signature is a cross-repo contract declared verbatim
in the observability-platform session doc, and three sibling repos are
already coding against `summarize_for_logging(value)` positionally as this
session runs. Changing it to keyword-only here would silently break those
callers. Positional-or-keyword keeps both call styles working.
"""

from typing import Any


def summarize_for_logging(value: Any) -> dict[str, Any]:
    """Redaction-safe stand-in for a raw value.

    Returns ``{"type": ..., "length": ...}`` for a scalar/bytes/str, or the
    same recursively per key for a dict - never the underlying value itself.

    For ``bytes``/``bytearray``/``memoryview``, ``length`` is the byte count
    (``len(value)``), not the length of its ``repr``/``str`` form - the repr
    of a bytes value (e.g. ``b'\\xc3\\xa9'``) can be up to 4x the actual byte
    count due to ``\\xNN`` escaping. For ``str``, ``length`` is ``len(value)``.
    Other scalars fall back to ``len(str(value))``.
    """
    if isinstance(value, dict):
        return {key: summarize_for_logging(item) for key, item in value.items()}

    if isinstance(value, (bytes, bytearray, memoryview)):
        length = len(value)
    elif isinstance(value, str):
        length = len(value)
    else:
        length = len(str(value))

    return {"type": type(value).__name__, "length": length}
