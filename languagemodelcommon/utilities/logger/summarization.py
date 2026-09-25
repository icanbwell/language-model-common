"""Redaction-safe summarization for values that would otherwise be logged raw.

BAI-955: this is the shared primitive `mcp-fhir-agent`, `baileyai`, and
`baileyai-skills-service` each take a contract dependency on, replacing their
own local (or absent) copies of this pattern. Modeled on
`baileyai-skills-service`'s `_safe_summarize_args()`
(`mcp_servers/middleware/audit_middleware.py`).

The result is always a flat, single-level dict describing the *shape* of a
value - never the value itself, and never any key that is not
identifier-shaped:

- ``dict``: ``{"type": "dict", "length": len(value), "keys": [...]}`` where
  ``keys`` is the sorted list of "safe" keys (``str`` and matching
  ``_SAFE_KEY``). There is no recursion into values - a nested dict's keys
  are never inspected. Any key that is not safe is not named, only counted
  via a ``"redacted_keys"`` entry (omitted when zero).
- ``list``/``tuple``/``set``/``frozenset``: ``{"type": ..., "length": ...}``
  with ``length`` as the element count. Elements are never inspected.
- ``bytes``/``bytearray``: ``length`` is the byte count (``len(value)``), not
  the length of its ``repr()``/``str()`` form - the repr of a bytes value
  (e.g. ``b'\\xc3\\xa9'``) can be far longer than the actual byte count due to
  ``\\xNN`` escaping. ``memoryview``: ``length`` is ``value.nbytes``.
- ``str``: ``length`` is ``len(value)``.
- ``str``/bytes-like content that parses as a JSON *object* (a dict) also
  gets the same ``keys``/``redacted_keys`` treatment as a dict, computed over
  the parsed object's keys - never over the raw text. Bytes are decoded as
  UTF-8 with ``errors="replace"`` for the parse attempt only; a parse
  failure (invalid UTF-8, invalid JSON, or JSON that isn't an object) adds
  nothing and never raises. A JSON *array* adds nothing beyond type/length.
- Any other value: ``{"type": type(value).__name__, "length": len(str(value))}``.

This function never raises and never includes a raw value or an unsafe key
name in its output.
"""

import json
import re
from typing import Any

_SAFE_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]{0,63}")


def _is_safe_key(key: Any) -> bool:
    return isinstance(key, str) and _SAFE_KEY.fullmatch(key) is not None


def _key_summary(mapping: dict[Any, Any]) -> dict[str, Any]:
    """Build the ``keys``/``redacted_keys`` portion for a dict-shaped value."""
    safe_keys = sorted(key for key in mapping if _is_safe_key(key))
    redacted_count = len(mapping) - len(safe_keys)

    summary: dict[str, Any] = {"keys": safe_keys}
    if redacted_count > 0:
        summary["redacted_keys"] = redacted_count
    return summary


def _parsed_json_object_key_summary(text: str) -> dict[str, Any] | None:
    """If `text` parses as a JSON object, return its key summary, else None.

    Never raises - any parse failure (invalid JSON, or valid JSON that isn't
    an object) results in ``None``.
    """
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None

    return _key_summary(parsed)


def _summarize(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "dict", "length": len(value), **_key_summary(value)}

    if isinstance(value, (list, tuple, set, frozenset)):
        return {"type": type(value).__name__, "length": len(value)}

    if isinstance(value, memoryview):
        return {"type": "memoryview", "length": value.nbytes}

    if isinstance(value, (bytes, bytearray)):
        result: dict[str, Any] = {"type": type(value).__name__, "length": len(value)}
        key_summary = _parsed_json_object_key_summary(
            bytes(value).decode("utf-8", errors="replace")
        )
        if key_summary is not None:
            result.update(key_summary)
        return result

    if isinstance(value, str):
        result = {"type": "str", "length": len(value)}
        key_summary = _parsed_json_object_key_summary(value)
        if key_summary is not None:
            result.update(key_summary)
        return result

    return {"type": type(value).__name__, "length": len(str(value))}


def summarize_for_logging(*, value: Any) -> dict[str, Any]:
    """Redaction-safe stand-in for a raw value. See module docstring for the shape.

    Never raises - any unexpected failure while inspecting ``value`` falls
    back to a bare type/length-less summary rather than propagating.
    """
    try:
        return _summarize(value)
    except Exception:
        return {"type": type(value).__name__, "length": 0}
