from __future__ import annotations

import re
from dataclasses import dataclass


_SLASH_COMMAND_RE = re.compile(r"^/([a-zA-Z][a-zA-Z0-9_-]*)(?:\s+(.*))?$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SlashCommandMatch:
    """Result of detecting a /command prefix in user input."""

    command_name: str
    remaining_message: str


def parse_slash_command(*, content: str) -> SlashCommandMatch | None:
    if not content or not content.startswith("/"):
        return None

    match = _SLASH_COMMAND_RE.match(content)
    if not match:
        return None

    command_name = match.group(1)
    remaining = (match.group(2) or "").strip()

    return SlashCommandMatch(
        command_name=command_name,
        remaining_message=remaining,
    )
