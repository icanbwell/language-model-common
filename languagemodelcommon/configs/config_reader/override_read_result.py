from dataclasses import dataclass
from enum import Enum

from languagemodelcommon.configs.schemas.config_schema import ChatModelConfig


class OverrideOutcome(str, Enum):
    """Outcome of reading one client's override config from its source.

    Only ``FOUND`` and ``ABSENT`` are safe to cache. ``LOCK_HELD`` (another
    worker is downloading) and ``FAILED`` (download or parse error) say nothing
    about whether an override exists.
    """

    FOUND = "found"
    ABSENT = "absent"
    LOCK_HELD = "lock_held"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class OverrideReadResult:
    outcome: OverrideOutcome
    models: tuple[ChatModelConfig, ...] = ()
    has_unresolved_mcp: bool = False


class DownloadLockHeldError(Exception):
    """Another worker holds the download lock, so nothing was read."""
