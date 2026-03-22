import logging
import os
import sys
from dataclasses import dataclass


def _resolve_log_level(raw_level: str | None, fallback_level: str) -> str:
    """Resolve a logger level name using only public logging APIs."""
    candidate = (raw_level or "").strip().upper()
    if not candidate:
        return fallback_level

    if candidate in logging.getLevelNamesMapping():
        return candidate

    if candidate.lstrip("+-").isdigit():
        resolved_name = logging.getLevelName(int(candidate))
        if isinstance(resolved_name, str) and not resolved_name.startswith("Level "):
            return resolved_name

    return fallback_level


GLOBAL_LOG_LEVEL = _resolve_log_level(os.environ.get("LOG_LEVEL"), "INFO")


def _configure_default_root_logging_if_needed(level_name: str) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    logging.basicConfig(
        stream=sys.stdout,
        level=level_name,
        format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
    )


_configure_default_root_logging_if_needed(GLOBAL_LOG_LEVEL)

log = logging.getLogger(__name__)
log.info(f"GLOBAL LOG_LEVEL: {GLOBAL_LOG_LEVEL}")


@dataclass
class _SourceLogLevels:
    """Container for source log levels exposed via dot notation only."""

    HTTP_TRACING: str
    CONFIG: str
    INITIALIZATION: str
    HTTP: str
    AUTH: str
    TOKEN_EXCHANGE: str
    DATABASE: str
    LLM: str
    FILES: str
    IMAGE_GENERATION: str
    IMAGE_PROCESSING: str
    MCP: str
    AGENTS: str
    ERRORS: str
    ROUTER: str
    SERVICES: str
    CACHE: str
    EVALUATOR: str
    RESPONSES: str
    TOOLS: str


LOG_SOURCES = tuple(_SourceLogLevels.__annotations__.keys())
_resolved_levels: dict[str, str] = {}

for source in LOG_SOURCES:
    log_env_var = source + "_LOG_LEVEL"
    level = _resolve_log_level(os.environ.get(log_env_var), GLOBAL_LOG_LEVEL)
    _resolved_levels[source] = level
    log.info(f"{log_env_var}: {level}")

SRC_LOG_LEVELS = _SourceLogLevels(**_resolved_levels)
