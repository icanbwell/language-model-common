import logging
import os
import sys
from dataclasses import dataclass

GLOBAL_LOG_LEVEL = os.environ.get("LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL in logging._nameToLevel:
    logging.basicConfig(
        stream=sys.stdout,
        level=GLOBAL_LOG_LEVEL,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s",
    )
else:
    GLOBAL_LOG_LEVEL = "INFO"

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
    level = os.environ.get(log_env_var, "").upper()
    if level not in logging.getLevelNamesMapping():
        level = GLOBAL_LOG_LEVEL
    _resolved_levels[source] = level
    log.info(f"{log_env_var}: {level}")

SRC_LOG_LEVELS = _SourceLogLevels(**_resolved_levels)
