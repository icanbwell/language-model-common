import logging
import os
import sys

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

log_sources = [
    "HTTP_TRACING",
    "CONFIG",
    "INITIALIZATION",
    "HTTP",
    "AUTH",
    "TOKEN_EXCHANGE",
    "DATABASE",
    "LLM",
    "FILES",
    "IMAGE_GENERATION",
    "IMAGE_PROCESSING",
    "MCP",
    "AGENTS",
    "ERRORS",
    "ROUTER",
    "SERVICES",
    "CACHE",
    "EVALUATOR",
    "RESPONSES",
    "TOOLS",
]


class _SourceLogLevels(dict[str, str]):
    """Dictionary-backed container that also supports attribute access."""

    def __getattr__(self, name: str) -> str:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


SRC_LOG_LEVELS = _SourceLogLevels()

for source in log_sources:
    log_env_var = source + "_LOG_LEVEL"
    SRC_LOG_LEVELS[source] = os.environ.get(log_env_var, "").upper()
    if SRC_LOG_LEVELS[source] not in logging.getLevelNamesMapping():
        SRC_LOG_LEVELS[source] = GLOBAL_LOG_LEVEL
    log.info(f"{log_env_var}: {SRC_LOG_LEVELS[source]}")
