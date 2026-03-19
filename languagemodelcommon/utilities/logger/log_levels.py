__all__ = ["logger", "SRC_LOG_LEVELS"]

import logging
import os
import sys


GLOBAL_LOG_LEVEL = os.environ.get("LOG_LEVEL", "").upper()
if not GLOBAL_LOG_LEVEL:
    GLOBAL_LOG_LEVEL = "INFO"

logger = logging.getLogger("languagemodelcommon")
logger.handlers.clear()
logger.propagate = False
logger.setLevel(getattr(logging, GLOBAL_LOG_LEVEL, logging.INFO))

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(getattr(logging, GLOBAL_LOG_LEVEL, logging.INFO))
handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
    )
)
logger.addHandler(handler)

logger.info("GLOBAL LOG_LEVEL: %s", GLOBAL_LOG_LEVEL)

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
]

SRC_LOG_LEVELS = {}

for source in log_sources:
    log_env_var = source + "_LOG_LEVEL"
    src_level = os.environ.get(log_env_var, "").upper()
    if not src_level:
        src_level = GLOBAL_LOG_LEVEL
    SRC_LOG_LEVELS[source] = src_level
    logger.info("%s: %s", log_env_var, SRC_LOG_LEVELS[source])
