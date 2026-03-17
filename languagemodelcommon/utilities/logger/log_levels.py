__all__ = ["logger", "SRC_LOG_LEVELS"]

import os
import sys
from loguru import logger


GLOBAL_LOG_LEVEL = os.environ.get("LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL:
    logger.remove()
    logger.add(
        sys.stdout,
        level=GLOBAL_LOG_LEVEL,
        format="{time} {level} {name} [{file}:{line}] {message}",
    )
else:
    GLOBAL_LOG_LEVEL = "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        level=GLOBAL_LOG_LEVEL,
        format="{time} {level} {name} [{file}:{line}] {message}",
    )

logger.info(f"GLOBAL LOG_LEVEL: {GLOBAL_LOG_LEVEL}")

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
    logger.info(f"{log_env_var}: {SRC_LOG_LEVELS[source]}")
