import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from languagemodelcommon.utilities.config_substitution import substitute_env_vars
from languagemodelcommon.configs.schemas.config_schema import (
    AgentConfig,
    ChatModelConfig,
)
from languagemodelcommon.utilities.logger.log_levels import SRC_LOG_LEVELS

logger = logging.getLogger(__name__)
logger.setLevel(SRC_LOG_LEVELS.CONFIG)

MCP_JSON_PATH_ENV = (
    "MCP_JSON_PATH"  # Environment variable to override the path to .mcp.json
)

MCP_JSON_FILENAME = ".mcp.json"


class McpServerEntry(BaseModel):
    """A single server entry inside .mcp.json ``mcpServers``."""

    type: str | None = None
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: Dict[str, str] | None = None

    model_config = ConfigDict(extra="allow")


class McpJsonConfig(BaseModel):
    """Root model for the ``.mcp.json`` file."""

    mcpServers: Dict[str, McpServerEntry] = {}


def read_mcp_json(config_dir: str | None = None) -> McpJsonConfig | None:
    """Read and parse ``.mcp.json``.

    Resolution order for the file path:
    1. ``MCP_JSON_PATH`` environment variable (absolute path to the file).
    2. ``.mcp.json`` in *config_dir* (the model-configs directory).

    Returns ``None`` when no ``.mcp.json`` is found.
    """
    env_path = os.environ.get(MCP_JSON_PATH_ENV)

    if env_path:
        mcp_json_path = Path(env_path)
    elif config_dir:
        mcp_json_path = Path(config_dir) / MCP_JSON_FILENAME
    else:
        return None

    if not mcp_json_path.is_file():
        logger.debug(".mcp.json not found at %s", mcp_json_path)
        return None

    logger.info("Loading MCP server registry from %s", mcp_json_path)
    with open(mcp_json_path, "r") as f:
        data = substitute_env_vars(json.load(f))
    return McpJsonConfig(**data)


def resolve_mcp_servers(
    configs: List[ChatModelConfig],
    mcp_config: McpJsonConfig,
) -> None:
    """Resolve ``mcp_server`` references on every ``AgentConfig``.

    For each agent/tool that has ``mcp_server`` set, look up the key in
    *mcp_config* and populate ``url`` from the server entry.  The
    ``mcp_server`` value is left intact for traceability.

    If a referenced server key is not found, a warning is logged and the
    agent's existing ``url`` (if any) is left unchanged as a fallback.
    """
    servers = mcp_config.mcpServers

    for model in configs:
        agents: List[AgentConfig] = model.get_agents()
        for agent in agents:
            if not agent.mcp_server:
                continue
            entry = servers.get(agent.mcp_server)
            if entry is None:
                logger.warning(
                    "MCP server '%s' referenced by tool '%s' in model '%s' "
                    "not found in .mcp.json (available: %s). "
                    "Falling back to inline url '%s'.",
                    agent.mcp_server,
                    agent.name,
                    model.name,
                    list(servers.keys()),
                    agent.url,
                )
                continue
            if entry.url:
                agent.url = entry.url
                logger.info(
                    "Resolved mcp_server '%s' -> url '%s' for tool '%s' in model '%s'",
                    agent.mcp_server,
                    entry.url,
                    agent.name,
                    model.name,
                )
