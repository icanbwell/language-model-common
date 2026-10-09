"""MCP protocol negotiation mode (BAI-1118, ADR 0003).

Kept in its own dependency-free module so configuration code (environment
variables) can name the mode without importing the MCP client stack.
"""

from enum import StrEnum


class McpProtocolNegotiationMode(StrEnum):
    """How a client session picks its MCP protocol era.

    ``LEGACY`` runs the ``initialize`` handshake only. ``AUTO`` probes
    ``server/discover`` first and falls back to ``initialize`` when the server
    does not answer it.
    """

    LEGACY = "legacy"
    AUTO = "auto"
